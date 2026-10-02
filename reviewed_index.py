"""Explicit, reviewed-corpus snapshots. No clients or files are opened on import."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

SCHEMA = 1
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MAX_DOCUMENTS = 200
MAX_CHUNKS = 3000
BATCH_SIZE = 16
DEFAULT_MODEL = "text-embedding-ada-002"
OWNER = {"application": "latoken-reviewed-index", "schema": SCHEMA}


class IndexError(ValueError):
    """Invalid review, provider response or local index state."""


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":")).encode("utf-8")


def _read_regular(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise IndexError("Expected a regular, non-symlink file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise IndexError("Input exceeds its byte limit")
    return data


def _read_json(path):
    try:
        return json.loads(_read_regular(path, 128 * 1024).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IndexError("Invalid UTF-8 JSON") from exc


def _atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Document:
    name: str
    source: str
    sha256: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    file: str
    number: int


def load_reviewed_documents(data_path, review_path):
    """Review is an operator assertion, not proof of third-party rights."""
    directory = Path(data_path)
    bundled = Path(__file__).resolve().parent / "data"
    resolved = directory.resolve()
    if resolved == bundled or bundled in resolved.parents:
        raise IndexError("The historical bundled corpus is not an approved input")
    if directory.is_symlink() or not directory.is_dir():
        raise IndexError("Data directory must be a regular non-symlink directory")
    review = _read_json(review_path)
    if not isinstance(review, dict) or type(review.get("schema")) is not int or review["schema"] != SCHEMA:
        raise IndexError("Unsupported review schema")
    entries = review.get("documents")
    if not isinstance(entries, list) or len(entries) > MAX_DOCUMENTS:
        raise IndexError("Invalid review document list")
    validated = {}
    for item in entries:
        if not isinstance(item, dict):
            raise IndexError("Review entries must be objects")
        name, sha, source = (item.get(k) for k in ("file", "sha256", "source"))
        if (not isinstance(name, str) or not name.endswith(".txt") or
                any(c in name for c in '/\\:') or name in (".txt", "..txt") or
                len(name) > 200 or any(ord(c) < 32 for c in name)):
            raise IndexError("Reviewed files must be plain .txt basenames")
        if name.casefold() in validated:
            raise IndexError("Duplicate or case-ambiguous reviewed file")
        if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise IndexError("Each reviewed file needs a SHA-256")
        if (not isinstance(source, str) or not source.strip() or len(source) > 160
                or any(ord(c) < 32 for c in source)):
            raise IndexError("Each reviewed file needs a short public source label")
        if item.get("approved_for_external_embedding") is not True:
            raise IndexError("Each file must be approved for external embedding")
        validated[name.casefold()] = item
    actual = {p.name for p in directory.iterdir() if p.suffix.lower() == ".txt"}
    expected = {item["file"] for item in validated.values()}
    if actual != expected:
        raise IndexError("Text files and review manifest must match exactly")
    documents, total = [], 0
    for item in sorted(validated.values(), key=lambda i: i["file"]):
        raw = _read_regular(directory / item["file"], MAX_FILE_BYTES)
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise IndexError("Corpus exceeds total byte limit")
        if _digest(raw) != item["sha256"]:
            raise IndexError("Reviewed content hash no longer matches")
        try:
            text = raw.decode("utf-8")
        except UnicodeError as exc:
            raise IndexError("Text must be UTF-8") from exc
        if not text.strip() or "\x00" in text:
            raise IndexError("Empty or binary-like text is not supported")
        documents.append(Document(item["file"], item["source"], item["sha256"], text))
    return documents


def make_chunks(documents):
    chunks = []
    for document in documents:
        for number, offset in enumerate(range(0, len(document.text), CHUNK_SIZE - CHUNK_OVERLAP)):
            text = document.text[offset:offset + CHUNK_SIZE]
            identity = [document.name, document.sha256, number, text]
            chunks.append(Chunk(_digest(_json_bytes(identity)), text, document.source,
                                document.name, number))
            if len(chunks) > MAX_CHUNKS:
                raise IndexError("Corpus exceeds chunk limit")
            if offset + CHUNK_SIZE >= len(document.text):
                break
    return chunks


def _model(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,100}", value):
        raise IndexError("Invalid embedding model identifier")
    return value


def validate_vectors(vectors, count, dimension=None):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise IndexError("Embedding response count mismatch")
    for vector in vectors:
        if not isinstance(vector, list) or not 1 <= len(vector) <= 4096:
            raise IndexError("Invalid embedding dimension")
        dimension = dimension or len(vector)
        if len(vector) != dimension:
            raise IndexError("Embedding dimensions differ")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
            raise IndexError("Embedding contains invalid numbers")
        if not any(v != 0 for v in vector):
            raise IndexError("Zero embeddings are not supported")
    return dimension


class OpenAIEmbedder:
    """Owns one explicit SDK/HTTP client. No retries and no implicit .env loading."""

    def __init__(self, api_key, model=DEFAULT_MODEL, *,
                 base_url="https://api.openai.com/v1", timeout=30):
        if not isinstance(api_key, str) or not api_key.strip():
            raise IndexError("OPENAI_API_KEY is required")
        self.model = _model(model)
        if type(timeout) not in (int, float) or not 0 < timeout <= 120:
            raise IndexError("Invalid provider timeout")
        import httpx
        from openai import OpenAI
        http = httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False)
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout,
                                 max_retries=0, http_client=http)
        except BaseException:
            http.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def embed(self, texts):
        if (not isinstance(texts, list) or not 1 <= len(texts) <= BATCH_SIZE or
                any(not isinstance(t, str) or not t.strip() or len(t) > CHUNK_SIZE for t in texts)):
            raise IndexError("Invalid embedding batch")
        response = self.client.embeddings.create(model=self.model, input=texts,
                                                  encoding_format="float")
        if response.model != self.model:
            raise IndexError("Embedding response model mismatch")
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise IndexError("Embedding response indexes mismatch")
        vectors = [item.embedding for item in ordered]
        validate_vectors(vectors, len(texts))
        return vectors


def _client(path):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(path / "chroma"),
                                    settings=Settings(anonymized_telemetry=False,
                                                      allow_reset=False))


def _owned(path):
    if path.is_symlink() or not path.is_dir():
        raise IndexError("Index directory must be a regular directory")
    if _read_json(path / "owner.json") != OWNER:
        raise IndexError("Unrecognized index owner marker")
    if (path / "chroma").is_symlink():
        raise IndexError("Index database must not be a symlink")


@contextmanager
def _writer(path):
    from filelock import FileLock, Timeout
    if path.is_symlink():
        raise IndexError("Index directory must not be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    lock_path = path / ".writer.lock"
    if lock_path.is_symlink():
        raise IndexError("Index lock must not be a symlink")
    try:
        with FileLock(str(lock_path), timeout=0):
            if not (path / "owner.json").exists():
                if any(p.name != ".writer.lock" for p in path.iterdir()):
                    raise IndexError("Refusing to adopt a nonempty, unmanaged index")
                _atomic_json(path / "owner.json", OWNER)
            _owned(path)
            yield
    except Timeout as exc:
        raise IndexError("Another index writer is active") from exc


def _manifest(path):
    value = _read_json(path / "active.json")
    if not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != SCHEMA:
        raise IndexError("Unsupported active index schema")
    if not re.fullmatch(r"reviewed_[a-f0-9]{32}", str(value.get("collection", ""))):
        raise IndexError("Invalid active collection")
    if not re.fullmatch(r"[a-f0-9]{64}", str(value.get("fingerprint", ""))):
        raise IndexError("Invalid corpus fingerprint")
    _model(value.get("model"))
    count, dimension = value.get("count"), value.get("dimension")
    if type(count) is not int or not 0 <= count <= MAX_CHUNKS:
        raise IndexError("Invalid active chunk count")
    if type(dimension) is not int or not (1 <= dimension <= 4096 if count else dimension == 0):
        raise IndexError("Invalid active embedding dimension")
    return value


def _collection(client, manifest):
    collection = client.get_collection(manifest["collection"], embedding_function=None)
    expected = {k: manifest[k] for k in ("schema", "model", "fingerprint")}
    if collection.metadata != expected or collection.count() != manifest["count"]:
        raise IndexError("Active collection does not match its manifest")
    return collection


def rebuild(documents, index_path, embedder_factory, *, model=DEFAULT_MODEL, allow_empty=False):
    """Publish a complete generation; partial batches never replace the active one."""
    model = _model(model)
    documents = sorted(documents, key=lambda d: d.name)
    if (len(documents) > MAX_DOCUMENTS or len({d.name.casefold() for d in documents}) != len(documents)
            or sum(len(d.text.encode("utf-8")) for d in documents) > MAX_TOTAL_BYTES):
        raise IndexError("Invalid document set")
    for document in documents:
        raw = document.text.encode("utf-8")
        if not document.text.strip() or len(raw) > MAX_FILE_BYTES or _digest(raw) != document.sha256:
            raise IndexError("Document text does not match its reviewed hash or limits")
    if not documents and not allow_empty:
        raise IndexError("Empty replacement requires explicit allow_empty")
    chunks = make_chunks(documents)
    fingerprint = _digest(_json_bytes({
        "schema": SCHEMA, "model": model, "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "documents": [[d.name, d.source, d.sha256] for d in documents],
    }))
    path = Path(index_path)
    with _writer(path):
        client = _client(path)
        if (path / "active.json").exists():
            previous = _manifest(path)
            _collection(client, previous)
            if previous["fingerprint"] == fingerprint:
                return {**previous, "changed": False}
        manifest = {"schema": SCHEMA, "collection": "reviewed_" + uuid.uuid4().hex,
                    "model": model, "fingerprint": fingerprint, "count": len(chunks), "dimension": 0}
        candidate = client.create_collection(manifest["collection"], embedding_function=None,
                    metadata={k: manifest[k] for k in ("schema", "model", "fingerprint")},
                    configuration={"hnsw": {"space": "cosine"}})
        if chunks:
            with embedder_factory() as embedder:
                if embedder.model != model:
                    raise IndexError("Configured embedding model mismatch")
                for start in range(0, len(chunks), BATCH_SIZE):
                    batch = chunks[start:start + BATCH_SIZE]
                    vectors = embedder.embed([c.text for c in batch])
                    manifest["dimension"] = validate_vectors(vectors, len(batch), manifest["dimension"])
                    candidate.add(ids=[c.id for c in batch], documents=[c.text for c in batch],
                                  embeddings=vectors,
                                  metadatas=[{"source": c.source, "file": c.file, "chunk": c.number}
                                             for c in batch])
        _collection(client, manifest)
        # Never delete generations in error handling: publication may have committed
        # immediately before interruption. Inactive orphans are an offline maintenance task.
        _atomic_json(path / "active.json", manifest)
        return {**manifest, "changed": True}


class Snapshot:
    """A pinned generation. Retained generations keep readers valid across updates."""

    def __init__(self, index_path):
        path = Path(index_path)
        _owned(path)
        self.manifest = _manifest(path)
        self.client = _client(path)
        self.collection = _collection(self.client, self.manifest)

    def search(self, vector, *, model, limit=5):
        if model != self.manifest["model"]:
            raise IndexError("Query and index embedding models differ")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise IndexError("Query limit must be between 1 and 10")
        if not self.manifest["count"]:
            return []
        validate_vectors([vector], 1, self.manifest["dimension"])
        result = self.collection.query(query_embeddings=[vector],
                                       n_results=min(limit, self.manifest["count"]),
                                       include=["documents", "metadatas", "distances"])
        return [dict(id=id_, text=text, source=meta["source"], file=meta["file"], distance=distance)
                for id_, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                                    result["metadatas"][0], result["distances"][0], strict=True)]


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-path", required=True)
    cli.add_argument("--review-manifest", required=True)
    cli.add_argument("--index-path", required=True)
    cli.add_argument("--embedding-model", default=DEFAULT_MODEL)
    cli.add_argument("--allow-external-embeddings", action="store_true")
    cli.add_argument("--allow-empty", action="store_true")
    args = cli.parse_args(argv)
    if not args.allow_external_embeddings:
        cli.error("Explicit --allow-external-embeddings is required after data review")
    try:
        documents = load_reviewed_documents(args.data_path, args.review_manifest)
        result = rebuild(documents, args.index_path,
                         lambda: OpenAIEmbedder(os.environ.get("OPENAI_API_KEY"), args.embedding_model),
                         model=args.embedding_model, allow_empty=args.allow_empty)
    except Exception as exc:
        # Provider exceptions can contain request text; do not echo them to the terminal.
        cli.exit(1, f"Index update failed ({type(exc).__name__}); active snapshot was not replaced.\n")
    print(json.dumps({k: result[k] for k in ("changed", "count", "dimension", "model")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
