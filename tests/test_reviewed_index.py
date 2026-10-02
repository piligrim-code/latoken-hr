"""Review and bounded input checks, without optional provider packages."""
import builtins
import hashlib
import importlib
import json
from pathlib import Path

import pytest

import reviewed_index as index


def corpus(tmp_path, texts=None):
    directory = tmp_path / "reviewed"
    directory.mkdir(exist_ok=True)
    entries = []
    for name, text in (texts or {"a.txt": "Synthetic approved text."}).items():
        raw = text.encode("utf-8")
        (directory / name).write_bytes(raw)
        entries.append({"file": name, "sha256": hashlib.sha256(raw).hexdigest(),
                        "source": "Synthetic fixture", "approved_for_external_embedding": True})
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"schema": 1, "documents": entries}), encoding="utf-8")
    return directory, path


def test_import_is_inert(monkeypatch, tmp_path):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith(("chromadb", "openai", "httpx", "dotenv", "langchain", "filelock")):
            pytest.fail("Provider imported at module initialization")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.chdir(tmp_path)
    importlib.reload(index)
    assert list(tmp_path.iterdir()) == []


def test_review_returns_content_and_portable_provenance(tmp_path):
    directory, review = corpus(tmp_path, {"b.txt": "Second", "a.txt": "First"})
    docs = index.load_reviewed_documents(directory, review)
    assert [d.name for d in docs] == ["a.txt", "b.txt"]
    assert [d.text for d in docs] == ["First", "Second"]
    chunks = index.make_chunks(docs)
    assert chunks == index.make_chunks(docs)
    assert all(str(tmp_path) not in repr(c) for c in chunks)


@pytest.mark.parametrize("update", [
    {"file": "../a.txt"}, {"file": "sub/a.txt"}, {"file": "C:a.txt"},
    {"file": "sub\\a.txt"}, {"file": "a.txt\n"}, {"file": ".txt"},
    {"sha256": "bad"}, {"sha256": None}, {"source": ""}, {"source": "x" * 161},
    {"source": "private\npath"}, {"approved_for_external_embedding": False},
    {"approved_for_external_embedding": 1},
])
def test_invalid_review_fails(tmp_path, update):
    directory, review = corpus(tmp_path)
    value = json.loads(review.read_text())
    value["documents"][0].update(update)
    review.write_text(json.dumps(value))
    with pytest.raises(index.IndexError):
        index.load_reviewed_documents(directory, review)


@pytest.mark.parametrize("value", [[], {}, {"schema": True, "documents": []},
    {"schema": 1, "documents": [None]}, {"schema": 1, "documents": {}},
    {"schema": 2, "documents": []}])
def test_invalid_manifest_shape(tmp_path, value):
    directory, review = corpus(tmp_path)
    review.write_text(json.dumps(value))
    with pytest.raises(index.IndexError):
        index.load_reviewed_documents(directory, review)


@pytest.mark.parametrize("mutation", ["changed", "added", "removed", "duplicate", "case"])
def test_review_detects_source_drift(tmp_path, mutation):
    directory, review = corpus(tmp_path)
    if mutation == "changed":
        (directory / "a.txt").write_text("Changed")
    elif mutation == "added":
        (directory / "b.txt").write_text("Unreviewed")
    elif mutation == "removed":
        (directory / "a.txt").unlink()
    else:
        value = json.loads(review.read_text())
        extra = dict(value["documents"][0])
        if mutation == "case":
            extra["file"] = "A.txt"
        value["documents"].append(extra)
        review.write_text(json.dumps(value))
    with pytest.raises(index.IndexError):
        index.load_reviewed_documents(directory, review)


def test_bundled_corpus_refused_before_any_read(monkeypatch):
    monkeypatch.setattr(index, "_read_json", lambda *args: pytest.fail("Read bundled data"))
    with pytest.raises(index.IndexError, match="historical"):
        index.load_reviewed_documents(Path(index.__file__).parent / "data", "unused.json")


@pytest.mark.parametrize("raw", [b"", b"   ", b"a\x00b", b"\xff", b"a" * (index.MAX_FILE_BYTES + 1)],
                         ids=["empty", "whitespace", "nul", "invalid-utf8", "oversized"])
def test_bad_text_refused(tmp_path, raw):
    directory, review = corpus(tmp_path)
    (directory / "a.txt").write_bytes(raw)
    value = json.loads(review.read_text())
    value["documents"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    review.write_text(json.dumps(value))
    with pytest.raises(index.IndexError):
        index.load_reviewed_documents(directory, review)


def test_regular_file_required(tmp_path):
    directory, review = corpus(tmp_path)
    (directory / "a.txt").unlink()
    (directory / "a.txt").mkdir()
    with pytest.raises(index.IndexError, match="regular"):
        index.load_reviewed_documents(directory, review)


def test_symlink_rejected_without_requiring_os_privilege(tmp_path, monkeypatch):
    directory, review = corpus(tmp_path)
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p.name == "a.txt" or original(p))
    with pytest.raises(index.IndexError, match="non-symlink"):
        index.load_reviewed_documents(directory, review)


def test_chunks_have_bounded_size_overlap_and_content_versions(tmp_path):
    directory, review = corpus(tmp_path, {"a.txt": "a" * 1601})
    docs = index.load_reviewed_documents(directory, review)
    chunks = index.make_chunks(docs)
    assert [len(c.text) for c in chunks] == [800, 800, 161]
    assert len({c.id for c in chunks}) == 3
    assert [c.number for c in chunks] == [0, 1, 2]
    directory, review = corpus(tmp_path, {"a.txt": "b" * 1601})
    changed = index.make_chunks(index.load_reviewed_documents(directory, review))
    assert not {c.id for c in chunks} & {c.id for c in changed}


@pytest.mark.parametrize("vectors,count,dimension", [
    ([], 1, None), ([[0, 0]], 1, None), ([[True]], 1, None),
    ([[float("nan")]], 1, None), ([[float("inf")]], 1, None),
    ([["1"]], 1, None), ([[1], [1, 2]], 2, None), ([[1]], 1, 2),
    ([[1] * 4097], 1, None), ([[]], 1, None), ([[1]], 2, None),
])
def test_invalid_embedding_contract(vectors, count, dimension):
    with pytest.raises(index.IndexError):
        index.validate_vectors(vectors, count, dimension)


def test_embedding_contract_accepts_finite_vectors():
    assert index.validate_vectors([[1.0, 0], [0, 2]], 2) == 2


@pytest.mark.parametrize("limit,value", [("MAX_FILE_BYTES", 3), ("MAX_TOTAL_BYTES", 3),
                                        ("MAX_DOCUMENTS", 0)])
def test_corpus_limits(tmp_path, monkeypatch, limit, value):
    directory, review = corpus(tmp_path)
    monkeypatch.setattr(index, limit, value)
    with pytest.raises(index.IndexError):
        index.load_reviewed_documents(directory, review)


def test_chunk_limit(tmp_path, monkeypatch):
    directory, review = corpus(tmp_path)
    documents = index.load_reviewed_documents(directory, review)
    monkeypatch.setattr(index, "MAX_CHUNKS", 0)
    with pytest.raises(index.IndexError, match="chunk limit"):
        index.make_chunks(documents)


@pytest.mark.parametrize("payload", [b"not json", b"\xff", b" " * (128 * 1024 + 1)],
                         ids=["not-json", "bad-encoding", "oversized"])
def test_invalid_or_oversized_manifest(tmp_path, payload):
    directory, review = corpus(tmp_path)
    review.write_bytes(payload)
    with pytest.raises(index.IndexError):
        index.load_reviewed_documents(directory, review)


def test_consent_and_review_required_before_loading(monkeypatch):
    monkeypatch.setattr(index, "load_reviewed_documents", lambda *a: pytest.fail("read without consent"))
    for args in ([], ["--data-path", "x", "--index-path", "y", "--review-manifest", "z"]):
        with pytest.raises(SystemExit) as exc:
            index.main(args)
        assert exc.value.code == 2


def test_cli_does_not_echo_provider_or_content_errors(monkeypatch, capsys):
    monkeypatch.setattr(index, "load_reviewed_documents", lambda *args: [])
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic private content that must not be logged")
    monkeypatch.setattr(index, "rebuild", fail)
    with pytest.raises(SystemExit) as exc:
        index.main(["--data-path", "x", "--index-path", "y", "--review-manifest", "z",
                    "--allow-external-embeddings"])
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert "RuntimeError" in output.err
    assert "synthetic private" not in output.err + output.out
