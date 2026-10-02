"""Real Chroma + loopback OpenAI-compatible HTTP; no model calls/downloads."""
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("openai")
from filelock import FileLock

import reviewed_index as index


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    original = socket.socket.connect
    def connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
            pytest.fail("External network access is forbidden in index tests")
        return original(sock, address)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    def forbidden(*args, **kwargs):
        pytest.fail("Default embedding/model download attempted")
    monkeypatch.setattr(DefaultEmbeddingFunction, "__call__", forbidden)


def doc(name="a.txt", text="Synthetic fact", source="Synthetic source"):
    return index.Document(name, source, hashlib.sha256(text.encode()).hexdigest(), text)


@contextmanager
def provider(*, status=200, transform=None, fail_after=None, delay=0):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, request))
            time.sleep(delay)
            code = 503 if fail_after is not None and len(calls) > fail_after else status
            if code == 200:
                value = {"object": "list", "model": request["model"],
                         "data": [{"object": "embedding", "index": i,
                                   "embedding": [float(len(text)), 1.0, 2.0]}
                                  for i, text in enumerate(request["input"])],
                         "usage": {"prompt_tokens": 1, "total_tokens": 1}}
                if transform:
                    value = transform(value)
            else:
                value = {"error": {"message": "synthetic private upstream detail", "type": "test"}}
            raw = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def factory(url, *, model=index.DEFAULT_MODEL, clients=None):
    def make():
        client = index.OpenAIEmbedder("synthetic-test-key", model, base_url=url)
        if clients is not None:
            clients.append(client)
        return client
    return make


def no_provider():
    pytest.fail("Unnecessary embedding client construction")


def results(path, model=index.DEFAULT_MODEL):
    return index.Snapshot(path).search([1, 1, 2], model=model, limit=10)


def test_build_query_reopen_and_noop(tmp_path):
    path = tmp_path / "index"
    clients = []
    with provider() as (url, calls):
        published = index.rebuild([doc()], path, factory(url, clients=clients))
        assert published["count"] == 1 and published["dimension"] == 3
        assert clients[0].client.is_closed()
        assert calls[0][0] == "/v1/embeddings"
        assert calls[0][1] == {"input": ["Synthetic fact"], "model": index.DEFAULT_MODEL,
                               "encoding_format": "float"}
        assert len(calls) == 1
    found = results(path)
    assert len(found) == 1 and found[0]["text"] == "Synthetic fact"
    assert found[0]["source"] == "Synthetic source" and found[0]["file"] == "a.txt"
    assert str(tmp_path) not in json.dumps(found)
    original = (path / "active.json").read_bytes()
    unchanged = index.rebuild([doc()], path, no_provider)
    assert unchanged["changed"] is False
    assert (path / "active.json").read_bytes() == original


def test_content_update_delete_and_pinned_readers(tmp_path):
    with provider() as (url, calls):
        index.rebuild([doc(), doc("b.txt", "Remove this")], tmp_path, factory(url))
        old = index.Snapshot(tmp_path)
        index.rebuild([doc(text="Updated synthetic fact")], tmp_path, factory(url))
        current = results(tmp_path)
        assert [r["text"] for r in current] == ["Updated synthetic fact"]
        assert {r["text"] for r in old.search([1, 1, 2], model=index.DEFAULT_MODEL)} == {
            "Synthetic fact", "Remove this"}
        assert len(calls) == 2
        assert current[0]["id"] not in {r["id"] for r in old.search([1, 1, 2], model=index.DEFAULT_MODEL)}


def test_source_label_change_and_model_change_trigger_new_generation(tmp_path):
    with provider() as (url, calls):
        first = index.rebuild([doc()], tmp_path, factory(url))
        second = index.rebuild([doc(source="New reviewed source")], tmp_path, factory(url))
        third = index.rebuild([doc(source="New reviewed source")], tmp_path,
                              factory(url, model="synthetic-model"), model="synthetic-model")
    assert len({s["collection"] for s in (first, second, third)}) == 3
    assert len(calls) == 3
    with pytest.raises(index.IndexError, match="models differ"):
        results(tmp_path)
    assert results(tmp_path, "synthetic-model")[0]["source"] == "New reviewed source"


@pytest.mark.parametrize("stage", ["provider", "publish", "collection_add"])
def test_partial_failure_keeps_previous_snapshot(tmp_path, monkeypatch, stage):
    clients = []
    with provider() as (url, _):
        index.rebuild([doc()], tmp_path, factory(url))
    previous = (tmp_path / "active.json").read_bytes()
    if stage == "publish":
        monkeypatch.setattr(index, "_atomic_json", lambda *a: (_ for _ in ()).throw(OSError("write failed")))
    elif stage == "collection_add":
        from chromadb.api.models.Collection import Collection
        monkeypatch.setattr(Collection, "add", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("add failed")))
    with provider(fail_after=1 if stage == "provider" else None) as (url, calls):
        with pytest.raises(Exception):
            index.rebuild([doc(text="b" * 20000)], tmp_path, factory(url, clients=clients))
        assert len(calls) == (2 if stage in ("provider", "publish") else 1)
        assert clients[0].client.is_closed()
    assert (tmp_path / "active.json").read_bytes() == previous
    assert [r["text"] for r in results(tmp_path)] == ["Synthetic fact"]


def test_failure_before_first_publish_can_be_retried(tmp_path):
    with provider(status=503) as (url, calls):
        with pytest.raises(Exception):
            index.rebuild([doc()], tmp_path, factory(url))
        assert len(calls) == 1
    assert not (tmp_path / "active.json").exists()
    with provider() as (url, _):
        assert index.rebuild([doc()], tmp_path, factory(url))["changed"] is True
    assert results(tmp_path)[0]["text"] == "Synthetic fact"


def test_empty_replacement_is_explicit_and_needs_no_provider(tmp_path):
    with provider() as (url, _):
        index.rebuild([doc()], tmp_path, factory(url))
    previous = (tmp_path / "active.json").read_bytes()
    with pytest.raises(index.IndexError, match="allow_empty"):
        index.rebuild([], tmp_path, no_provider)
    assert (tmp_path / "active.json").read_bytes() == previous
    result = index.rebuild([], tmp_path, no_provider, allow_empty=True)
    assert result["count"] == result["dimension"] == 0
    assert results(tmp_path) == []
    assert index.rebuild([], tmp_path, no_provider, allow_empty=True)["changed"] is False


def test_writer_lock_prevents_concurrent_updates(tmp_path):
    with FileLock(str(tmp_path / ".writer.lock")):
        with pytest.raises(index.IndexError, match="writer"):
            index.rebuild([doc()], tmp_path, no_provider)
    assert not (tmp_path / "owner.json").exists()


def test_unmanaged_index_not_adopted(tmp_path):
    (tmp_path / "chroma.sqlite3").write_text("historical placeholder")
    with pytest.raises(index.IndexError, match="unmanaged"):
        index.rebuild([doc()], tmp_path, no_provider)
    assert (tmp_path / "chroma.sqlite3").read_text() == "historical placeholder"


def test_bad_document_hash_fails_before_index_creation(tmp_path):
    changed = index.Document("a.txt", "Synthetic", "0" * 64, "Actual text")
    with pytest.raises(index.IndexError, match="hash"):
        index.rebuild([changed], tmp_path / "not-created", no_provider)
    assert not (tmp_path / "not-created").exists()


def test_reordered_documents_are_noop(tmp_path):
    with provider() as (url, _):
        index.rebuild([doc(), doc("b.txt", "B")], tmp_path, factory(url))
    assert index.rebuild([doc("b.txt", "B"), doc()], tmp_path, no_provider)["changed"] is False


def test_corrupt_collection_not_silently_replaced(tmp_path):
    with provider() as (url, _):
        index.rebuild([doc()], tmp_path, factory(url))
    snapshot = index.Snapshot(tmp_path)
    snapshot.collection.delete(ids=snapshot.collection.get()["ids"])
    with pytest.raises(index.IndexError, match="manifest"):
        index.rebuild([doc()], tmp_path, no_provider)


@pytest.mark.parametrize("change", [
    {"count": True}, {"dimension": False}, {"collection": "other"}, {"fingerprint": "bad"},
    {"count": -1}, {"model": ""}, {"schema": True},
])
def test_bad_active_manifest_refused(tmp_path, change):
    with provider() as (url, _):
        index.rebuild([doc()], tmp_path, factory(url))
    value = json.loads((tmp_path / "active.json").read_text())
    value.update(change)
    (tmp_path / "active.json").write_text(json.dumps(value))
    with pytest.raises(index.IndexError):
        index.Snapshot(tmp_path)


@pytest.mark.parametrize("transform", [
    lambda r: {**r, "model": "wrong"},
    lambda r: {**r, "data": []},
    lambda r: {**r, "data": [{**r["data"][0], "index": 2}]},
    lambda r: {**r, "data": [{**r["data"][0], "embedding": [0, 0, 0]}]},
])
def test_sdk_validates_upstream_contract(transform):
    with provider(transform=transform) as (url, calls):
        with index.OpenAIEmbedder("synthetic", base_url=url) as embedder:
            with pytest.raises(index.IndexError):
                embedder.embed(["Synthetic"])
        assert embedder.client.is_closed()
        assert len(calls) == 1


def test_sdk_reorders_indexed_vectors():
    with provider(transform=lambda r: {**r, "data": list(reversed(r["data"]))}) as (url, _):
        with index.OpenAIEmbedder("synthetic", base_url=url) as embedder:
            assert embedder.embed(["A", "BBBB"]) == [[1, 1, 2], [4, 1, 2]]


@pytest.mark.parametrize("status", [401, 429, 503])
def test_sdk_error_is_not_retried_and_client_closes(status):
    with provider(status=status) as (url, calls):
        with pytest.raises(Exception):
            with index.OpenAIEmbedder("synthetic", base_url=url) as embedder:
                embedder.embed(["Synthetic"])
        assert len(calls) == 1
        assert embedder.client.is_closed()


def test_fresh_process_reads_persisted_snapshot(tmp_path):
    with provider() as (url, _):
        index.rebuild([doc()], tmp_path, factory(url))
    code = """
import socket, sys
def denied(*a, **k):
    raise AssertionError('No network permitted')
socket.socket.connect = denied
from reviewed_index import Snapshot, DEFAULT_MODEL
result = Snapshot(sys.argv[1]).search([1, 1, 2], model=DEFAULT_MODEL)
assert result[0]['text'] == 'Synthetic fact'
print('persisted snapshot verified')
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_")}
    run = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                         cwd=Path(__file__).resolve().parents[1], env=env,
                         capture_output=True, text=True, timeout=45)
    assert run.returncode == 0, run.stderr
    assert "persisted snapshot verified" in run.stdout


def test_sdk_read_timeout_closes_without_retry():
    from openai import APITimeoutError
    with provider(delay=0.3) as (url, calls):
        with pytest.raises(APITimeoutError):
            with index.OpenAIEmbedder("synthetic", base_url=url, timeout=0.05) as embedder:
                embedder.embed(["Synthetic"])
        assert embedder.client.is_closed()
        assert len(calls) == 1


@pytest.mark.parametrize("texts", [[], [""], [" "], ["a" * 801], ["x"] * 17, [None]])
def test_bad_batch_never_reaches_http(texts):
    with provider() as (url, calls):
        with index.OpenAIEmbedder("synthetic", base_url=url) as embedder:
            with pytest.raises(index.IndexError):
                embedder.embed(texts)
        assert calls == []


@pytest.mark.parametrize("key,timeout", [(None, 1), ("", 1), ("  ", 1),
    ("synthetic", 0), ("synthetic", float("nan")), ("synthetic", True)])
def test_invalid_provider_config_rejected(key, timeout):
    with pytest.raises(index.IndexError):
        index.OpenAIEmbedder(key, timeout=timeout)


def test_query_contract_and_no_default_embedding(tmp_path):
    with provider() as (url, _):
        index.rebuild([doc()], tmp_path, factory(url))
    snapshot = index.Snapshot(tmp_path)
    for limit in (0, 11, True):
        with pytest.raises(index.IndexError, match="limit"):
            snapshot.search([1, 1, 2], model=index.DEFAULT_MODEL, limit=limit)
    with pytest.raises(index.IndexError, match="dimensions"):
        snapshot.search([1, 2], model=index.DEFAULT_MODEL)


def test_interruption_after_pointer_commit_keeps_valid_generation(tmp_path, monkeypatch):
    with provider() as (url, _):
        index.rebuild([doc()], tmp_path, factory(url))
    atomic = index._atomic_json
    def interrupted(path, value):
        atomic(path, value)
        raise KeyboardInterrupt("after committed rename")
    monkeypatch.setattr(index, "_atomic_json", interrupted)
    with provider() as (url, _):
        with pytest.raises(KeyboardInterrupt):
            index.rebuild([doc(text="Replacement")], tmp_path, factory(url))
    assert results(tmp_path)[0]["text"] == "Replacement"
