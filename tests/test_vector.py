import builtins
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import vector


def stub(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def document(source, page=None):
    return SimpleNamespace(metadata={'source': source, 'page': page}, page_content='synthetic')


def test_import_needs_no_provider_or_data(tmp_path, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith(('langchain', 'openai', 'dotenv', 'chromadb')):
            pytest.fail('Provider import during module initialization: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.chdir(tmp_path)
    importlib.reload(vector)
    assert not list(tmp_path.iterdir())


def test_noncontiguous_source_ids_are_unique_and_stable():
    docs = [document('a'), document('b'), document('a'), document('a', 2)]
    assert vector.calculate_chunk_ids(docs) is docs
    ids = [d.metadata['id'] for d in docs]
    assert ids == ['a:None:0', 'b:None:0', 'a:None:1', 'a:2:0']
    vector.calculate_chunk_ids(docs)
    assert [d.metadata['id'] for d in docs] == ids


def test_consent_checked_before_loading(monkeypatch):
    monkeypatch.setattr(vector, 'load_documents', lambda *a: pytest.fail('read before consent'))
    with pytest.raises(SystemExit) as error:
        vector.main(['--data-path', 'reviewed', '--chroma-path', 'index'])
    assert error.value.code == 2


@pytest.mark.parametrize('args', [[], ['--allow-external-embeddings']])
def test_paths_are_required(args):
    with pytest.raises(SystemExit) as error:
        vector.main(args)
    assert error.value.code == 2


def test_explicit_index_flow(monkeypatch):
    calls = []
    monkeypatch.setattr(vector, 'load_documents', lambda path: calls.append(('load', path)) or ['doc'])
    monkeypatch.setattr(vector, 'split_documents', lambda docs: calls.append(('split', docs)) or ['chunk'])
    monkeypatch.setattr(vector, 'add_to_chroma', lambda chunks, path: calls.append(('add', chunks, path)))
    assert vector.main(['--data-path', 'reviewed', '--chroma-path', 'index',
                        '--allow-external-embeddings']) == 0
    assert calls == [('load', 'reviewed'), ('split', ['doc']), ('add', ['chunk'], 'index')]


def test_empty_input_does_not_call_provider(monkeypatch):
    monkeypatch.setattr(vector, 'load_documents', lambda path: [])
    monkeypatch.setattr(vector, 'split_documents', lambda docs: pytest.fail('empty split'))
    assert vector.main(['--data-path', 'reviewed', '--chroma-path', 'index',
                        '--allow-external-embeddings']) == 0
    assert vector.add_to_chroma([]) == 0


def test_loader_reads_only_sorted_text_files(tmp_path, monkeypatch):
    for name in ('b.txt', 'a.txt', 'a.txt.source.json'):
        (tmp_path / name).write_text('synthetic', encoding='utf-8')
    loaded = []
    class Loader:
        def __init__(self, path, encoding):
            assert encoding == 'utf-8'
            self.path = path
        def load(self):
            loaded.append(Path(self.path).name)
            return [document(self.path)]
    stub(monkeypatch, 'langchain.document_loaders', TextLoader=Loader)
    assert len(vector.load_documents(tmp_path)) == 2
    assert loaded == ['a.txt', 'b.txt']


def test_loader_rejects_directory_named_txt(tmp_path, monkeypatch):
    (tmp_path / 'bad.txt').mkdir()
    stub(monkeypatch, 'langchain.document_loaders', TextLoader=lambda *a, **k: pytest.fail('directory load'))
    with pytest.raises(ValueError):
        vector.load_documents(tmp_path)


def test_empty_and_missing_directories_need_no_provider(tmp_path):
    assert vector.load_documents(tmp_path) == []
    with pytest.raises(ValueError, match='does not exist'):
        vector.load_documents(tmp_path / 'missing')


def test_missing_key_rejected_without_embedding_client(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    stub(monkeypatch, 'dotenv', load_dotenv=lambda: None)
    stub(monkeypatch, 'langchain.embeddings', OpenAIEmbeddings=lambda **kw: pytest.fail('client with no key'))
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        vector.get_embedding_function()


def test_chroma_skips_existing_ids(monkeypatch, tmp_path):
    calls = []
    class Chroma:
        def __init__(self, **kwargs):
            calls.append(('init', kwargs))
        def get(self, include):
            assert include == []
            return {'ids': ['a:None:0']}
        def add_documents(self, chunks, ids):
            calls.append(('add', ids))
        def persist(self):
            calls.append(('persist',))
    stub(monkeypatch, 'langchain.vectorstores.chroma', Chroma=Chroma)
    monkeypatch.setattr(vector, 'get_embedding_function', lambda: 'fake-embedding')
    assert vector.add_to_chroma([document('a'), document('a')], tmp_path) == 1
    assert calls[0] == ('init', {'persist_directory': str(tmp_path), 'embedding_function': 'fake-embedding'})
    assert calls[1:] == [('add', ['a:None:1']), ('persist',)]
