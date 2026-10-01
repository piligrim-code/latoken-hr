"""Focused startup and prompt tests; not a live Telegram/model integration."""
import ast
import asyncio
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def startup_function(token, events, fail=False):
    tree = ast.parse((ROOT / 'main1.py').read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'main')
    class Bot:
        def __init__(self, value):
            events.append(('bot', value))
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            events.append(('close',))
    async def poll(bot):
        events.append(('poll',))
        if fail:
            raise RuntimeError('synthetic polling failure')
    namespace = {'TOKEN': token, 'Bot': Bot, 'dp': SimpleNamespace(start_polling=poll)}
    exec(compile(ast.Module(body=[main], type_ignores=[]), 'main1.py', 'exec'), namespace)
    return namespace['main']


def test_missing_bot_token_does_not_create_client():
    events = []
    with pytest.raises(ValueError, match='BOT_TOKEN'):
        asyncio.run(startup_function(None, events)())
    assert events == []


@pytest.mark.parametrize('fail', [False, True])
def test_bot_session_closed_on_success_and_error(fail):
    events = []
    main = startup_function('synthetic-token', events, fail)
    if fail:
        with pytest.raises(RuntimeError):
            asyncio.run(main())
    else:
        asyncio.run(main())
    assert events == [('bot', 'synthetic-token'), ('poll',), ('close',)]


def test_llm_import_from_other_directory_and_no_response_logging(tmp_path, monkeypatch, capsys):
    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
    class Chroma:
        def __init__(self, **kwargs):
            pass
        def similarity_search_with_score(self, query, k):
            return [(SimpleNamespace(page_content='synthetic context', metadata={'id': 'doc'}), 0.1)]
    class Prompt:
        @staticmethod
        def from_template(template):
            return SimpleNamespace(format=lambda **kw: 'synthetic prompt')
    module('openai', OpenAI=lambda **kw: pytest.fail('real model client'))
    module('langchain.vectorstores', Chroma=Chroma)
    module('langchain.prompts', ChatPromptTemplate=Prompt)
    module('dotenv', load_dotenv=lambda: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, 'llm', raising=False)
    llm = importlib.import_module('llm')
    monkeypatch.setattr(llm, 'get_embedding_function', lambda: 'fake')
    monkeypatch.setattr(llm, 'query_chatgpt', lambda prompt: 'synthetic private reply')
    assert llm.search_and_respond('question') == 'synthetic private reply'
    assert capsys.readouterr().out == ''
