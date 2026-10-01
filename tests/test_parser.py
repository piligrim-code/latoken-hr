import hashlib
import importlib
import json

import pytest
import requests

import parser


class Response:
    status_code = 200
    encoding = 'utf-8'

    def __init__(self, body=b'<h1>Public example</h1>', content_type='text/html', status=200):
        self.body = body
        self.headers = {'Content-Type': content_type}
        self.status_code = status
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError('synthetic server failure')

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 4):
            yield self.body[i:i + 4]


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('Unexpected real HTTP request')
    monkeypatch.setattr(requests.sessions.Session, 'request', fail)


def test_import_has_no_download_or_directory_creation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    importlib.reload(parser)
    assert list(tmp_path.iterdir()) == []


def test_html_extraction_and_provenance(tmp_path, monkeypatch):
    response = Response(b'<html><style>hidden</style><h1>Public example</h1>'
                        b'<script>private()</script><p>Body</p></html>')
    calls = []
    def get(*args, **kwargs):
        calls.append((args, kwargs))
        return response
    monkeypatch.setattr(requests, 'get', get)
    url = 'https://example.com/page?session=synthetic#part'
    path = parser.parse_url(url, 'page.txt', tmp_path)
    assert path.read_text(encoding='utf-8') == 'Public example\nBody'
    meta = json.loads(path.with_name(path.name + '.source.json').read_text())
    assert meta['source_url_without_query_or_fragment'] == 'https://example.com/page'
    assert meta['rights_status'] == 'not_verified'
    assert meta['text_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert 'synthetic' not in json.dumps(meta)
    assert calls == [((url,), {'timeout': 20, 'stream': True, 'allow_redirects': False})]
    assert response.closed


@pytest.mark.parametrize('url', [
    'file:///etc/passwd', 'ftp://example.com', 'https:///missing',
    'https://name:password@example.com', 'https://name@example.com',
])
def test_rejects_bad_urls_without_network(tmp_path, url):
    with pytest.raises(ValueError):
        parser.parse_url(url, 'page.txt', tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
def test_invalid_timeout(tmp_path, timeout):
    with pytest.raises(ValueError):
        parser.parse_url('https://example.com', 'page.txt', tmp_path, timeout=timeout)


@pytest.mark.parametrize('limit', [0, -1, 1.5, float('inf'), float('nan'), True])
def test_invalid_size_limit(tmp_path, limit):
    with pytest.raises(ValueError):
        parser.parse_url('https://example.com', 'page.txt', tmp_path, max_bytes=limit)


@pytest.mark.parametrize('status', [301, 302, 204, 404, 500])
def test_http_failure_creates_no_files(tmp_path, monkeypatch, status):
    response = Response(status=status)
    monkeypatch.setattr(requests, 'get', lambda *a, **k: response)
    with pytest.raises((ValueError, requests.HTTPError)):
        parser.parse_url('https://example.com', 'page.txt', tmp_path)
    assert response.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('body,content_type,limit', [
    (b'<h1>long response</h1>', 'text/html', 10),
    (b'<script>only script</script>', 'text/html', 100),
    (b'binary', 'application/octet-stream', 100),
])
def test_rejects_oversize_empty_and_nonhtml(tmp_path, monkeypatch, body, content_type, limit):
    response = Response(body, content_type)
    monkeypatch.setattr(requests, 'get', lambda *a, **k: response)
    with pytest.raises(ValueError):
        parser.parse_url('https://example.com', 'page.txt', tmp_path, max_bytes=limit)
    assert response.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('name', ['../escape.txt', 'sub/file.txt', 'sub\\file.txt',
                                 'C:escape.txt', '.', '..', ''])
def test_rejects_path_names(tmp_path, name):
    with pytest.raises(ValueError):
        parser.save_text_to_file('synthetic', name, tmp_path)


@pytest.mark.parametrize('existing', ['page.txt', 'page.txt.source.json'])
def test_no_overwrite(tmp_path, monkeypatch, existing):
    protected = tmp_path / existing
    protected.write_text('keep', encoding='utf-8')
    monkeypatch.setattr(requests, 'get', lambda *a, **k: Response())
    with pytest.raises(FileExistsError):
        parser.parse_url('https://example.com', 'page.txt', tmp_path)
    assert protected.read_text() == 'keep'
    assert len(list(tmp_path.iterdir())) == 1


def test_cli_failure_redacts_request_details(tmp_path, monkeypatch, capsys):
    def timeout(*args, **kwargs):
        raise requests.Timeout('synthetic-private-query')
    monkeypatch.setattr(requests, 'get', timeout)
    assert parser.main(['https://example.com', '--output-dir', str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert 'Timeout' in output
    assert 'synthetic-private-query' not in output


def test_cli_success(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, 'get', lambda *a, **k: Response())
    assert parser.main(['https://example.com', '--output-dir', str(tmp_path)]) == 0
    assert len(list(tmp_path.glob('*.txt'))) == 1
    assert len(list(tmp_path.glob('*.source.json'))) == 1


def test_cli_requires_urls():
    with pytest.raises(SystemExit) as error:
        parser.main([])
    assert error.value.code == 2
