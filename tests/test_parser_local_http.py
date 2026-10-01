"""Real requests/BeautifulSoup exercise against a loopback-only synthetic page."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from threading import Thread

import requests

import parser


def test_loopback_http_download(tmp_path, monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'<h1>Synthetic page</h1><p>Local fixture</p>'
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with HTTPServer(('127.0.0.1', 0), Handler) as server, requests.Session() as session:
        session.trust_env = False
        monkeypatch.setattr(parser.requests, 'get', session.get)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/synthetic'
            path = parser.parse_url(url, 'synthetic.txt', tmp_path)
            assert path.read_text(encoding='utf-8') == 'Synthetic page\nLocal fixture'
            meta = json.loads(path.with_name(path.name + '.source.json').read_text())
            assert meta['source_url_without_query_or_fragment'] == url
        finally:
            server.shutdown()
            thread.join(timeout=5)
        assert not thread.is_alive()
