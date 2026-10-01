"""Explicit, bounded HTML downloads. Importing this module performs no I/O."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import requests
from bs4 import BeautifulSoup

MAX_BYTES = 2 * 1024 * 1024


def validate_url(url):
    parts = urlsplit(url)
    if (parts.scheme not in ('http', 'https') or not parts.hostname
            or parts.username is not None or parts.password is not None):
        raise ValueError('Use an HTTP(S) URL without embedded credentials')
    return parts


def save_text_to_file(text, file_name, output_dir='scraped-data'):
    if (not file_name or file_name in ('.', '..')
            or any(c in file_name for c in '/\\:')
            or Path(file_name).name != file_name):
        raise ValueError('file_name must be a plain file name')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / file_name
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    return path


def parse_url(url, file_name, output_dir='scraped-data', *, timeout=20, max_bytes=MAX_BYTES):
    parts = validate_url(url)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError('max_bytes must be a positive integer')
    with requests.get(url, timeout=timeout, stream=True, allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('Expected HTTP 200; redirects are not followed')
        content_type = response.headers.get('Content-Type', '').split(';', 1)[0].strip().lower()
        if content_type not in ('text/html', 'application/xhtml+xml'):
            raise ValueError('Expected an HTML response')
        body = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            body.extend(chunk)
            if len(body) > max_bytes:
                raise ValueError('Response exceeds download size limit')
        soup = BeautifulSoup(bytes(body), 'html.parser', from_encoding=response.encoding)
    for element in soup(['script', 'style', 'noscript']):
        element.decompose()
    text = soup.get_text(separator='\n', strip=True)
    if not text:
        raise ValueError('Page contains no extractable text')
    directory = Path(output_dir)
    # Keep provenance separate from text consumed by the legacy indexer.
    metadata = {
        'source_url_without_query_or_fragment': urlunsplit((parts.scheme, parts.netloc, parts.path, '', '')),
        'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
        'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'rights_status': 'not_verified',
    }
    if (directory / (file_name + '.source.json')).exists():
        raise FileExistsError('Provenance file already exists')
    path = save_text_to_file(text, file_name, directory)
    with path.with_name(path.name + '.source.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2)
    return path


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('urls', nargs='+', help='Approved public HTTP(S) pages to fetch')
    cli.add_argument('--output-dir', default='scraped-data')
    cli.add_argument('--timeout', type=float, default=20)
    args = cli.parse_args(argv)
    failures = 0
    for url in args.urls:
        try:
            file_name = 'parsed_page_' + uuid4().hex + '.txt'
            path = parse_url(url, file_name, args.output_dir, timeout=args.timeout)
            print('Saved ' + path.name)
        except (requests.RequestException, OSError, ValueError) as error:
            # Exceptions can contain query credentials or private server responses.
            print('Download failed: ' + type(error).__name__)
            failures += 1
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
