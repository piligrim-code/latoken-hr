# LATOKEN HR Bot Prototype

Historical Telegram Q&A and quiz prototype using retrieved company text,
OpenAI and a local Chroma index. This is not an official company service or
a qualified hiring system. The full bot is not production-ready.

## Offline checks and HTML parser

Python 3.12 is the CI target. From a virtual environment:

```console
python -m pip install -r requirements-test.txt
python -m pytest tests -q
python parser.py --help
python vector.py --help
```

These checks need no credentials, model downloads, Telegram connection or
external embedding calls. Tests use synthetic HTML and documents. The test
requirements do NOT install or qualify the full legacy bot stack.

## Reviewed snapshot index

`reviewed_index.py` adds a separately tested index path with explicit per-file
review hashes, bounded inputs, content/model-aware generations and atomic
publication. Updates and removals replace the active snapshot only after all
batches succeed. Unchanged input makes no embedding request.

```console
python -m pip install -r requirements-test.txt -r requirements-index.txt
python -m pip check
python -m pytest tests -q
python reviewed_index.py --help
```

With the optional index dependencies installed, the suite also exercises real
temporary Chroma and loopback embeddings HTTP. Without them those integration
tests skip. CI installs them on Windows and Linux. Read
[setup, review format and retention semantics](docs/reviewed-index.md) before
indexing. Old snapshots remain on disk: removal from search is not data erasure.
This is not yet connected to the legacy Telegram bot below.

To fetch a page you are authorized to use:

```console
python parser.py https://example.com --output-dir scraped-data
```

The parser accepts HTTP(S) HTML, uses a 20-second socket timeout and a 2 MiB
decoded-body limit, removes scripts/styles and refuses to overwrite text or
provenance files. Redirects are rejected; pass the reviewed final URL. A
socket timeout is not an overall wall-clock deadline. Do not expose this CLI
as a URL-fetching web service: it is not an SSRF security boundary.

Each new text file has a `.source.json` sidecar with a source URL (query and
fragment omitted), retrieval time, text SHA256 and unverified rights status.
URLs can still carry sensitive information in their path. Use public URLs
without secrets. Provenance metadata does not establish publication rights.

## Legacy bot and indexer boundaries

- `main1.py`: aiogram polling and quiz flow. Bot construction occurs only
  during explicit startup; missing `BOT_TOKEN` produces a configuration error.
- `llm.py`: retrieval and synchronous model calls. Prompts resolve relative
  to the module, not the shell's working directory. Responses are not printed.
- `vector.py`: document loading, chunk IDs and Chroma ingestion. Importing
  it does not read documents, load credentials or call an embedding service.
- `parser.py`: standalone page downloader; no downloads on import.

The legacy bot additionally uses aiogram, OpenAI, python-dotenv, LangChain,
langchain-community, langchain-text-splitters and Chroma. Its historical
LangChain imports have not been migrated or validated against a complete
locked dependency set. Do not treat the parser/test manifest as a deployable
bot environment. Full-stack migration and service integration tests remain.

After resolving that environment and reviewing your own data, indexing is
an explicit operation that sends document text to an external provider and
can incur charges:

```console
python vector.py --data-path reviewed-text --chroma-path chroma --allow-external-embeddings
```

The CLI requires both paths and consent. No bundled corpus is selected by
default. The flag is an operator acknowledgement, not a security barrier:
direct calls to embedding/indexing helpers can still contact the provider.
Queries and retrieved context are also sent to OpenAI during bot operation.
Do not supply candidate records, private chats or other unreviewed material.

## Bundled data and provenance

The 30 historical `data/parsed_page_*.txt` files predate the new sidecars.
They have no verified source/permission manifest and were left unchanged.
Filename timestamps are not proof of origin, consent or redistribution rights.
Do not automatically train, index or redistribute this corpus. An owner
review must establish source URLs, collection basis and permitted uses first.
No repository LICENSE file was present at this review; this patch does not
assign a new license to code or third-party content.

## Remaining work

- Validate a pinned full bot stack and add disposable service integration tests.
- Move synchronous retrieval/model calls off async handlers; add timeouts,
  failure handling and rate limits at the bot boundary.
- Test quiz-state transitions, retrieval quality and provider failures.
- Migrate the bot to the reviewed snapshot index. The old `vector.py` path
  still uses positional IDs and is not the qualified update/delete path.
- Establish retention, access control and data rights before any public use.
