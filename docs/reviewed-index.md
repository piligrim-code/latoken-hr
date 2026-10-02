# Reviewed Index

This is a standalone replacement index path. It does not qualify the legacy
Telegram bot: `main1.py` and `llm.py` still use the old layout until the separate
bot migration. Do not point the legacy bot at this directory or deploy it merely
because these index tests pass.

## Setup

Python 3.12 is the CI target; the local Windows check also covers Python 3.13.

```console
python -m venv .venv
python -m pip install -r requirements-test.txt -r requirements-index.txt
python -m pip check
python -m pytest tests -q
python reviewed_index.py --help
```

Activate the virtual environment first using your shell's usual activation
command. Direct index dependencies are pinned. Transitive versions are resolved
by pip; this is not yet a complete locked bot environment.

Tests use generated text, a real disposable embedded Chroma database and a
loopback HTTP server speaking the embeddings API shape. No Telegram, OpenAI,
model weights or historical corpus is used. These checks establish the local
storage/provider contract, not semantic retrieval accuracy or data rights.

## Review Before Sending Text

Place only reviewed UTF-8 `.txt` files in a separate directory. The bundled
`data/` corpus is explicitly rejected. Supply a separate JSON review manifest:

```json
{
  "schema": 1,
  "documents": [
    {
      "file": "public-faq.txt",
      "sha256": "REPLACE_WITH_64_CHARACTER_LOWERCASE_SHA256_OF_FILE_BYTES",
      "source": "Reviewed public FAQ",
      "approved_for_external_embedding": true
    }
  ]
}
```

Compute each hash after review, for example with `Get-FileHash -Algorithm SHA256`
on PowerShell or `sha256sum` on Linux. Store lowercase hex. Do not normalize line
endings after hashing: byte changes require a new review. The manifest must list
every `.txt` file exactly once. Changed, unlisted, missing, empty, binary-like,
non-UTF-8 and symlink files are rejected. Subdirectories are not traversed.

The boolean is an operator acknowledgement, not independent evidence of consent
or copyright permission. The source label and basename are stored in Chroma;
use public labels with no local paths, personal identifiers, URLs with secrets
or credentials. Do not use candidate records or private chat transcripts.

Limits: 200 documents, 256 KiB per file, 2 MiB in total, 3,000 chunks, 800 Unicode
characters per chunk with 80-character overlap. The fixed splitter has no
semantic/sentence-boundary guarantees. Inputs beyond the bounds are refused,
not silently truncated. Index and review directories must be operator-owned;
the filesystem checks are not a boundary against an attacker modifying files
concurrently on the same machine.

## Indexing

Set `OPENAI_API_KEY` through a secret manager or your shell, not a committed
file. The command does not load `.env`, inherit a custom `OPENAI_BASE_URL`, or
use environment proxies. It sends text only after both review and the flag:

```console
python reviewed_index.py --data-path reviewed-text --review-manifest review.json --index-path reviewed-index --allow-external-embeddings
```

Default embedding model: `text-embedding-ada-002`, matching the legacy
LangChain embedding default. `--embedding-model` permits an explicit change.
Every snapshot records the model; queries must use the same identifier and
dimension. Changing the model, bytes, names or source labels creates a new
generation. A changed corpus currently re-embeds all its chunks; estimate cost
before large updates. An unchanged corpus does not create a provider client or
make any embedding request, but still verifies the active collection.

One SDK/HTTP client is owned and closed per rebuild. Batches contain at most 16
chunks, SDK retries are disabled, and each HTTP operation has a 30-second socket
timeout. This is not an overall wall-clock deadline or a transport response-byte
limit. The CLI prints counts/status only; provider exception bodies are not
printed because they may contain submitted data. Treat external SDK debug
logging as sensitive.

## Update, Delete And Failure Semantics

Update the reviewed files and manifest, then rerun the same command. Removed
documents disappear from the new active generation. To deliberately publish an
empty corpus, use an empty reviewed directory, `"documents": []`, and add
`--allow-empty`. Empty publication needs no API key/provider client but retains
the explicit external-embedding flag as an operation acknowledgement.

Only one writer may run per index directory. A second writer fails promptly.
An unmanaged/nonempty directory is never adopted; use a new directory instead
of overwriting a historical Chroma index.

The layout contains `owner.json`, an embedded `chroma/` database, a writer lock
and the atomically replaced `active.json` pointer. A generation is published
only after every batch and the final count/metadata check succeed. Failed
provider/batch/pointer writes leave the previous pointer intact. The first
failed build leaves no active generation. Retry is explicit, never automatic.
Atomic rename is not a guarantee against arbitrary disk corruption or power
loss; use local storage and validate backups/restores independently.

Existing `Snapshot` readers remain pinned to their original generation; create
a new instance to adopt an update. Old generations and failed-build orphans are
retained to avoid invalidating readers and to make an interrupted commit safe.
**Deletion from active search is not physical erasure.** Monitor disk usage.
For data erasure/retention, stop all readers and writers, build a new index at a
new path from the approved corpus, switch consumers, and remove the old index
and backups only under the operator's retention policy. No automatic cleanup
or backup policy is claimed here.

Chroma's embedded client is process-lived; this module does not call private
Chroma shutdown APIs. The index CLI is a bounded-duration process, not a
multi-tenant storage service. OS permissions, disk encryption, access control,
backup/restore and retention are separate deployment requirements.

## Retrieval Contract

`Snapshot(path).search(vector, model=..., limit=5)` queries a pinned reviewed
generation with explicit vectors and no default Chroma embedding function.
It returns chunk text, a portable source label, basename, ID and cosine
distance. A smaller distance is not a calibrated confidence score. Model,
dimension and count mismatches fail closed. Empty snapshots return no results.
Embedding a user's question and placing context into a bot answer are separate
operations; the legacy bot migration and grounding evaluation remain pending.
