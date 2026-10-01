"""Legacy indexer; external embedding calls require explicit CLI opt-in."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.schema.document import Document

CHROMA_PATH = "chroma"

def get_embedding_function():
    from dotenv import load_dotenv
    from langchain.embeddings import OpenAIEmbeddings

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for external embeddings")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings

def load_documents(data_path):
    directory = Path(data_path)
    if not directory.is_dir():
        raise ValueError("Data directory does not exist")
    text_files = sorted(p for p in directory.iterdir() if p.suffix == '.txt')
    if not text_files:
        return []
    from langchain.document_loaders import TextLoader
    
    documents = []
    for text_file in text_files:
        if text_file.is_symlink() or not text_file.is_file():
            raise ValueError("Data entries must be regular, non-symlink text files")
        loader = TextLoader(str(text_file), encoding='utf-8')
        documents.extend(loader.load())  
    
    return documents

def split_documents(documents: list[Document]):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], chroma_path=CHROMA_PATH):
    if not chunks:
        return 0
    from langchain.vectorstores.chroma import Chroma

    db = Chroma(
        persist_directory=str(chroma_path), embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            existing_ids.add(chunk.metadata["id"])

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")
    return len(new_chunks)


def calculate_chunk_ids(chunks):

    counts = {}

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        current_chunk_index = counts.get(current_page_id, 0)
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        counts[current_page_id] = current_chunk_index + 1
        chunk.metadata["id"] = chunk_id

    return chunks

def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data-path', required=True, help='Directory of reviewed text files')
    cli.add_argument('--chroma-path', required=True, help='Local index directory')
    cli.add_argument('--allow-external-embeddings', action='store_true',
                     help='Allow sending document text to the configured embedding provider')
    args = cli.parse_args(argv)
    if not args.allow_external_embeddings:
        cli.error('External embeddings require --allow-external-embeddings after data review')
    documents = load_documents(args.data_path)
    if not documents:
        print('No text documents; no external request made')
        return 0
    chunks = split_documents(documents)
    add_to_chroma(chunks, args.chroma_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
