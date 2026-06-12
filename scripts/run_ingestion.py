"""
Build ChromaDB from all data/employees/* folders.
Run once before deploying. Commit chroma_db/ to git afterward.

Usage: python scripts/run_ingestion.py [--force]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from core.embedder import Embedder
from core.parser import DocumentParser
from ingestion.chromadb_store import CandidateStore
from ingestion.pipeline import ingest_all

DATA_DIR = Path(__file__).parent.parent / "data" / "employees"


def main():
    s = Settings()
    if not s.effective_llm_api_key():
        print("ERROR: No API key. Set LLM_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)

    store = CandidateStore(s.chroma_persist_dir)
    parser = DocumentParser(s)
    embedder = Embedder(s)

    print(f"Ingesting employees from {DATA_DIR}")
    print(f"Model: {s.vision_model} | Embeddings: {s.embedding_model}\n")

    profiles = ingest_all(DATA_DIR, store, parser, embedder, verbose=True)

    total_docs = store.count()
    print(f"\nDone. {len(profiles)} employees ingested.")
    print(f"ChromaDB: {total_docs} documents in collection '{store._col.name}'")
    print(f"Stored at: {s.chroma_persist_dir}")


if __name__ == "__main__":
    main()
