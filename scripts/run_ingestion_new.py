"""
Ingest only employees not yet in ChromaDB.
Safe to run repeatedly — already-ingested employees are skipped.

Usage: uv run python scripts/run_ingestion_new.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from core.embedder import Embedder
from core.parser import DocumentParser
from ingestion.chromadb_store import CandidateStore
from ingestion.pipeline import ingest_employee

DATA_DIR = Path(__file__).parent.parent / "data" / "employees"


def main():
    s = Settings()
    if not s.effective_llm_api_key():
        print("ERROR: No API key. Set LLM_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)

    store = CandidateStore(s.chroma_persist_dir)
    parser = DocumentParser(s)
    embedder = Embedder(s)

    existing_ids = {p.employee_id for p in store.get_all_profiles()}
    print(f"Already ingested: {len(existing_ids)} employees")

    def folder_to_id(path: Path) -> str:
        parts = path.name.split("_")
        return f"{parts[0]}_{parts[1]}"

    all_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("emp_"))
    new_dirs = [d for d in all_dirs if folder_to_id(d) not in existing_ids]

    if not new_dirs:
        print("No new employees found. Nothing to ingest.")
        return

    print(f"New employees to ingest: {len(new_dirs)}")
    print(f"Model: {s.vision_model} | Embeddings: {s.embedding_model}\n")

    ingested = 0
    for i, emp_dir in enumerate(new_dirs, 1):
        eid = folder_to_id(emp_dir)
        print(f"  [{i}/{len(new_dirs)}] {emp_dir.name} ...", end=" ", flush=True)
        try:
            profile = ingest_employee(emp_dir, store, parser, embedder)
            ingested += 1
            print(f"OK ({len(profile.assessments)} assessments)")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone. {ingested}/{len(new_dirs)} new employees ingested.")
    print(f"ChromaDB: {store.count()} total documents")
    print(f"Stored at: {s.chroma_persist_dir}")


if __name__ == "__main__":
    main()
