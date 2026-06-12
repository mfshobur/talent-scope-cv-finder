import time
from pathlib import Path

from core.embedder import Embedder
from core.parser import DocumentParser
from core.schemas import EmployeeProfile
from ingestion.chromadb_store import CandidateStore
from ingestion.file_loader import discover_employee_files


def ingest_employee(
    employee_dir: Path,
    store: CandidateStore,
    parser: DocumentParser,
    embedder: Embedder,
) -> EmployeeProfile:
    files = discover_employee_files(employee_dir)
    eid = files.employee_id

    cv = parser.parse_cv(files.cv_path, eid)

    assessments = []
    for category, path in files.assessment_paths.items():
        asm = parser.parse_assessment(path, eid, category)
        assessments.append(asm)

    profile = EmployeeProfile(employee_id=eid, cv=cv, assessments=assessments)
    store.upsert_employee(profile, embedder.embed)
    return profile


def ingest_all(
    data_dir: Path,
    store: CandidateStore,
    parser: DocumentParser,
    embedder: Embedder,
    verbose: bool = True,
    delay_between: float = 3.0,
) -> list[EmployeeProfile]:
    employee_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("emp_")
    )
    profiles = []
    for i, emp_dir in enumerate(employee_dirs, 1):
        if verbose:
            print(f"  [{i}/{len(employee_dirs)}] {emp_dir.name} ...", end=" ", flush=True)
        try:
            profile = ingest_employee(emp_dir, store, parser, embedder)
            profiles.append(profile)
            if verbose:
                print(f"OK ({len(profile.assessments)} assessments)")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
        if i < len(employee_dirs):
            time.sleep(delay_between)
    return profiles
