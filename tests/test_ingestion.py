import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.file_loader import discover_employee_files, EmployeeFiles
from ingestion.pipeline import ingest_employee, ingest_all
from ingestion.chromadb_store import CandidateStore
from core.schemas import ParsedCV, ParsedAssessment, EmployeeProfile


# ── file_loader tests ────────────────────────────────────────────────────────

class TestDiscoverEmployeeFiles:
    def test_finds_cv_and_assessments(self, tmp_path):
        emp_dir = tmp_path / "emp_test"
        emp_dir.mkdir()
        (emp_dir / "cv.pdf").write_bytes(b"fake pdf")
        asm_dir = emp_dir / "assessments"
        asm_dir.mkdir()
        (asm_dir / "psychotest.pdf").write_bytes(b"fake")
        (asm_dir / "technical.pdf").write_bytes(b"fake")
        (asm_dir / "english.pdf").write_bytes(b"fake")

        files = discover_employee_files(emp_dir)
        assert files.employee_id == "emp_test"
        assert files.cv_path.name == "cv.pdf"
        assert "psychotest" in files.assessment_paths
        assert "technical" in files.assessment_paths
        assert "english_proficiency" in files.assessment_paths

    def test_no_cv_raises(self, tmp_path):
        emp_dir = tmp_path / "emp_empty"
        emp_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            discover_employee_files(emp_dir)

    def test_no_assessments_ok(self, tmp_path):
        emp_dir = tmp_path / "emp_no_asm"
        emp_dir.mkdir()
        (emp_dir / "cv.pdf").write_bytes(b"fake")
        files = discover_employee_files(emp_dir)
        assert files.assessment_paths == {}


# ── pipeline tests ─────────────────────────────────────────────────────────

def fake_embedder(text: str) -> list[float]:
    return [0.1] * 1536


def make_mock_parser(employee_id: str):
    parser = MagicMock()
    parser.parse_cv.return_value = ParsedCV(
        employee_id=employee_id,
        full_name="Test Employee",
        current_role="Engineer",
        department="IT",
        skills=["Python", "SQL"],
    )
    parser.parse_assessment.return_value = ParsedAssessment(
        employee_id=employee_id,
        category="technical",
        score_label="Good",
        key_findings=["Strong skills"],
    )
    return parser


class TestIngestEmployee:
    def test_ingests_cv_and_assessments(self, tmp_path):
        emp_dir = tmp_path / "emp_test"
        emp_dir.mkdir()
        (emp_dir / "cv.pdf").write_bytes(b"fake")
        asm_dir = emp_dir / "assessments"
        asm_dir.mkdir()
        (asm_dir / "technical.pdf").write_bytes(b"fake")

        with tempfile.TemporaryDirectory() as chroma_dir:
            store = CandidateStore(chroma_dir)
            embedder = MagicMock()
            embedder.embed.return_value = [0.1] * 1536

            parser = make_mock_parser("emp_test")
            profile = ingest_employee(emp_dir, store, parser, embedder)

        assert profile.employee_id == "emp_test"
        assert profile.cv.full_name == "Test Employee"
        assert len(profile.assessments) == 1

    def test_ingest_all_skips_on_error(self, tmp_path):
        good_dir = tmp_path / "emp_good"
        good_dir.mkdir()
        (good_dir / "cv.pdf").write_bytes(b"fake")

        bad_dir = tmp_path / "emp_bad"
        bad_dir.mkdir()
        # no cv.pdf → will raise FileNotFoundError

        with tempfile.TemporaryDirectory() as chroma_dir:
            store = CandidateStore(chroma_dir)
            embedder = MagicMock()
            embedder.embed.return_value = [0.1] * 1536
            parser = make_mock_parser("emp_good")

            profiles = ingest_all(tmp_path, store, parser, embedder, verbose=False)

        assert len(profiles) == 1
        assert profiles[0].employee_id == "emp_good"
