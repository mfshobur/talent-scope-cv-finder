import json
import tempfile
import pytest

from ingestion.chromadb_store import CandidateStore
from core.schemas import EmployeeProfile


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield CandidateStore(tmpdir)


def fake_embedder(text: str) -> list[float]:
    # Deterministic fake embedding based on text length
    return [float(len(text) % 100) / 100.0] * 1536


class TestCandidateStore:
    def test_upsert_and_count(self, temp_store, sample_profile):
        assert temp_store.count() == 0
        temp_store.upsert_employee(sample_profile, fake_embedder)
        assert temp_store.count() == 2  # cv doc + asm doc

    def test_get_profile_round_trip(self, temp_store, sample_profile):
        temp_store.upsert_employee(sample_profile, fake_embedder)
        retrieved = temp_store.get_profile("emp_001")
        assert retrieved is not None
        assert retrieved.cv.full_name == "Muh. Shobur Fattah"
        assert retrieved.cv.employee_id == "emp_001"

    def test_get_profile_not_found(self, temp_store):
        result = temp_store.get_profile("emp_999")
        assert result is None

    def test_search_returns_results(self, temp_store, sample_profile):
        temp_store.upsert_employee(sample_profile, fake_embedder)
        query_embedding = [0.5] * 1536
        results = temp_store.search(query_embedding, n_results=5)
        assert len(results) >= 1
        assert results[0]["employee_id"] == "emp_001"

    def test_search_deduplicates_by_employee(self, temp_store, sample_profile):
        temp_store.upsert_employee(sample_profile, fake_embedder)
        # Even though there are 2 docs (cv + asm), search should deduplicate
        results = temp_store.search([0.5] * 1536, n_results=10)
        employee_ids = [r["employee_id"] for r in results]
        assert len(employee_ids) == len(set(employee_ids))

    def test_search_department_filter(self, temp_store, sample_profile, sample_cv):
        from core.schemas import ParsedCV, EmployeeProfile
        # Add an HR employee
        hr_cv = ParsedCV(
            employee_id="emp_011",
            full_name="Joko Santoso",
            current_role="HR Business Partner",
            department="HR",
            skills=["HRIS", "Recruitment"],
        )
        hr_profile = EmployeeProfile(employee_id="emp_011", cv=hr_cv)
        temp_store.upsert_employee(sample_profile, fake_embedder)
        temp_store.upsert_employee(hr_profile, fake_embedder)

        results = temp_store.search([0.5] * 1536, n_results=10, department_filter="IT")
        depts = [r["department"] for r in results]
        assert all(d == "IT" for d in depts)

    def test_get_all_profiles(self, temp_store, sample_profile, sample_cv):
        from core.schemas import ParsedCV, EmployeeProfile
        cv2 = ParsedCV(employee_id="emp_002", full_name="Arini", current_role="Data Engineer", department="IT", skills=[])
        profile2 = EmployeeProfile(employee_id="emp_002", cv=cv2)
        temp_store.upsert_employee(sample_profile, fake_embedder)
        temp_store.upsert_employee(profile2, fake_embedder)
        all_profiles = temp_store.get_all_profiles()
        assert len(all_profiles) == 2

    def test_upsert_is_idempotent(self, temp_store, sample_profile):
        temp_store.upsert_employee(sample_profile, fake_embedder)
        temp_store.upsert_employee(sample_profile, fake_embedder)
        assert temp_store.count() == 2  # still 2, not 4
