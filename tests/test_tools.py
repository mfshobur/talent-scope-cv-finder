import json
import tempfile
import pytest
from unittest.mock import MagicMock

from agent.tools import (
    search_candidates, get_candidate_detail, compare_candidates,
    filter_candidates, dispatch_tool, build_tool_schemas,
)
from ingestion.chromadb_store import CandidateStore
from core.schemas import ParsedCV, EmployeeProfile


def make_store_with_profiles(*profiles: EmployeeProfile) -> CandidateStore:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CandidateStore(tmpdir)
    # Re-open in same tempdir (already deleted but chromadb in-memory is fine for tests)
    store = MagicMock(spec=CandidateStore)

    profile_map = {p.employee_id: p for p in profiles}

    def mock_search(query_embedding, n_results=20, department_filter=None):
        results = []
        for p in profiles:
            if department_filter and p.cv.department != department_filter:
                continue
            results.append({
                "employee_id": p.employee_id,
                "full_name": p.cv.full_name,
                "current_role": p.cv.current_role,
                "department": p.cv.department,
                "skills": p.cv.skills,
                "similarity": 0.9,
                "profile_json": p.model_dump_json(),
            })
        return results[:n_results]

    store.search.side_effect = mock_search
    store.get_profile.side_effect = lambda eid: profile_map.get(eid)
    store.get_all_profiles.return_value = list(profiles)
    return store


@pytest.fixture
def mock_embedder():
    e = MagicMock()
    e.embed.return_value = [0.1] * 1536
    return e


@pytest.fixture
def it_profile(sample_profile):
    return sample_profile


@pytest.fixture
def hr_profile():
    cv = ParsedCV(
        employee_id="emp_011",
        full_name="Joko Santoso",
        current_role="HR Business Partner",
        department="HR",
        skills=["HRIS", "Recruitment", "Performance Management"],
    )
    return EmployeeProfile(employee_id="emp_011", cv=cv)


class TestSearchCandidates:
    def test_returns_candidates(self, it_profile, mock_embedder):
        store = make_store_with_profiles(it_profile)
        result = search_candidates("Python developer", 5, store, mock_embedder)
        assert "candidates" in result
        assert len(result["candidates"]) >= 1
        assert result["candidates"][0]["employee_id"] == "emp_001"

    def test_respects_top_k(self, it_profile, hr_profile, mock_embedder):
        store = make_store_with_profiles(it_profile, hr_profile)
        result = search_candidates("engineer", 1, store, mock_embedder)
        assert len(result["candidates"]) <= 1


class TestGetCandidateDetail:
    def test_returns_full_profile(self, it_profile):
        store = make_store_with_profiles(it_profile)
        result = get_candidate_detail("emp_001", store)
        assert result["full_name"] == "Muh. Shobur Fattah"
        assert "skills" in result
        assert "assessments" in result
        assert "experience" in result

    def test_not_found_returns_error(self, it_profile):
        store = make_store_with_profiles(it_profile)
        result = get_candidate_detail("emp_999", store)
        assert "error" in result


class TestCompareCandidates:
    def test_compares_multiple(self, it_profile, hr_profile):
        store = make_store_with_profiles(it_profile, hr_profile)
        result = compare_candidates(["emp_001", "emp_011"], store)
        assert len(result["compared"]) == 2

    def test_skips_missing(self, it_profile):
        store = make_store_with_profiles(it_profile)
        result = compare_candidates(["emp_001", "emp_999"], store)
        assert len(result["compared"]) == 1


class TestFilterCandidates:
    def test_filter_by_department(self, it_profile, hr_profile, mock_embedder):
        store = make_store_with_profiles(it_profile, hr_profile)
        result = filter_candidates(store, mock_embedder, department="IT")
        depts = [c["department"] for c in result["candidates"]]
        assert all(d == "IT" for d in depts)

    def test_filter_by_skills(self, it_profile, hr_profile, mock_embedder):
        store = make_store_with_profiles(it_profile, hr_profile)
        result = filter_candidates(store, mock_embedder, required_skills=["Python"])
        ids = [c["employee_id"] for c in result["candidates"]]
        assert "emp_001" in ids
        assert "emp_011" not in ids


class TestDispatchTool:
    def test_dispatch_search(self, it_profile, mock_embedder):
        store = make_store_with_profiles(it_profile)
        result = dispatch_tool(
            "search_candidates",
            json.dumps({"query": "Python ML engineer", "top_k": 3}),
            store,
            mock_embedder,
        )
        assert "candidates" in result

    def test_dispatch_unknown_tool(self, it_profile, mock_embedder):
        store = make_store_with_profiles(it_profile)
        result = dispatch_tool("nonexistent_tool", "{}", store, mock_embedder)
        assert "error" in result


class TestToolSchemas:
    def test_all_schemas_valid(self):
        schemas = build_tool_schemas()
        assert len(schemas) == 4
        names = [s["function"]["name"] for s in schemas]
        assert "search_candidates" in names
        assert "get_candidate_detail" in names
        assert "compare_candidates" in names
        assert "filter_candidates" in names
