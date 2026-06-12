"""
Agent output quality tests — 3 layers:

Layer 1 (TestRetrievalCorrectness):
    Real ChromaDB + real embeddings. Deterministic: same DB → same top results.
    Ground truth verified by running actual queries before writing assertions.

Layer 2 (TestAgentClarificationBehavior):
    Unit: mocked LLM to test loop mechanics (no tool calls on vague queries).
    Integration: real LLM to verify vague queries produce questions, not cards.

Layer 3 (TestAgentEndToEnd):
    Real LLM + real ChromaDB. Uses multi-turn helper so clarification turns are
    handled before asserting on candidate card tokens.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from agent.tools import search_candidates
from agent.agent import collect_response
from core.schemas import ParsedCV, EmployeeProfile


# ── Multi-turn helper ─────────────────────────────────────────────────────────

def drive_until_candidates(
    turns: list[str],
    store,
    embedder,
    settings,
) -> str:
    """
    Send queries in sequence until __CANDIDATE_CARD__: appears or all turns are used.
    Returns the last response text.
    """
    history = []
    response = ""
    for query in turns:
        response, history = collect_response(query, history, store, embedder, settings)
        if "__CANDIDATE_CARD__:" in response:
            break
    return response


# ── Mock helpers (reused from test_agent.py pattern) ─────────────────────────

def make_llm_text_response(content: str):
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def make_llm_tool_response(tool_name: str, tool_args: dict, call_id: str = "call_001"):
    call = MagicMock()
    call.id = call_id
    call.function.name = tool_name
    call.function.arguments = json.dumps(tool_args)
    msg = MagicMock()
    msg.tool_calls = [call]
    msg.content = None
    msg.model_dump.return_value = {"role": "assistant", "tool_calls": []}
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def make_mock_store_with_emp(employee_id: str, name: str, role: str, dept: str):
    store = MagicMock()
    cv = ParsedCV(employee_id=employee_id, full_name=name, current_role=role, department=dept, skills=["Python"])
    profile = EmployeeProfile(employee_id=employee_id, cv=cv)
    store.search.return_value = [{
        "employee_id": employee_id,
        "full_name": name,
        "current_role": role,
        "department": dept,
        "skills": ["Python"],
        "similarity": 0.9,
        "profile_json": profile.model_dump_json(),
    }]
    store.get_profile.return_value = profile
    store.get_all_profiles.return_value = [profile]
    return store


def make_mock_embedder():
    e = MagicMock()
    e.embed.return_value = [0.1] * 1536
    return e


# ── Layer 1: Retrieval correctness ────────────────────────────────────────────

@pytest.mark.integration
class TestRetrievalCorrectness:
    """
    Ground truth verified by running actual queries before writing these assertions.
    All expected IDs below were confirmed to appear in real store results.
    """

    def test_data_engineer_query(self, real_store, real_embedder):
        result = search_candidates(
            "data engineer with Python and Apache Airflow", 3, real_store, real_embedder
        )
        ids = [c["employee_id"] for c in result["candidates"]]
        assert "emp_002" in ids, f"Expected emp_002 (Arini Kusumawati, Data Engineer) in {ids}"

    def test_data_analyst_query(self, real_store, real_embedder):
        result = search_candidates(
            "data analyst SQL Python Power BI", 3, real_store, real_embedder
        )
        ids = [c["employee_id"] for c in result["candidates"]]
        assert "emp_021" in ids, f"Expected emp_021 (Tommy Hartono Susilo, Data Analyst) in {ids}"

    def test_internal_audit_query(self, real_store, real_embedder):
        result = search_candidates(
            "internal auditor financial control SOX compliance", 3, real_store, real_embedder
        )
        ids = [c["employee_id"] for c in result["candidates"]]
        assert "emp_010" in ids, f"Expected emp_010 (Indah Permata Sari, Internal Auditor) in {ids}"

    def test_hr_recruitment_query(self, real_store, real_embedder):
        result = search_candidates(
            "HR talent acquisition specialist sourcing ATS", 3, real_store, real_embedder
        )
        ids = [c["employee_id"] for c in result["candidates"]]
        assert "emp_012" in ids, f"Expected emp_012 (Kartika Sari Dewi, Talent Acquisition) in {ids}"

    def test_cloud_infrastructure_query(self, real_store, real_embedder):
        result = search_candidates(
            "cloud infrastructure engineer AWS Kubernetes Terraform", 3, real_store, real_embedder
        )
        ids = [c["employee_id"] for c in result["candidates"]]
        assert "emp_004" in ids, f"Expected emp_004 (Citra Dewi Lestari, Cloud Infra Engineer) in {ids}"

    def test_legal_query(self, real_store, real_embedder):
        result = search_candidates(
            "legal counsel contract drafting regulatory compliance", 2, real_store, real_embedder
        )
        ids = [c["employee_id"] for c in result["candidates"]]
        assert "emp_020" in ids, f"Expected emp_020 (Siti Nurhaliza Rahmat, Legal Counsel) in {ids}"


# ── Layer 2: Agent clarification behavior ─────────────────────────────────────

class TestAgentClarificationBehavior:
    """Unit tests — mocked LLM, no API key required."""

    def test_vague_query_no_tool_call(self, mock_settings):
        """When LLM decides to clarify, the loop must end after 1 LLM call (no tools dispatched)."""
        store = make_mock_store_with_emp("emp_002", "Arini", "Data Engineer", "IT")
        embedder = make_mock_embedder()
        clarify_resp = make_llm_text_response("Untuk posisi apa yang Anda cari di IT?")

        with patch("agent.agent.litellm.completion", return_value=clarify_resp) as mock_llm:
            text, history = collect_response("cari orang IT", [], store, embedder, mock_settings)

        assert mock_llm.call_count == 1  # no second call = no tools dispatched
        assert "?" in text
        assert "__CANDIDATE_CARD__:" not in text

    def test_specific_query_dispatches_tool_then_responds(self, mock_settings):
        """A specific query triggers search_candidates, then LLM produces a final response."""
        store = make_mock_store_with_emp("emp_002", "Arini Kusumawati", "Data Engineer", "IT")
        embedder = make_mock_embedder()

        tool_resp = make_llm_tool_response("search_candidates", {"query": "data engineer Python", "top_k": 5})
        final_resp = make_llm_text_response(
            "Saya menemukan 1 kandidat:\n**Arini Kusumawati**\n__CANDIDATE_CARD__:emp_002"
        )

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return tool_resp if call_count == 1 else final_resp

        with patch("agent.agent.litellm.completion", side_effect=side_effect):
            text, history = collect_response(
                "Data engineer dengan Python", [], store, embedder, mock_settings
            )

        assert call_count == 2
        assert "__CANDIDATE_CARD__:emp_002" in text

    def test_history_passed_correctly_on_clarification(self, mock_settings):
        """After a clarification exchange, the full conversation history is sent to the LLM."""
        store = make_mock_store_with_emp("emp_012", "Kartika", "Talent Acquisition", "HR")
        embedder = make_mock_embedder()
        clarify_resp = make_llm_text_response("Level apa yang Anda cari? Junior, mid, atau senior?")

        with patch("agent.agent.litellm.completion", return_value=clarify_resp) as mock_llm:
            _, history = collect_response("cari kandidat HR", [], store, embedder, mock_settings)

        # history should now include user message + assistant clarification
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles


@pytest.mark.integration
class TestAgentClarificationBehaviorIntegration:
    """Integration variants — real LLM needed."""

    def test_vague_it_query_does_not_return_cards(self, real_store, real_embedder, real_settings):
        text, _ = collect_response("cari orang IT", [], real_store, real_embedder, real_settings)
        assert "__CANDIDATE_CARD__:" not in text, (
            "Vague 'cari orang IT' should produce a clarification, not candidate cards"
        )

    def test_vague_it_query_contains_question(self, real_store, real_embedder, real_settings):
        text, _ = collect_response("cari orang IT", [], real_store, real_embedder, real_settings)
        assert "?" in text, "Agent should ask a clarifying question for vague queries"

    def test_vague_finance_query_does_not_return_cards(self, real_store, real_embedder, real_settings):
        text, _ = collect_response("find a finance candidate", [], real_store, real_embedder, real_settings)
        assert "__CANDIDATE_CARD__:" not in text, (
            "Vague 'find a finance candidate' should trigger clarification"
        )


# ── Layer 3: End-to-end output correctness ────────────────────────────────────

@pytest.mark.integration
class TestAgentEndToEnd:
    """
    Real LLM + real ChromaDB.
    Uses drive_until_candidates() so clarification turns are handled before asserting.
    Each test provides a primary query + 1 fallback that forces the agent to search.
    """

    def test_returns_data_engineer(self, real_store, real_embedder, real_settings):
        response = drive_until_candidates(
            [
                "Data engineer dengan pengalaman Python, Apache Spark, dan Airflow",
                "Mid-level, department IT, tolong cari sekarang",
            ],
            real_store, real_embedder, real_settings,
        )
        assert "__CANDIDATE_CARD__:emp_002" in response, (
            f"Expected emp_002 (Arini, Data Engineer) in response.\nResponse: {response[:500]}"
        )

    def test_returns_internal_auditor(self, real_store, real_embedder, real_settings):
        response = drive_until_candidates(
            [
                "Internal auditor dengan keahlian SOX compliance dan CISA",
                "Mid-level, Finance department, silakan cari kandidatnya",
            ],
            real_store, real_embedder, real_settings,
        )
        assert "__CANDIDATE_CARD__:emp_010" in response, (
            f"Expected emp_010 (Indah, Internal Auditor) in response.\nResponse: {response[:500]}"
        )

    def test_returns_talent_acquisition(self, real_store, real_embedder, real_settings):
        response = drive_until_candidates(
            [
                "Talent acquisition specialist dengan pengalaman sourcing dan ATS",
                "Mid-level, HR department, tolong tampilkan kandidat",
            ],
            real_store, real_embedder, real_settings,
        )
        assert "__CANDIDATE_CARD__:emp_012" in response, (
            f"Expected emp_012 (Kartika, Talent Acquisition) in response.\nResponse: {response[:500]}"
        )

    def test_returns_cloud_or_devops_engineer(self, real_store, real_embedder, real_settings):
        response = drive_until_candidates(
            [
                "Cloud infrastructure atau DevOps engineer dengan Kubernetes dan AWS",
                "Mid to senior level, IT department, cari sekarang",
            ],
            real_store, real_embedder, real_settings,
        )
        has_cloud = "__CANDIDATE_CARD__:emp_004" in response  # Citra, Cloud Infra
        has_devops = "__CANDIDATE_CARD__:emp_005" in response  # Dimas, DevOps
        assert has_cloud or has_devops, (
            f"Expected emp_004 or emp_005 in response.\nResponse: {response[:500]}"
        )

    def test_returns_legal_counsel(self, real_store, real_embedder, real_settings):
        response = drive_until_candidates(
            [
                "Legal counsel dengan keahlian contract drafting dan regulatory compliance",
                "Any level, Legal department",
            ],
            real_store, real_embedder, real_settings,
        )
        assert "__CANDIDATE_CARD__:emp_020" in response, (
            f"Expected emp_020 (Siti, Legal Counsel) in response.\nResponse: {response[:500]}"
        )
