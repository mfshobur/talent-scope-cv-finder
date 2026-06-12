import json
import pytest
from unittest.mock import MagicMock, patch

from agent.agent import run_agent, collect_response
from core.schemas import ParsedCV, EmployeeProfile


def make_mock_store():
    from ingestion.chromadb_store import CandidateStore
    store = MagicMock(spec=CandidateStore)
    cv = ParsedCV(
        employee_id="emp_001",
        full_name="Muh. Shobur Fattah",
        current_role="AI/ML Engineer",
        department="IT",
        skills=["Python", "LangChain"],
    )
    profile = EmployeeProfile(employee_id="emp_001", cv=cv)
    store.search.return_value = [{
        "employee_id": "emp_001",
        "full_name": "Muh. Shobur Fattah",
        "current_role": "AI/ML Engineer",
        "department": "IT",
        "skills": ["Python", "LangChain"],
        "similarity": 0.95,
        "profile_json": profile.model_dump_json(),
    }]
    store.get_profile.return_value = profile
    store.get_all_profiles.return_value = [profile]
    return store


def make_mock_embedder():
    e = MagicMock()
    e.embed.return_value = [0.1] * 1536
    return e


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


class TestRunAgent:
    def test_direct_text_response(self, mock_settings):
        store = make_mock_store()
        embedder = make_mock_embedder()
        text_resp = make_llm_text_response("Bisakah Anda menjelaskan posisi yang dicari?")

        with patch("agent.agent.litellm.completion", return_value=text_resp):
            gen = run_agent("cari orang", [], store, embedder, mock_settings)
            chunks = []
            updated_history = []
            try:
                while True:
                    chunks.append(next(gen))
            except StopIteration as e:
                updated_history = e.value

        full_text = "".join(chunks)
        assert "Bisakah" in full_text
        assert any(m["role"] == "assistant" for m in updated_history)

    def test_tool_call_then_response(self, mock_settings):
        store = make_mock_store()
        embedder = make_mock_embedder()

        tool_resp = make_llm_tool_response("search_candidates", {"query": "AI ML engineer", "top_k": 5})
        final_resp = make_llm_text_response("Saya menemukan 1 kandidat:\n\n**Muh. Shobur Fattah**\n__CANDIDATE_CARD__:emp_001")

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return tool_resp if call_count == 1 else final_resp

        with patch("agent.agent.litellm.completion", side_effect=side_effect):
            text, history = collect_response("AI ML engineer Python", [], store, embedder, mock_settings)

        assert "emp_001" in text or "Shobur" in text
        assert call_count == 2  # once for tool call, once for final response

    def test_history_preserved(self, mock_settings):
        store = make_mock_store()
        embedder = make_mock_embedder()
        prior_history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        text_resp = make_llm_text_response("Tentu, apa yang Anda cari?")

        with patch("agent.agent.litellm.completion", return_value=text_resp) as mock_llm:
            collect_response("I need help", prior_history, store, embedder, mock_settings)
            call_args = mock_llm.call_args
            messages_sent = call_args.kwargs["messages"]

        # Prior history should be in the messages sent to LLM
        assert any(m.get("content") == "Hello" for m in messages_sent)
        assert any(m.get("content") == "Hi" for m in messages_sent)


class TestSessionStore:
    def test_save_and_load(self, tmp_path):
        from session.store import init_db, save_history, load_history
        db = str(tmp_path / "test.db")
        init_db(db)
        messages = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "reply"}]
        save_history(db, "session-abc", messages)
        loaded = load_history(db, "session-abc")
        assert loaded == messages

    def test_load_empty_session(self, tmp_path):
        from session.store import init_db, load_history
        db = str(tmp_path / "test.db")
        init_db(db)
        result = load_history(db, "nonexistent")
        assert result == []

    def test_clear_session(self, tmp_path):
        from session.store import init_db, save_history, load_history, clear_session
        db = str(tmp_path / "test.db")
        init_db(db)
        save_history(db, "session-abc", [{"role": "user", "content": "hi"}])
        clear_session(db, "session-abc")
        assert load_history(db, "session-abc") == []

    def test_overwrite_history(self, tmp_path):
        from session.store import init_db, save_history, load_history
        db = str(tmp_path / "test.db")
        init_db(db)
        save_history(db, "s1", [{"role": "user", "content": "first"}])
        save_history(db, "s1", [{"role": "user", "content": "second"}])
        loaded = load_history(db, "s1")
        assert loaded[0]["content"] == "second"
