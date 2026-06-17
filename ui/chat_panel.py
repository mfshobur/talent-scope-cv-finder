"""
Chat panel: renders conversation history with inline candidate cards,
handles input, and runs the agent.
"""

import re
import streamlit as st

from agent.agent import collect_response
from core.embedder import Embedder
from config.settings import Settings
from ingestion.chromadb_store import CandidateStore
from session.store import save_history
from ui.candidate_card import render_inline_card

_CARD_TOKEN_RE = re.compile(r"__CANDIDATE_CARD__:(emp_\w+)")


def _render_message_content(content: str, store: CandidateStore, msg_index: int = 0):
    """Render a message, replacing __CANDIDATE_CARD__:emp_XXX tokens with cards."""
    parts = _CARD_TOKEN_RE.split(content)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            for line in part.split("\n"):
                if line.strip():
                    st.markdown(line)
        else:
            profile = store.get_profile(part)
            if profile:
                render_inline_card(profile, card_key=f"msg{msg_index}_{part}")
            else:
                st.caption(f"[Candidate {part} not found in database]")


def render_chat_panel(store: CandidateStore, embedder: Embedder, settings: Settings, session_id: str, db_path: str):
    history = st.session_state.get("messages", [])

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg_index, msg in enumerate(history):
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content") or ""
            if not content:
                continue
            with st.chat_message(role):
                if role == "assistant":
                    _render_message_content(content, store, msg_index=msg_index)
                else:
                    st.markdown(content)

    # Handle pending quick-search query
    pending = st.session_state.pop("pending_query", None)

    # Input
    user_input = st.chat_input("Describe the vacancy or ask about candidates...")
    query = pending or user_input

    if query:
        # Show user message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)

        history.append({"role": "user", "content": query})

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response_text, updated_history = collect_response(
                            query=query,
                            history=[m for m in history[:-1] if m["role"] in ("user", "assistant", "tool")],
                            store=store,
                            embedder=embedder,
                            settings=settings,
                        )
                        _render_message_content(response_text, store, msg_index=len(history))
                        st.session_state["messages"] = updated_history
                        save_history(db_path, session_id, updated_history)
                    except Exception as e:
                        error_msg = f"⚠ Error: {e}"
                        st.error(error_msg)
                        history.append({"role": "assistant", "content": error_msg})
                        st.session_state["messages"] = history

        st.rerun()
