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
                _TOOL_STATUS = {
                    "search_candidates": lambda a: f"Searching: *{a.get('query', '')}*",
                    "get_candidate_detail": lambda a: f"Getting profile: *{a.get('employee_id', '')}*",
                    "compare_candidates": lambda a: f"Comparing {len(a.get('employee_ids', []))} candidates...",
                    "filter_candidates": lambda a: "Filtering candidates...",
                }

                _SPINNER = (
                    "<style>@keyframes _tsspin{to{transform:rotate(360deg)}}"
                    "._tsspin{display:inline-block;width:13px;height:13px;"
                    "border:2px solid rgba(255,255,255,0.15);border-top-color:#e2e8f0;"
                    "border-radius:50%;animation:_tsspin .75s linear infinite;"
                    "vertical-align:middle;margin-right:7px}</style>"
                )

                def _status_md(text: str) -> str:
                    return f'{_SPINNER}<span class="_tsspin"></span><em>{text}</em>'

                status_placeholder = st.empty()
                status_placeholder.markdown(_status_md("Thinking..."), unsafe_allow_html=True)

                def on_tool_call(tool_name: str, args: dict):
                    label = _TOOL_STATUS.get(tool_name, lambda a: "Processing...")(args)
                    status_placeholder.markdown(_status_md(label), unsafe_allow_html=True)

                try:
                    response_text, updated_history = collect_response(
                        query=query,
                        history=[m for m in history[:-1] if m["role"] in ("user", "assistant", "tool")],
                        store=store,
                        embedder=embedder,
                        settings=settings,
                        on_tool_call=on_tool_call,
                    )
                    status_placeholder.empty()
                    _render_message_content(response_text, store, msg_index=len(history))
                    st.session_state["messages"] = updated_history
                    save_history(db_path, session_id, updated_history)
                    found_ids = _CARD_TOKEN_RE.findall(response_text)
                    if found_ids:
                        st.session_state["nine_box_emp_ids"] = found_ids
                        st.session_state["show_nine_box"] = False
                except Exception as e:
                    status_placeholder.empty()
                    error_msg = f"⚠ Error: {e}"
                    st.error(error_msg)
                    history.append({"role": "assistant", "content": error_msg})
                    st.session_state["messages"] = history

        st.rerun()
