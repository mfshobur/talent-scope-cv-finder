"""
TalentScope — AI-powered internal candidate finder
Entry point for Streamlit app.
"""

import uuid
import streamlit as st

from config.settings import Settings
from core.embedder import Embedder
from ingestion.chromadb_store import CandidateStore
from session.store import (
    init_db, load_history,
    generate_access_key, revoke_access_key, list_access_keys, _validate_hash,
)
from ui.sidebar import render_sidebar
from ui.chat_panel import render_chat_panel
from ui.browse_view import render_browse_view
from ui.detail_panel import render_detail_panel
from ui.access_gate import render_access_gate

st.set_page_config(
    page_title="TalentScope",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session init ──────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    params = st.query_params
    if "session_id" in params:
        st.session_state["session_id"] = params["session_id"]
    else:
        new_id = str(uuid.uuid4())
        st.session_state["session_id"] = new_id
        st.query_params["session_id"] = new_id

if "settings" not in st.session_state:
    st.session_state["settings"] = Settings()

settings: Settings = render_sidebar()

db_path = settings.session_db_path
init_db(db_path)

session_id = st.session_state["session_id"]
if "messages" not in st.session_state:
    st.session_state["messages"] = load_history(db_path, session_id)

# ── Resources (cached per session) ────────────────────────────────────────────

@st.cache_resource
def get_store(chroma_dir: str) -> CandidateStore:
    return CandidateStore(chroma_dir)


@st.cache_resource
def get_embedder(_settings_key: str) -> Embedder:
    return Embedder(st.session_state.get("settings", Settings()))


store = get_store(settings.chroma_persist_dir)
embedder = get_embedder(settings.llm_model + settings.effective_embedding_api_key())

# ── Auto-restore chat access from URL param ───────────────────────────────────

if not st.session_state.get("chat_access_granted"):
    stored_hash = st.query_params.get("access_key_hash")
    if stored_hash and _validate_hash(db_path, stored_hash):
        st.session_state["chat_access_granted"] = True

# ── Layout ────────────────────────────────────────────────────────────────────

nav = st.session_state.get("nav", "Chat")

if nav == "Browse All":
    selected = st.session_state.get("selected_candidate")
    if selected:
        browse_col, detail_col = st.columns([1, 1])
        with browse_col:
            render_browse_view(store)
        with detail_col:
            with st.container(height=750):
                render_detail_panel(selected)
    else:
        render_browse_view(store)

elif nav == "Admin":
    if not st.session_state.get("admin_authed"):
        st.markdown("### Admin Access")
        with st.form("admin_login_form"):
            pw = st.text_input("Password", type="password", label_visibility="collapsed",
                               placeholder="Admin password")
            if st.form_submit_button("Login", type="primary"):
                if pw == settings.admin_password and settings.admin_password:
                    st.session_state["admin_authed"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
    else:
        st.markdown("## Database Admin")
        col1, col2 = st.columns(2)
        with col1:
            doc_count = store.count()
            st.metric("Documents in DB", doc_count)
            st.metric("Employees", doc_count // 2)
        with col2:
            st.info(
                "To rebuild the database, run:\n"
                "```\npython scripts/run_ingestion.py\n```\n"
                "then re-deploy."
            )
        if st.button("Clear My Chat History"):
            from session.store import clear_session
            clear_session(db_path, session_id)
            st.session_state["messages"] = []
            st.success("Chat history cleared.")

        st.divider()

        st.markdown("### Generate Access Key")
        with st.form("key_gen_form"):
            label_input = st.text_input("Label (optional)", placeholder='e.g. "for Ahmad"')
            if st.form_submit_button("Generate Key", type="primary"):
                raw_key = generate_access_key(db_path, label=label_input.strip())
                st.session_state["last_generated_key"] = raw_key

        if "last_generated_key" in st.session_state:
            st.success("Copy this key now — it won't be shown again.")
            st.code(st.session_state["last_generated_key"], language=None)
            if st.button("Dismiss"):
                del st.session_state["last_generated_key"]

        st.divider()

        st.markdown("### Access Keys")
        keys = list_access_keys(db_path)
        if not keys:
            st.caption("No keys yet.")
        else:
            for k in keys:
                c1, c2, c3, c4 = st.columns([3, 1, 2, 1])
                c1.markdown(f"**{k['label'] or '(no label)'}**")
                c1.caption(k["key_hash"][:12] + "...")
                color = {"valid": "green", "expired": "orange", "revoked": "red"}[k["status"]]
                c2.markdown(
                    f'<span style="color:{color};font-weight:600">{k["status"]}</span>',
                    unsafe_allow_html=True,
                )
                c3.caption(k["expires_at"][:16].replace("T", " ") + " UTC")
                if k["status"] == "valid":
                    if c4.button("Revoke", key=f"revoke_{k['key_hash']}"):
                        revoke_access_key(db_path, k["key_hash"])
                        st.rerun()

else:
    # Chat view — gate behind access key
    if not st.session_state.get("chat_access_granted"):
        render_access_gate(db_path)
    else:
        selected = st.session_state.get("selected_candidate")
        if selected:
            chat_col, detail_col = st.columns([1, 1])
            with chat_col:
                with st.container(height=750, border=False):
                    render_chat_panel(store, embedder, settings, session_id, db_path)
            with detail_col:
                with st.container(height=750):
                    render_detail_panel(selected)
        else:
            render_chat_panel(store, embedder, settings, session_id, db_path)
