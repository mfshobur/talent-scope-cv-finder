"""
TalentScope — AI-powered internal candidate finder
Entry point for Streamlit app.
"""

import uuid
import streamlit as st

from config.settings import Settings
from core.embedder import Embedder
from ingestion.chromadb_store import CandidateStore
from session.store import init_db, load_history
from ui.sidebar import render_sidebar
from ui.chat_panel import render_chat_panel
from ui.browse_view import render_browse_view
from ui.detail_panel import render_detail_panel

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

else:
    # Chat view — split panel if a candidate is selected
    selected = st.session_state.get("selected_candidate")
    if selected:
        chat_col, detail_col = st.columns([1, 1])
        with chat_col:
            render_chat_panel(store, embedder, settings, session_id, db_path)
        with detail_col:
            render_detail_panel(selected)
    else:
        render_chat_panel(store, embedder, settings, session_id, db_path)
