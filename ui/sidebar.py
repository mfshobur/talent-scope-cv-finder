import streamlit as st
from config.settings import Settings

QUICK_SEARCHES = [
    ("🤖 AI/ML Engineer", "Cari kandidat untuk posisi AI/ML Engineer, pengalaman Python dan machine learning"),
    ("📊 Data Analyst", "Cari kandidat untuk posisi Data Analyst, menguasai SQL, Python, dan Power BI"),
    ("👥 HR Business Partner", "Cari kandidat untuk posisi HR Business Partner dengan pengalaman talent management"),
    ("💰 Financial Controller", "Cari kandidat untuk posisi Financial Controller, pengalaman akuntansi dan SAP"),
    ("📣 Marketing Manager", "Cari kandidat untuk posisi Marketing Manager, pengalaman brand strategy"),
]

MODEL_DEFAULTS = {
    "openai": "openai/gpt-5-nano",
    "anthropic": "anthropic/claude-haiku-4-5-20251001",
    "groq": "groq/llama-3.3-70b-versatile",
    "ollama": "ollama/qwen2.5",
}


def render_sidebar() -> Settings:
    with st.sidebar:
        # ── Brand ─────────────────────────────────────────────────────────────
        col_logo, col_title = st.columns([1, 3])
        with col_logo:
            st.image("assets/logo.png", width=52)
        with col_title:
            st.markdown(
                """
                <div style="padding: 0.4rem 0 0 0;">
                    <div style="font-size: 1.3rem; font-weight: 800; letter-spacing: -0.5px;">
                        TalentScope
                    </div>
                    <div style="font-size: 0.72rem; opacity: 0.5; margin-top: 1px;">
                        Internal talent mobility · AI-powered
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Apply any programmatic nav request before the radio is instantiated
        if "pending_nav" in st.session_state:
            st.session_state["nav"] = st.session_state.pop("pending_nav")

        # ── Navigation ────────────────────────────────────────────────────────
        NAV_LABELS = {
            "Chat": "💬  Chat",
            "Browse All": "👤  Browse Employees",
            "Admin": "⚙️  Admin",
        }
        st.radio(
            "nav",
            list(NAV_LABELS.keys()),
            format_func=lambda x: NAV_LABELS[x],
            label_visibility="collapsed",
            key="nav",  # directly drives st.session_state["nav"]
        )

        st.divider()

        # ── Quick searches ────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.75rem;font-weight:600;opacity:0.5;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">Quick Search</p>',
            unsafe_allow_html=True,
        )
        for label, query in QUICK_SEARCHES:
            if st.button(label, use_container_width=True, key=f"qs_{label}"):
                st.session_state["pending_query"] = query
                st.session_state["nav"] = "Chat"

        st.divider()

        # ── Model config (collapsed) ──────────────────────────────────────────
        with st.expander("⚙ Model Configuration", expanded=False):
            saved = st.session_state.get("settings", Settings())

            provider = st.selectbox(
                "Provider",
                ["openai", "anthropic", "groq", "ollama"],
                index=0,
                key="sb_provider",
            )
            default_model = saved.llm_model or MODEL_DEFAULTS[provider]
            llm_model = st.text_input("Model", value=default_model, key="sb_model")
            api_key = st.text_input("API Key", value=saved.llm_api_key, type="password", key="sb_apikey")
            base_url = st.text_input(
                "Base URL",
                value=saved.llm_base_url,
                placeholder="http://localhost:11434 (Ollama only)",
                key="sb_baseurl",
            )

            if st.button("Save Config", use_container_width=True, type="primary"):
                new_settings = Settings(
                    llm_model=llm_model,
                    llm_api_key=api_key,
                    llm_base_url=base_url,
                    vision_model=llm_model,
                    embedding_model=saved.embedding_model,
                    chroma_persist_dir=saved.chroma_persist_dir,
                    session_db_path=saved.session_db_path,
                )
                st.session_state["settings"] = new_settings
                st.success("Saved!")

    return st.session_state.get("settings", Settings())
