import hashlib

import streamlit as st

from session.store import validate_access_key


def render_access_gate(db_path: str) -> None:
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("### Access Required")
        st.caption("Enter the access key you received to use TalentScope Chat.")
        with st.form("access_key_form", clear_on_submit=True):
            raw_key = st.text_input(
                "Access Key",
                placeholder="XXXX-XXXX-XXXX-XXXX",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

        if submitted:
            if not raw_key.strip():
                st.error("Please enter an access key.")
            elif validate_access_key(db_path, raw_key):
                key_hash = hashlib.sha256(raw_key.strip().upper().encode()).hexdigest()
                st.session_state["chat_access_granted"] = True
                st.query_params["access_key_hash"] = key_hash
                st.rerun()
            else:
                st.error("Invalid, expired, or revoked key.")
