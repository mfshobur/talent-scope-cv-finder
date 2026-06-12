"""
Inline candidate card rendered inside chat messages.
Clicking "View Profile" sets st.session_state["selected_candidate"].
"""

import streamlit as st
from core.schemas import CandidateMatch, EmployeeProfile


def render_inline_card(profile: EmployeeProfile, match_reasons: list[str] | None = None, fit_gaps: list[str] | None = None, relevance_score: float | None = None):
    eid = profile.employee_id
    cv = profile.cv

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{cv.full_name}**")
            st.caption(f"{cv.current_role} · {cv.department}")
            if cv.skills:
                skills_display = " · ".join(cv.skills[:5])
                st.caption(f"Skills: {skills_display}")

        with col2:
            if relevance_score is not None:
                pct = int(relevance_score * 100)
                color = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 60 else "#94a3b8"
                st.markdown(
                    f'<div style="text-align:center;font-size:1.4rem;font-weight:700;color:{color}">{pct}%</div>',
                    unsafe_allow_html=True,
                )
                st.caption("match")

        if match_reasons:
            for reason in match_reasons[:2]:
                st.markdown(f"✓ {reason}")
        if fit_gaps:
            for gap in fit_gaps[:1]:
                st.markdown(f"⚠ {gap}")

        if st.button("View Profile", key=f"view_{eid}", use_container_width=True):
            st.session_state["selected_candidate"] = profile
            st.session_state["selected_match"] = {
                "match_reasons": match_reasons or [],
                "fit_gaps": fit_gaps or [],
                "relevance_score": relevance_score,
            }
            st.rerun()
