"""
Full candidate profile panel — shown on the right side when a card is clicked.
"""

from pathlib import Path
import streamlit as st
from core.schemas import EmployeeProfile

ASSESSMENT_LABELS = {
    "psychotest": "Psikotes",
    "technical": "Technical Test",
    "behavioral": "Behavioral Assessment",
    "english_proficiency": "English Proficiency",
}

DATA_DIR = Path(__file__).parent.parent / "data" / "employees"


def render_detail_panel(profile: EmployeeProfile):
    cv = profile.cv
    match_info = st.session_state.get("selected_match", {})

    st.markdown(f"### {cv.full_name}")
    st.caption(f"{cv.current_role} · {cv.department} · {cv.location}")
    st.divider()

    tab_overview, tab_cv, tab_assessments, tab_files = st.tabs(
        ["Overview", "CV Details", "Assessments", "Files"]
    )

    with tab_overview:
        if cv.summary:
            st.markdown(cv.summary)

        if match_info.get("relevance_score") is not None:
            pct = int(match_info["relevance_score"] * 100)
            st.progress(match_info["relevance_score"], text=f"Match score: {pct}%")

        if match_info.get("match_reasons"):
            st.markdown("**Why they match:**")
            for r in match_info["match_reasons"]:
                st.markdown(f"✓ {r}")

        if match_info.get("fit_gaps"):
            st.markdown("**Development areas:**")
            for g in match_info["fit_gaps"]:
                st.markdown(f"⚠ {g}")

        if cv.skills:
            st.markdown("**Skills:**")
            cols = st.columns(4)
            for i, skill in enumerate(cv.skills):
                cols[i % 4].markdown(
                    f'<span style="background:#1e40af;color:#e0f2fe;padding:2px 8px;border-radius:12px;font-size:0.8rem;display:inline-block">{skill}</span>',
                    unsafe_allow_html=True,
                )

    with tab_cv:
        if cv.work_experience:
            st.markdown("#### Work Experience")
            for exp in cv.work_experience:
                end = str(exp.end_year) if exp.end_year else "Present"
                st.markdown(f"**{exp.role}** at {exp.company} ·  *{exp.start_year}–{end}*")
                if exp.description:
                    st.markdown(exp.description)
                st.divider()

        if cv.education:
            st.markdown("#### Education")
            for edu in cv.education:
                gpa_str = f" · GPA: {edu.gpa}" if edu.gpa else ""
                year_str = f" ({edu.graduation_year})" if edu.graduation_year else ""
                st.markdown(f"**{edu.degree}** in {edu.field}, {edu.institution}{year_str}{gpa_str}")

        if cv.certifications:
            st.markdown("#### Certifications")
            for cert in cv.certifications:
                st.markdown(f"• {cert}")

        if cv.languages:
            st.markdown("#### Languages")
            for lang in cv.languages:
                st.markdown(f"• {lang.language}: {lang.level.capitalize()}")

    with tab_assessments:
        if not profile.assessments:
            st.info("No assessment data available for this employee.")
        for asm in profile.assessments:
            label = ASSESSMENT_LABELS.get(asm.category, asm.category.replace("_", " ").title())
            with st.expander(f"{label} — {asm.score_label or 'See details'}", expanded=False):
                if asm.key_findings:
                    st.markdown("**Key Findings:**")
                    for f in asm.key_findings:
                        st.markdown(f"• {f}")
                if asm.strengths:
                    st.markdown("**Strengths:**")
                    for s in asm.strengths:
                        st.markdown(f"✓ {s}")
                if asm.development_areas:
                    st.markdown("**Development Areas:**")
                    for d in asm.development_areas:
                        st.markdown(f"⚠ {d}")

    with tab_files:
        emp_dir = DATA_DIR / profile.employee_id
        cv_path = emp_dir / "cv.pdf"
        if cv_path.exists():
            with open(cv_path, "rb") as f:
                st.download_button(
                    "⬇ Download CV",
                    f,
                    file_name=f"{cv.full_name.replace(' ', '_')}_CV.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        asm_dir = emp_dir / "assessments"
        if asm_dir.exists():
            st.markdown("**Assessment Files:**")
            for asm_file in sorted(asm_dir.glob("*.pdf")):
                label = ASSESSMENT_LABELS.get(asm_file.stem, asm_file.stem.replace("_", " ").title())
                with open(asm_file, "rb") as f:
                    st.download_button(
                        f"⬇ {label}",
                        f,
                        file_name=asm_file.name,
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"dl_{profile.employee_id}_{asm_file.stem}",
                    )
