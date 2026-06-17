"""
Full candidate profile panel — shown on the right side when a card is clicked.
"""

from pathlib import Path
import streamlit as st
from core.schemas import EmployeeProfile
from ui.nine_box import load_kpi, load_akhlak

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

    tab_overview, tab_perf, tab_cv, tab_assessments, tab_files = st.tabs(
        ["Overview", "Performance", "CV Details", "Assessments", "Files"]
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

    with tab_perf:
        eid = profile.employee_id
        kpi = load_kpi().get(eid)
        akhlak = load_akhlak().get(eid)

        if kpi:
            st.markdown("#### Performance (KPI)")
            score = kpi["score"]
            label = kpi["label"]
            kpi_color = "#16a34a" if score >= 80 else "#ca8a04" if score >= 60 else "#dc2626"
            col_score, col_label = st.columns([1, 2])
            col_score.markdown(
                f"<div style='font-size:2.5rem;font-weight:700;color:{kpi_color};line-height:1'>"
                f"{score}<span style='font-size:1rem'>%</span></div>",
                unsafe_allow_html=True,
            )
            col_label.markdown(
                f"<div style='margin-top:0.6rem;font-size:1rem;color:{kpi_color};font-weight:600'>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )
            st.progress(score / 100)
        else:
            st.info("No KPI data available.")

        st.divider()

        if akhlak:
            st.markdown("#### Potential (AKHLAK)")
            composite = akhlak["composite"]
            comp_color = "#16a34a" if composite >= 80 else "#ca8a04" if composite >= 60 else "#dc2626"
            st.markdown(
                f"<div style='font-size:1rem;font-weight:600;color:{comp_color};"
                f"margin-bottom:4px'>Composite: {composite}%</div>",
                unsafe_allow_html=True,
            )
            st.progress(composite / 100)
            st.markdown("")

            dims = [
                ("Amanat",      "amanat"),
                ("Kompeten",    "kompeten"),
                ("Harmonis",    "harmonis"),
                ("Loyal",       "loyal"),
                ("Adaptif",     "adaptif"),
                ("Kolaboratif", "kolaboratif"),
            ]
            for label_dim, key in dims:
                val = akhlak[key]
                bar_color = "#16a34a" if val >= 80 else "#ca8a04" if val >= 60 else "#dc2626"
                col_name, col_bar, col_val = st.columns([2, 5, 1])
                col_name.caption(label_dim)
                col_bar.markdown(
                    f"<div style='margin-top:6px;background:#1e293b;border-radius:4px;height:8px'>"
                    f"<div style='width:{val}%;background:{bar_color};border-radius:4px;height:8px'></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                col_val.caption(f"{val}%")
        else:
            st.info("No AKHLAK data available.")

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
