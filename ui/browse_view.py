import streamlit as st
from ingestion.chromadb_store import CandidateStore


def render_browse_view(store: CandidateStore):
    st.markdown("## Browse All Employees")

    profiles = store.get_all_profiles()
    if not profiles:
        st.warning("No employees in database. Run the ingestion script first.")
        return

    # Build dataframe-friendly rows
    rows = []
    for p in profiles:
        cv = p.cv
        rows.append({
            "ID": p.employee_id,
            "Name": cv.full_name,
            "Role": cv.current_role,
            "Department": cv.department,
            "Location": cv.location or "—",
            "Skills (preview)": ", ".join(cv.skills[:4]) or "—",
            "_profile": p,
        })

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        dept_options = ["All"] + sorted({r["Department"] for r in rows if r["Department"]})
        selected_dept = st.selectbox("Filter by Department", dept_options)
    with col2:
        search_text = st.text_input("Search by name or skill", placeholder="e.g. Python, Arini")

    filtered = rows
    if selected_dept != "All":
        filtered = [r for r in filtered if r["Department"] == selected_dept]
    if search_text:
        q = search_text.lower()
        filtered = [
            r for r in filtered
            if q in r["Name"].lower() or q in r["Skills (preview)"].lower() or q in r["Role"].lower()
        ]

    st.caption(f"Showing {len(filtered)} of {len(rows)} employees")

    with st.container(height=650):
        for row in filtered:
            profile = row["_profile"]
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 2, 1])
                with c1:
                    st.markdown(f"**{row['Name']}**")
                    st.caption(f"{row['Role']} · {row['Department']}")
                with c2:
                    st.caption(row["Skills (preview)"])
                with c3:
                    if st.button("View", key=f"browse_view_{profile.employee_id}"):
                        st.session_state["pending_profile"] = profile
                        st.session_state["selected_match"] = {}
