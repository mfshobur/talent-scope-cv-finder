"""
9-Box Talent Matrix component.
X axis = Performance (KPI score from mock_kpi.json)
Y axis = Potential (AKHLAK composite score from mock_akhlak.json)
"""

import json
from functools import cache
from pathlib import Path

import streamlit as st

from ingestion.chromadb_store import CandidateStore

GRID_LABELS = [
    ["Enigma",         "High Potential ⬆", "Star ⭐"],        # High Potential row
    ["Inconsistent",   "Core Employee",     "High Performer"], # Mid Potential row
    ["Underperformer", "Solid Worker",      "Trusted Pro"],    # Low Potential row
]

# Diagonal color gradient: red (bottom-left) → green (top-right)
CELL_COLORS = [
    ["#ca8a04", "#16a34a", "#15803d"],  # High Potential
    ["#c2410c", "#ca8a04", "#16a34a"],  # Mid Potential
    ["#991b1b", "#c2410c", "#ca8a04"],  # Low Potential
]


@cache
def load_kpi(path: str = "data/mock_kpi.json") -> dict:
    return json.loads(Path(path).read_text())


@cache
def load_akhlak(path: str = "data/mock_akhlak.json") -> dict:
    return json.loads(Path(path).read_text())


def kpi_bucket(score: float) -> int:
    """0 = Low (<60), 1 = Medium (60–79), 2 = High (≥80)"""
    if score >= 80:
        return 2
    if score >= 60:
        return 1
    return 0


def potential_bucket(score: float) -> int:
    """0 = Low (<60), 1 = Medium (60–79), 2 = High (≥80)"""
    if score >= 80:
        return 2
    if score >= 60:
        return 1
    return 0


def place_candidate(kpi_score: float, potential_score: float) -> tuple[int, int]:
    """
    Returns (row, col) for the 9-box grid.
    row 0 = High Potential (top), row 2 = Low Potential (bottom).
    col 0 = Low Performance (left), col 2 = High Performance (right).
    """
    col = kpi_bucket(kpi_score)
    row = 2 - potential_bucket(potential_score)
    return row, col


def render_nine_box(emp_ids: list[str], store: CandidateStore) -> None:
    kpi_data = load_kpi()
    akhlak_data = load_akhlak()

    # Place each candidate in their cell
    cells: list[list[list]] = [[[] for _ in range(3)] for _ in range(3)]
    for eid in emp_ids:
        kpi_score = kpi_data.get(eid, {}).get("score", 50)
        potential_score = akhlak_data.get(eid, {}).get("composite", 50)
        row, col = place_candidate(kpi_score, potential_score)
        profile = store.get_profile(eid)
        if profile:
            cells[row][col].append((profile.cv.full_name, eid, kpi_score, potential_score, profile))

    st.markdown("#### 9-Box Talent Matrix")
    st.caption("**X** — Performance (KPI) · **Y** — Potential (AKHLAK)")

    # Column headers
    _, *header_cols = st.columns([0.8, 3, 3, 3])
    for hcol, label in zip(
        header_cols,
        ["🔴 Low Performance", "🟡 Medium Performance", "🟢 High Performance"],
    ):
        hcol.markdown(
            f"<div style='text-align:center;font-size:0.75rem;opacity:0.65;padding-bottom:4px'>{label}</div>",
            unsafe_allow_html=True,
        )

    row_labels = ["🟢 High Potential", "🟡 Mid Potential", "🔴 Low Potential"]

    for row_i in range(3):
        label_col, *grid_cols = st.columns([0.8, 3, 3, 3])
        label_col.markdown(
            f"<div style='font-size:0.72rem;opacity:0.65;margin-top:1.2rem'>{row_labels[row_i]}</div>",
            unsafe_allow_html=True,
        )
        for col_i, gcol in enumerate(grid_cols):
            color = CELL_COLORS[row_i][col_i]
            label = GRID_LABELS[row_i][col_i]
            candidates_in_cell = cells[row_i][col_i]
            with gcol:
                st.markdown(
                    f"<div style='background:{color};color:white;padding:4px 10px;"
                    f"border-radius:6px 6px 0 0;font-size:0.72rem;font-weight:600;"
                    f"text-align:center;letter-spacing:0.02em'>{label}</div>",
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    if candidates_in_cell:
                        for name, eid, kpi, potential, profile in candidates_in_cell:
                            if st.button(
                                f"👤 {name}",
                                key=f"nb_{row_i}_{col_i}_{eid}",
                                use_container_width=True,
                            ):
                                st.session_state["selected_candidate"] = profile
                                st.session_state["selected_match"] = {}
                                st.rerun()
                            st.caption(f"KPI {kpi}% · AKHLAK {potential}%")
                    else:
                        st.markdown(
                            "<div style='opacity:0.3;text-align:center;padding:10px 0'>—</div>",
                            unsafe_allow_html=True,
                        )
