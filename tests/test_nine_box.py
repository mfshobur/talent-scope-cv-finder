"""
Tests for the 9-box talent matrix component.

Unit tests: bucket logic, candidate placement, mock data loading.
Integration tests: end-to-end placement with real ChromaDB profiles.
"""

import json
import pytest

from ui.nine_box import kpi_bucket, potential_bucket, place_candidate, load_kpi, load_akhlak, GRID_LABELS, CELL_COLORS


# ── Unit: bucket logic ────────────────────────────────────────────────────────

class TestBucketLogic:
    def test_kpi_low(self):
        assert kpi_bucket(0) == 0
        assert kpi_bucket(59) == 0
        assert kpi_bucket(59.9) == 0

    def test_kpi_medium(self):
        assert kpi_bucket(60) == 1
        assert kpi_bucket(79) == 1
        assert kpi_bucket(79.9) == 1

    def test_kpi_high(self):
        assert kpi_bucket(80) == 2
        assert kpi_bucket(100) == 2
        assert kpi_bucket(95) == 2

    def test_potential_mirrors_kpi(self):
        for score in [0, 59, 60, 79, 80, 100]:
            assert potential_bucket(score) == kpi_bucket(score)


# ── Unit: candidate placement ─────────────────────────────────────────────────

class TestPlacement:
    def test_star_top_right(self):
        # High KPI (≥80) + High Potential (≥80) → row=0, col=2
        row, col = place_candidate(kpi_score=90, potential_score=85)
        assert row == 0 and col == 2
        assert GRID_LABELS[row][col] == "Star ⭐"

    def test_underperformer_bottom_left(self):
        # Low KPI + Low Potential → row=2, col=0
        row, col = place_candidate(kpi_score=40, potential_score=45)
        assert row == 2 and col == 0
        assert GRID_LABELS[row][col] == "Underperformer"

    def test_high_potential_enigma(self):
        # Low KPI + High Potential → row=0, col=0
        row, col = place_candidate(kpi_score=45, potential_score=82)
        assert row == 0 and col == 0
        assert GRID_LABELS[row][col] == "Enigma"

    def test_core_employee_center(self):
        # Medium KPI + Medium Potential → row=1, col=1
        row, col = place_candidate(kpi_score=70, potential_score=70)
        assert row == 1 and col == 1
        assert GRID_LABELS[row][col] == "Core Employee"

    def test_trusted_pro_bottom_right(self):
        # High KPI + Low Potential → row=2, col=2
        row, col = place_candidate(kpi_score=88, potential_score=50)
        assert row == 2 and col == 2
        assert GRID_LABELS[row][col] == "Trusted Pro"

    def test_boundary_60_is_medium(self):
        row, col = place_candidate(kpi_score=60, potential_score=60)
        assert row == 1 and col == 1  # both medium

    def test_boundary_80_is_high(self):
        row, col = place_candidate(kpi_score=80, potential_score=80)
        assert row == 0 and col == 2  # both high


# ── Unit: grid structure ──────────────────────────────────────────────────────

class TestGridStructure:
    def test_grid_labels_3x3(self):
        assert len(GRID_LABELS) == 3
        assert all(len(row) == 3 for row in GRID_LABELS)

    def test_cell_colors_3x3(self):
        assert len(CELL_COLORS) == 3
        assert all(len(row) == 3 for row in CELL_COLORS)

    def test_cell_colors_are_hex(self):
        for row in CELL_COLORS:
            for color in row:
                assert color.startswith("#")
                assert len(color) == 7


# ── Unit: mock data loading ───────────────────────────────────────────────────

class TestMockData:
    def test_kpi_loads(self):
        data = load_kpi.__wrapped__()  # bypass functools.cache
        assert isinstance(data, dict)
        assert "emp_002" in data
        assert "score" in data["emp_002"]
        assert "label" in data["emp_002"]

    def test_kpi_covers_all_employees(self):
        data = load_kpi.__wrapped__()
        for i in range(2, 22):
            assert f"emp_{i:03d}" in data, f"emp_{i:03d} missing from mock_kpi.json"

    def test_kpi_scores_in_range(self):
        data = load_kpi.__wrapped__()
        for eid, kpi in data.items():
            assert 0 <= kpi["score"] <= 100, f"{eid} score out of range"

    def test_kpi_labels_valid(self):
        data = load_kpi.__wrapped__()
        valid_labels = {"Below Target", "Meets Target", "Exceeds Target"}
        for eid, kpi in data.items():
            assert kpi["label"] in valid_labels, f"{eid} has invalid label: {kpi['label']}"

    def test_kpi_label_matches_score(self):
        data = load_kpi.__wrapped__()
        for eid, kpi in data.items():
            score = kpi["score"]
            label = kpi["label"]
            if score >= 80:
                assert label == "Exceeds Target", f"{eid}: score {score} should be Exceeds Target"
            elif score >= 60:
                assert label == "Meets Target", f"{eid}: score {score} should be Meets Target"
            else:
                assert label == "Below Target", f"{eid}: score {score} should be Below Target"

    def test_akhlak_loads(self):
        data = load_akhlak.__wrapped__()
        assert isinstance(data, dict)
        assert "emp_002" in data

    def test_akhlak_covers_all_employees(self):
        data = load_akhlak.__wrapped__()
        for i in range(2, 22):
            assert f"emp_{i:03d}" in data, f"emp_{i:03d} missing from mock_akhlak.json"

    def test_akhlak_has_all_dimensions(self):
        data = load_akhlak.__wrapped__()
        dims = {"amanat", "kompeten", "harmonis", "loyal", "adaptif", "kolaboratif", "composite"}
        for eid, akhlak in data.items():
            assert dims == set(akhlak.keys()), f"{eid} missing dimensions"

    def test_akhlak_composite_is_average(self):
        data = load_akhlak.__wrapped__()
        dims = ["amanat", "kompeten", "harmonis", "loyal", "adaptif", "kolaboratif"]
        for eid, akhlak in data.items():
            expected = round(sum(akhlak[d] for d in dims) / 6)
            assert abs(akhlak["composite"] - expected) <= 1, \
                f"{eid} composite {akhlak['composite']} doesn't match average {expected}"

    def test_akhlak_scores_in_range(self):
        data = load_akhlak.__wrapped__()
        all_dims = ["amanat", "kompeten", "harmonis", "loyal", "adaptif", "kolaboratif", "composite"]
        for eid, akhlak in data.items():
            for dim in all_dims:
                assert 0 <= akhlak[dim] <= 100, f"{eid}.{dim} out of range"

    def test_nine_box_distribution_has_all_buckets(self):
        """Verify mock data produces candidates in all 3 KPI and potential buckets."""
        kpi_data = load_kpi.__wrapped__()
        akhlak_data = load_akhlak.__wrapped__()
        kpi_buckets = {kpi_bucket(v["score"]) for v in kpi_data.values()}
        pot_buckets = {potential_bucket(v["composite"]) for v in akhlak_data.values()}
        assert kpi_buckets == {0, 1, 2}, "Mock KPI data must cover all 3 performance buckets"
        assert pot_buckets == {0, 1, 2}, "Mock AKHLAK data must cover all 3 potential buckets"


# ── Integration: placement with real ChromaDB profiles ────────────────────────

@pytest.mark.integration
class TestNineBoxIntegration:
    def test_all_mock_employees_can_be_placed(self, real_store):
        """Every employee in mock data should have a valid profile in ChromaDB."""
        kpi_data = load_kpi.__wrapped__()
        missing = []
        for eid in kpi_data:
            if real_store.get_profile(eid) is None:
                missing.append(eid)
        assert not missing, f"Profiles not found in ChromaDB: {missing}"

    def test_placement_produces_valid_grid_positions(self, real_store):
        """Placement output is always a valid (row, col) within 0–2 bounds."""
        kpi_data = load_kpi.__wrapped__()
        akhlak_data = load_akhlak.__wrapped__()
        for eid in kpi_data:
            kpi_score = kpi_data[eid]["score"]
            potential_score = akhlak_data.get(eid, {}).get("composite", 50)
            row, col = place_candidate(kpi_score, potential_score)
            assert 0 <= row <= 2, f"{eid}: row {row} out of bounds"
            assert 0 <= col <= 2, f"{eid}: col {col} out of bounds"

    def test_star_employees_have_correct_position(self, real_store):
        """Employees with high KPI and high AKHLAK should land in the Star cell."""
        kpi_data = load_kpi.__wrapped__()
        akhlak_data = load_akhlak.__wrapped__()
        stars = [
            eid for eid in kpi_data
            if kpi_data[eid]["score"] >= 80
            and akhlak_data.get(eid, {}).get("composite", 0) >= 80
        ]
        assert stars, "No Star candidates found — check mock data"
        for eid in stars:
            row, col = place_candidate(kpi_data[eid]["score"], akhlak_data[eid]["composite"])
            assert (row, col) == (0, 2), f"{eid} should be Star (0,2), got ({row},{col})"
