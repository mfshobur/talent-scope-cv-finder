import pytest
from pydantic import ValidationError
from core.schemas import (
    ParsedCV, ParsedAssessment, EmployeeProfile, CandidateMatch,
    WorkExperience, Education, LanguageProficiency,
)


class TestWorkExperience:
    def test_present_role(self):
        exp = WorkExperience(company="Acme", role="Engineer", start_year=2022)
        assert exp.end_year is None

    def test_past_role(self):
        exp = WorkExperience(company="Acme", role="Engineer", start_year=2020, end_year=2022)
        assert exp.end_year == 2022


class TestLanguageProficiency:
    def test_valid_levels(self):
        for level in ("basic", "intermediate", "professional", "native"):
            lp = LanguageProficiency(language="English", level=level)
            assert lp.level == level

    def test_invalid_level(self):
        with pytest.raises(ValidationError):
            LanguageProficiency(language="English", level="fluent")


class TestParsedCV:
    def test_defaults_are_empty(self):
        cv = ParsedCV(employee_id="emp_002", full_name="Test User")
        assert cv.skills == []
        assert cv.work_experience == []
        assert cv.parse_confidence == 1.0
        assert cv.current_role == ""

    def test_full_cv(self, sample_cv):
        assert sample_cv.full_name == "Muh. Shobur Fattah"
        assert "Python" in sample_cv.skills
        assert sample_cv.education[0].gpa == 3.91

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ParsedCV(full_name="No ID")  # employee_id required


class TestParsedAssessment:
    def test_valid_categories(self):
        for cat in ("psychotest", "technical", "behavioral", "english_proficiency"):
            a = ParsedAssessment(employee_id="emp_001", category=cat)
            assert a.category == cat

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            ParsedAssessment(employee_id="emp_001", category="iq_test")

    def test_optional_score(self):
        a = ParsedAssessment(employee_id="emp_001", category="psychotest")
        assert a.raw_score is None


class TestEmployeeProfile:
    def test_embedding_text_contains_key_fields(self, sample_profile):
        text = sample_profile.embedding_text()
        assert "Muh. Shobur Fattah" in text
        assert "AI/ML Engineer" in text
        assert "Python" in text
        assert "LangChain" in text

    def test_assessment_summary(self, sample_profile):
        summary = sample_profile.assessment_summary()
        assert "TECHNICAL" in summary
        assert "Above Average" in summary

    def test_no_assessments(self, sample_cv):
        profile = EmployeeProfile(employee_id="emp_002", cv=sample_cv)
        assert profile.assessment_summary() == ""


class TestCandidateMatch:
    def test_construction(self, sample_profile):
        match = CandidateMatch(
            employee_id="emp_001",
            full_name="Muh. Shobur Fattah",
            current_role="AI/ML Engineer",
            department="IT",
            relevance_score=0.92,
            match_reasons=["Strong LLM experience"],
            fit_gaps=["Limited cloud experience"],
            profile=sample_profile,
        )
        assert match.relevance_score == 0.92
        assert len(match.match_reasons) == 1
