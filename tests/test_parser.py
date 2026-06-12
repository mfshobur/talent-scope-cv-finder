import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.parser import DocumentParser, _extract_json, _pdf_to_images
from core.schemas import ParsedCV, ParsedAssessment


VALID_CV_JSON = {
    "full_name": "Arini Kusumawati",
    "current_role": "Data Engineer",
    "department": "IT",
    "location": "Jakarta",
    "email": "arini@email.com",
    "phone": "+62 812 xxxx xxxx",
    "summary": "Experienced Data Engineer with 4+ years.",
    "skills": ["Apache Spark", "dbt", "Python", "PostgreSQL"],
    "work_experience": [
        {"company": "PT Data Solusi", "role": "Data Engineer", "start_year": 2019, "end_year": None, "description": "Built ETL pipelines."}
    ],
    "education": [
        {"institution": "Universitas Indonesia", "degree": "Bachelor", "field": "Informatics Engineering", "graduation_year": 2018, "gpa": 3.75}
    ],
    "certifications": ["Google Cloud Professional Data Engineer"],
    "languages": [
        {"language": "Indonesian", "level": "native"},
        {"language": "English", "level": "professional"},
    ],
}

VALID_ASM_JSON = {
    "raw_score": 82.0,
    "score_label": "Above Average",
    "key_findings": ["Strong Python skills", "Good pipeline design"],
    "strengths": ["Data modeling", "ETL optimization"],
    "development_areas": ["Cloud architecture"],
}


class TestExtractJson:
    def test_plain_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_wrapped(self):
        result = _extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_prefix_text(self):
        result = _extract_json('Here is the result:\n{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_raises(self):
        with pytest.raises((ValueError, Exception)):
            _extract_json("no json here")


class TestDocumentParser:
    @pytest.fixture
    def parser(self, mock_settings):
        return DocumentParser(mock_settings)

    @pytest.fixture
    def mock_pdf_images(self):
        return [b"fake_image_bytes"]

    def test_parse_cv_success(self, parser, mock_pdf_images):
        with patch("core.parser._pdf_to_images", return_value=mock_pdf_images), \
             patch.object(parser, "_call_vision", return_value=json.dumps(VALID_CV_JSON)):
            cv = parser.parse_cv(Path("fake/cv.pdf"), "emp_002")

        assert isinstance(cv, ParsedCV)
        assert cv.full_name == "Arini Kusumawati"
        assert cv.employee_id == "emp_002"
        assert "Apache Spark" in cv.skills
        assert cv.education[0].gpa == 3.75
        assert cv.parse_confidence > 0.8

    def test_parse_cv_retries_on_bad_json(self, parser, mock_pdf_images):
        call_count = 0

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "not valid json at all"
            return json.dumps(VALID_CV_JSON)

        with patch("core.parser._pdf_to_images", return_value=mock_pdf_images), \
             patch.object(parser, "_call_vision", side_effect=side_effect):
            cv = parser.parse_cv(Path("fake/cv.pdf"), "emp_002")

        assert cv.full_name == "Arini Kusumawati"

    def test_parse_cv_missing_name_lowers_confidence(self, parser, mock_pdf_images):
        data = {**VALID_CV_JSON, "full_name": ""}
        with patch("core.parser._pdf_to_images", return_value=mock_pdf_images), \
             patch.object(parser, "_call_vision", return_value=json.dumps(data)):
            cv = parser.parse_cv(Path("fake/cv.pdf"), "emp_002")

        assert cv.parse_confidence < 0.8
        assert "full_name not found" in cv.parse_warnings

    def test_parse_cv_invalid_end_year_skipped(self, parser, mock_pdf_images):
        data = {**VALID_CV_JSON, "work_experience": [
            {"company": "Acme", "role": "Eng", "start_year": "bad", "end_year": None, "description": ""}
        ]}
        with patch("core.parser._pdf_to_images", return_value=mock_pdf_images), \
             patch.object(parser, "_call_vision", return_value=json.dumps(data)):
            cv = parser.parse_cv(Path("fake/cv.pdf"), "emp_002")

        assert cv.work_experience == []

    def test_parse_assessment_success(self, parser, mock_pdf_images):
        with patch("core.parser._pdf_to_images", return_value=mock_pdf_images), \
             patch.object(parser, "_call_vision", return_value=json.dumps(VALID_ASM_JSON)):
            asm = parser.parse_assessment(Path("fake/technical.pdf"), "emp_002", "technical")

        assert isinstance(asm, ParsedAssessment)
        assert asm.employee_id == "emp_002"
        assert asm.category == "technical"
        assert asm.raw_score == 82.0
        assert asm.score_label == "Above Average"
        assert len(asm.strengths) == 2

    def test_parse_assessment_no_score(self, parser, mock_pdf_images):
        data = {**VALID_ASM_JSON, "raw_score": None, "score_label": ""}
        with patch("core.parser._pdf_to_images", return_value=mock_pdf_images), \
             patch.object(parser, "_call_vision", return_value=json.dumps(data)):
            asm = parser.parse_assessment(Path("fake/psychotest.pdf"), "emp_002", "psychotest")

        assert asm.raw_score is None
        assert asm.parse_confidence == 0.6

    @pytest.mark.integration
    def test_parse_real_cv(self, mock_settings):
        """Integration test — requires real API key and emp_001 CV on disk."""
        from config.settings import Settings
        s = Settings()
        if not s.effective_llm_api_key():
            pytest.skip("No API key available")

        cv_path = Path("data/employees/emp_001_shobur/cv.pdf")
        if not cv_path.exists():
            pytest.skip("emp_001 CV not found")

        parser = DocumentParser(s)
        cv = parser.parse_cv(cv_path, "emp_001")
        assert "Shobur" in cv.full_name
        assert len(cv.skills) > 3
        assert cv.parse_confidence > 0.7
