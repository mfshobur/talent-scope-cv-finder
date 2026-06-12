"""
Format-agnostic document parser.
Converts any CV or assessment PDF/image → normalized Pydantic model via LLM vision.
"""

import json
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import Settings
from core.schemas import (
    AssessmentCategory, Education, EmployeeProfile, LanguageProficiency,
    ParsedAssessment, ParsedCV, WorkExperience,
)

_CV_SCHEMA = {
    "full_name": "string",
    "current_role": "string",
    "department": "string (infer from role if not explicit)",
    "location": "string",
    "email": "string",
    "phone": "string",
    "summary": "string",
    "skills": ["list of skill strings"],
    "work_experience": [{"company": "string", "role": "string", "start_year": "int", "end_year": "int or null", "description": "string"}],
    "education": [{"institution": "string", "degree": "string", "field": "string", "graduation_year": "int or null", "gpa": "float or null"}],
    "certifications": ["list of strings"],
    "languages": [{"language": "string", "level": "basic|intermediate|professional|native"}],
}

_ASM_SCHEMA = {
    "raw_score": "float or null (numeric score if present)",
    "score_label": "string (e.g. 'Above Average', 'B2', 'Proficient')",
    "key_findings": ["list of 2-4 key finding strings"],
    "strengths": ["list of 2-3 strength strings"],
    "development_areas": ["list of 1-2 development area strings"],
}

_CV_PROMPT = f"""You are an HR data extraction assistant. Extract ALL information from this CV/resume.
The CV may be in Indonesian, English, or a mix of both. Extract everything accurately.

Return ONLY a valid JSON object matching this schema (no markdown, no explanation):
{json.dumps(_CV_SCHEMA, indent=2)}

Rules:
- If a field is not found, use "" for strings, [] for lists, null for optional numbers.
- For department: infer from role if not explicitly stated (e.g. "Data Engineer" → "IT").
- For language level: map "native/ibu" → "native", "professional/advanced/C1/C2" → "professional", "intermediate/B1/B2" → "intermediate", "basic/elementary/A1/A2" → "basic".
- start_year and end_year must be integers (e.g. 2022), not strings.
- If end_year is "present/sekarang/current", use null.
"""

_ASM_PROMPT_TEMPLATE = """You are an HR data extraction assistant. Extract assessment results from this {category} assessment document.
The document may be in Indonesian or English.

Return ONLY a valid JSON object matching this schema (no markdown, no explanation):
{schema}

Rules:
- raw_score: extract numeric score if present (e.g. 85.0), otherwise null.
- score_label: overall result label (e.g. "Above Average", "B2", "Proficient", "High").
- key_findings: 2-4 most important findings/observations.
- strengths: 2-3 clear strengths identified.
- development_areas: 1-2 areas for improvement.
- Keep all text concise (under 100 chars per item).
"""


def _pdf_to_images(path: Path, max_pages: int = 4) -> list[bytes]:
    """Render PDF pages to PNG bytes at 150 DPI."""
    doc = fitz.open(str(path))
    images = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        mat = fitz.Matrix(1.5, 1.5)  # ~108 DPI
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def _build_vision_messages(system_prompt: str, images: list[bytes]) -> list[dict]:
    content: list[dict] = []
    for img_bytes in images:
        import base64
        b64 = base64.b64encode(img_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })
    content.append({"type": "text", "text": "Extract the information as instructed."})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start:end])


class DocumentParser:
    def __init__(self, settings: Settings):
        self._vision_model = settings.vision_model
        self._api_key = settings.effective_vision_api_key()
        self._base_url = settings.llm_base_url or None

    def _call_vision(self, messages: list[dict]) -> str:
        resp = litellm.completion(
            model=self._vision_model,
            messages=messages,
            api_key=self._api_key or None,
            base_url=self._base_url,
            max_tokens=1200,
            temperature=0.0,
        )
        return resp.choices[0].message.content

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60))
    def parse_cv(self, file_path: Path, employee_id: str) -> ParsedCV:
        images = _pdf_to_images(file_path)
        messages = _build_vision_messages(_CV_PROMPT, images)

        raw_text = self._call_vision(messages)
        try:
            data = _extract_json(raw_text)
        except (ValueError, json.JSONDecodeError) as e:
            # Retry with error feedback
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": f"Your response had a JSON parse error: {e}. Return ONLY valid JSON, no other text."})
            raw_text = self._call_vision(messages)
            data = _extract_json(raw_text)

        return self._build_parsed_cv(employee_id, data, raw_text)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60))
    def parse_assessment(
        self,
        file_path: Path,
        employee_id: str,
        category: AssessmentCategory,
    ) -> ParsedAssessment:
        images = _pdf_to_images(file_path)
        prompt = _ASM_PROMPT_TEMPLATE.format(
            category=category.replace("_", " "),
            schema=json.dumps(_ASM_SCHEMA, indent=2),
        )
        messages = _build_vision_messages(prompt, images)

        raw_text = self._call_vision(messages)
        try:
            data = _extract_json(raw_text)
        except (ValueError, json.JSONDecodeError) as e:
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": f"JSON parse error: {e}. Return ONLY valid JSON."})
            raw_text = self._call_vision(messages)
            data = _extract_json(raw_text)

        return ParsedAssessment(
            employee_id=employee_id,
            category=category,
            raw_score=data.get("raw_score"),
            score_label=data.get("score_label", ""),
            key_findings=data.get("key_findings", []),
            strengths=data.get("strengths", []),
            development_areas=data.get("development_areas", []),
            raw_text=raw_text,
            parse_confidence=0.9 if data.get("score_label") else 0.6,
        )

    @staticmethod
    def _build_parsed_cv(employee_id: str, data: dict[str, Any], raw_text: str) -> ParsedCV:
        experiences = []
        for e in data.get("work_experience", []):
            try:
                experiences.append(WorkExperience(
                    company=e.get("company", ""),
                    role=e.get("role", ""),
                    start_year=int(e.get("start_year") or 2020),
                    end_year=int(e["end_year"]) if e.get("end_year") else None,
                    description=e.get("description", ""),
                ))
            except (TypeError, ValueError):
                continue

        educations = []
        for ed in data.get("education", []):
            try:
                educations.append(Education(
                    institution=ed.get("institution", ""),
                    degree=ed.get("degree", ""),
                    field=ed.get("field", ""),
                    graduation_year=int(ed["graduation_year"]) if ed.get("graduation_year") else None,
                    gpa=float(ed["gpa"]) if ed.get("gpa") else None,
                ))
            except (TypeError, ValueError):
                continue

        languages = []
        valid_levels = {"basic", "intermediate", "professional", "native"}
        for lang in data.get("languages", []):
            level = str(lang.get("level", "intermediate")).lower()
            if level not in valid_levels:
                level = "intermediate"
            try:
                languages.append(LanguageProficiency(
                    language=lang.get("language", ""),
                    level=level,
                ))
            except Exception:
                continue

        confidence = 1.0
        warnings = []
        if not data.get("full_name"):
            confidence -= 0.3
            warnings.append("full_name not found")
        if not data.get("skills"):
            confidence -= 0.1
            warnings.append("skills not found")

        return ParsedCV(
            employee_id=employee_id,
            full_name=data.get("full_name", ""),
            current_role=data.get("current_role", ""),
            department=data.get("department", ""),
            location=data.get("location", ""),
            email=data.get("email", ""),
            phone=data.get("phone", ""),
            summary=data.get("summary", ""),
            skills=data.get("skills", []),
            work_experience=experiences,
            education=educations,
            certifications=data.get("certifications", []),
            languages=languages,
            raw_text=raw_text,
            parse_confidence=max(0.0, confidence),
            parse_warnings=warnings,
        )
