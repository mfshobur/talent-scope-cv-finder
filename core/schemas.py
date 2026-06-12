from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    company: str
    role: str
    start_year: int
    end_year: Optional[int] = None  # None = present
    description: str = ""


class Education(BaseModel):
    institution: str
    degree: str
    field: str
    graduation_year: Optional[int] = None
    gpa: Optional[float] = None


class LanguageProficiency(BaseModel):
    language: str
    level: Literal["basic", "intermediate", "professional", "native"]


class ParsedCV(BaseModel):
    employee_id: str
    full_name: str
    current_role: str = ""
    department: str = ""
    location: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[LanguageProficiency] = Field(default_factory=list)
    raw_text: str = ""
    parse_confidence: float = 1.0
    parse_warnings: list[str] = Field(default_factory=list)


AssessmentCategory = Literal[
    "psychotest",
    "technical",
    "behavioral",
    "english_proficiency",
]


class ParsedAssessment(BaseModel):
    employee_id: str
    category: AssessmentCategory
    raw_score: Optional[float] = None
    score_label: str = ""
    key_findings: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    development_areas: list[str] = Field(default_factory=list)
    raw_text: str = ""
    parse_confidence: float = 1.0


class EmployeeProfile(BaseModel):
    employee_id: str
    cv: ParsedCV
    assessments: list[ParsedAssessment] = Field(default_factory=list)

    def assessment_summary(self) -> str:
        lines = []
        for a in self.assessments:
            score_str = a.score_label
            findings = "; ".join(a.key_findings[:3])
            lines.append(f"{a.category.upper()}: {score_str}. {findings}")
        return "\n".join(lines)

    def embedding_text(self) -> str:
        cv = self.cv
        exp_lines = [
            f"{e.role} at {e.company} ({e.start_year}–{e.end_year or 'present'}): {e.description}"
            for e in cv.work_experience
        ]
        edu_lines = [
            f"{e.degree} in {e.field} from {e.institution}"
            for e in cv.education
        ]
        return "\n".join([
            f"Name: {cv.full_name}",
            f"Role: {cv.current_role}",
            f"Department: {cv.department}",
            f"Summary: {cv.summary}",
            f"Skills: {', '.join(cv.skills)}",
            "Experience:\n" + "\n".join(exp_lines),
            "Education:\n" + "\n".join(edu_lines),
            f"Languages: {', '.join(l.language for l in cv.languages)}",
            f"Certifications: {', '.join(cv.certifications)}",
        ])

    def assessment_embedding_text(self) -> str:
        return self.assessment_summary()


class CandidateMatch(BaseModel):
    employee_id: str
    full_name: str
    current_role: str
    department: str
    relevance_score: float
    match_reasons: list[str] = Field(default_factory=list)
    fit_gaps: list[str] = Field(default_factory=list)
    profile: EmployeeProfile
