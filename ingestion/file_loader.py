from dataclasses import dataclass, field
from pathlib import Path

from core.schemas import AssessmentCategory

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

ASSESSMENT_FILENAME_MAP: dict[str, AssessmentCategory] = {
    "psychotest": "psychotest",
    "technical": "technical",
    "behavioral": "behavioral",
    "english": "english_proficiency",
}


@dataclass
class EmployeeFiles:
    employee_id: str
    cv_path: Path
    assessment_paths: dict[AssessmentCategory, Path] = field(default_factory=dict)


def discover_employee_files(employee_dir: Path) -> EmployeeFiles:
    employee_id = employee_dir.name

    cv_path = None
    for name in ("cv.pdf", "cv.png", "cv.jpg", "resume.pdf", "resume.png"):
        candidate = employee_dir / name
        if candidate.exists():
            cv_path = candidate
            break

    if cv_path is None:
        # Pick first supported file that isn't in assessments/
        for f in sorted(employee_dir.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.parent == employee_dir:
                cv_path = f
                break

    if cv_path is None:
        raise FileNotFoundError(f"No CV file found in {employee_dir}")

    assessment_paths: dict[AssessmentCategory, Path] = {}
    asm_dir = employee_dir / "assessments"
    if asm_dir.exists():
        for f in sorted(asm_dir.iterdir()):
            if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            for key, category in ASSESSMENT_FILENAME_MAP.items():
                if key in f.stem.lower():
                    assessment_paths[category] = f
                    break

    return EmployeeFiles(
        employee_id=employee_id,
        cv_path=cv_path,
        assessment_paths=assessment_paths,
    )
