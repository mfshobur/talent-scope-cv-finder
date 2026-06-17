"""
Agent tool implementations and LiteLLM tool schema definitions.
Each tool takes parsed arguments and a CandidateStore, returns a JSON-serializable dict.
"""

import json
from ingestion.chromadb_store import CandidateStore
from core.schemas import EmployeeProfile


# ── Tool schemas (OpenAI function-calling format) ─────────────────────────────

def build_tool_schemas() -> list[dict]:
    from agent.prompts import (
        TOOL_SEARCH_DESCRIPTION, TOOL_GET_DETAIL_DESCRIPTION,
        TOOL_COMPARE_DESCRIPTION, TOOL_FILTER_DESCRIPTION,
    )
    return [
        {
            "type": "function",
            "function": {
                "name": "search_candidates",
                "description": TOOL_SEARCH_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query"},
                        "top_k": {"type": "integer", "description": "Number of results to return (default 5)", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_candidate_detail",
                "description": TOOL_GET_DETAIL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "employee_id": {"type": "string", "description": "Employee ID, e.g. 'emp_001'"},
                    },
                    "required": ["employee_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_candidates",
                "description": TOOL_COMPARE_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "employee_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of employee IDs to compare",
                        },
                    },
                    "required": ["employee_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filter_candidates",
                "description": TOOL_FILTER_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "department": {"type": "string", "description": "Filter by department name"},
                        "required_skills": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Skills that candidates must have",
                        },
                    },
                    "required": [],
                },
            },
        },
    ]


# ── Tool implementations ──────────────────────────────────────────────────────

def _calc_yoe(profile: EmployeeProfile) -> int:
    """Total career span in years (current year minus earliest start year)."""
    import datetime
    exps = profile.cv.work_experience
    if not exps:
        return 0
    current_year = datetime.date.today().year
    return current_year - min(e.start_year for e in exps)


def search_candidates(query: str, top_k: int, store: CandidateStore, embedder) -> dict:
    query_embedding = embedder.embed(query)
    results = store.search(query_embedding, n_results=top_k * 2)  # over-fetch for dedup

    candidates = []
    seen = set()
    for r in results:
        eid = r["employee_id"]
        if eid in seen:
            continue
        seen.add(eid)
        profile = store.get_profile(eid)
        yoe = _calc_yoe(profile) if profile else 0
        candidates.append({
            "employee_id": eid,
            "full_name": r["full_name"],
            "current_role": r["current_role"],
            "department": r["department"],
            "skills": r["skills"][:8],
            "years_of_experience": yoe,
            "similarity_score": r["similarity"],
        })
        if len(candidates) >= top_k:
            break
    return {"query": query, "candidates": candidates, "total_found": len(results)}


def get_candidate_detail(employee_id: str, store: CandidateStore) -> dict:
    profile = store.get_profile(employee_id)
    if profile is None:
        return {"error": f"Employee {employee_id} not found"}

    cv = profile.cv
    assessments = []
    for asm in profile.assessments:
        assessments.append({
            "category": asm.category,
            "score_label": asm.score_label,
            "key_findings": asm.key_findings,
            "strengths": asm.strengths,
            "development_areas": asm.development_areas,
        })

    experience = [
        {
            "company": e.company,
            "role": e.role,
            "period": f"{e.start_year}–{e.end_year or 'present'}",
            "description": e.description,
        }
        for e in cv.work_experience
    ]

    return {
        "employee_id": employee_id,
        "full_name": cv.full_name,
        "current_role": cv.current_role,
        "department": cv.department,
        "summary": cv.summary,
        "skills": cv.skills,
        "experience": experience,
        "education": [
            {"institution": e.institution, "degree": e.degree, "field": e.field, "gpa": e.gpa}
            for e in cv.education
        ],
        "languages": [{"language": l.language, "level": l.level} for l in cv.languages],
        "certifications": cv.certifications,
        "assessments": assessments,
    }


def compare_candidates(employee_ids: list[str], store: CandidateStore) -> dict:
    comparisons = []
    for eid in employee_ids:
        detail = get_candidate_detail(eid, store)
        if "error" not in detail:
            comparisons.append(detail)
    return {"compared": comparisons}


def filter_candidates(
    store: CandidateStore,
    embedder,
    department: str | None = None,
    required_skills: list[str] | None = None,
) -> dict:
    all_profiles = store.get_all_profiles()
    results = []
    for profile in all_profiles:
        cv = profile.cv
        if department and cv.department.lower() != department.lower():
            continue
        if required_skills:
            cv_skills_lower = {s.lower() for s in cv.skills}
            if not all(s.lower() in cv_skills_lower for s in required_skills):
                continue
        results.append({
            "employee_id": profile.employee_id,
            "full_name": cv.full_name,
            "current_role": cv.current_role,
            "department": cv.department,
            "skills": cv.skills[:8],
        })
    return {"filter_criteria": {"department": department, "required_skills": required_skills}, "candidates": results}


def dispatch_tool(name: str, arguments_json: str, store: CandidateStore, embedder) -> dict:
    args = json.loads(arguments_json) if isinstance(arguments_json, str) else arguments_json

    if name == "search_candidates":
        return search_candidates(
            query=args["query"],
            top_k=args.get("top_k", 5),
            store=store,
            embedder=embedder,
        )
    elif name == "get_candidate_detail":
        return get_candidate_detail(args["employee_id"], store)
    elif name == "compare_candidates":
        return compare_candidates(args["employee_ids"], store)
    elif name == "filter_candidates":
        return filter_candidates(
            store=store,
            embedder=embedder,
            department=args.get("department"),
            required_skills=args.get("required_skills"),
        )
    else:
        return {"error": f"Unknown tool: {name}"}
