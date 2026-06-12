import json
from typing import Callable

import chromadb

from core.schemas import CandidateMatch, EmployeeProfile

COLLECTION_NAME = "employee_profiles"


class CandidateStore:
    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def upsert_employee(
        self,
        profile: EmployeeProfile,
        embedder: Callable[[str], list[float]],
    ) -> None:
        eid = profile.employee_id
        cv = profile.cv
        base_meta = {
            "employee_id": eid,
            "full_name": cv.full_name,
            "current_role": cv.current_role,
            "department": cv.department,
            "location": cv.location,
            "skills_json": json.dumps(cv.skills),
            "profile_json": profile.model_dump_json(),
        }

        cv_text = profile.embedding_text()
        asm_text = profile.assessment_embedding_text() or cv_text

        self._col.upsert(
            ids=[f"{eid}_cv", f"{eid}_asm"],
            documents=[cv_text, asm_text],
            embeddings=[embedder(cv_text), embedder(asm_text)],
            metadatas=[
                {**base_meta, "doc_type": "cv"},
                {**base_meta, "doc_type": "assessments"},
            ],
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 20,
        department_filter: str | None = None,
    ) -> list[dict]:
        where = {"department": department_filter} if department_filter else None
        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self._col.count() or 1),
            where=where,
            include=["metadatas", "distances", "documents"],
        )
        out = []
        seen: set[str] = set()
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            eid = meta["employee_id"]
            if eid in seen:
                continue
            seen.add(eid)
            out.append({
                "employee_id": eid,
                "full_name": meta.get("full_name", ""),
                "current_role": meta.get("current_role", ""),
                "department": meta.get("department", ""),
                "skills": json.loads(meta.get("skills_json", "[]")),
                "similarity": round(1 - dist, 4),
                "profile_json": meta.get("profile_json", "{}"),
            })
        return out

    def get_profile(self, employee_id: str) -> EmployeeProfile | None:
        result = self._col.get(
            ids=[f"{employee_id}_cv"],
            include=["metadatas"],
        )
        if not result["metadatas"]:
            return None
        raw = result["metadatas"][0].get("profile_json", "{}")
        return EmployeeProfile.model_validate_json(raw)

    def get_all_profiles(self) -> list[EmployeeProfile]:
        result = self._col.get(
            where={"doc_type": "cv"},
            include=["metadatas"],
        )
        profiles = []
        for meta in result["metadatas"]:
            raw = meta.get("profile_json", "{}")
            try:
                profiles.append(EmployeeProfile.model_validate_json(raw))
            except Exception:
                continue
        return profiles

    def count(self) -> int:
        return self._col.count()
