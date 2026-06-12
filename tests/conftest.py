import pytest
from core.schemas import (
    ParsedCV, ParsedAssessment, EmployeeProfile,
    WorkExperience, Education, LanguageProficiency,
)
from config.settings import Settings


@pytest.fixture
def sample_cv() -> ParsedCV:
    return ParsedCV(
        employee_id="emp_001",
        full_name="Muh. Shobur Fattah",
        current_role="AI/ML Engineer",
        department="IT",
        location="South Jakarta, Indonesia",
        email="mfshobur@gmail.com",
        summary="AI Engineer with hands-on experience building LLM systems.",
        skills=["Python", "LangChain", "PyTorch", "FastAPI", "ChromaDB"],
        work_experience=[
            WorkExperience(
                company="PT Holding Nusantara",
                role="AI/ML Engineer",
                start_year=2025,
                end_year=None,
                description="Built 3-agent LLM chatbot and RAG systems.",
            )
        ],
        education=[
            Education(
                institution="Universitas Hasanuddin",
                degree="Bachelor",
                field="Informatics Engineering",
                graduation_year=2025,
                gpa=3.91,
            )
        ],
        languages=[
            LanguageProficiency(language="Indonesian", level="native"),
            LanguageProficiency(language="English", level="professional"),
        ],
        parse_confidence=0.98,
    )


@pytest.fixture
def sample_assessment() -> ParsedAssessment:
    return ParsedAssessment(
        employee_id="emp_001",
        category="technical",
        raw_score=88.0,
        score_label="Above Average",
        key_findings=["Strong Python", "Good system design", "ML fundamentals solid"],
        strengths=["LLM pipelines", "RAG systems"],
        development_areas=["Cloud architecture"],
        parse_confidence=0.95,
    )


@pytest.fixture
def sample_profile(sample_cv, sample_assessment) -> EmployeeProfile:
    return EmployeeProfile(
        employee_id="emp_001",
        cv=sample_cv,
        assessments=[sample_assessment],
    )


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        llm_model="openai/gpt-5-nano",
        llm_api_key="test-key",
        vision_model="openai/gpt-5-nano",
        embedding_model="openai/text-embedding-3-small",
        chroma_persist_dir="/tmp/test_chroma",
        session_db_path="/tmp/test_sessions.db",
    )


# ── Integration fixtures (real API key + real ChromaDB) ───────────────────────

@pytest.fixture(scope="session")
def real_settings() -> Settings:
    s = Settings()
    if not s.effective_llm_api_key():
        pytest.skip("No LLM API key — set OPENAI_API_KEY or LLM_API_KEY in .env")
    return s


@pytest.fixture(scope="session")
def real_store(real_settings):
    from ingestion.chromadb_store import CandidateStore
    store = CandidateStore(real_settings.chroma_persist_dir)
    if store.count() == 0:
        pytest.skip("ChromaDB is empty — run scripts/run_ingestion.py first")
    return store


@pytest.fixture(scope="session")
def real_embedder(real_settings):
    from core.embedder import Embedder
    return Embedder(real_settings)
