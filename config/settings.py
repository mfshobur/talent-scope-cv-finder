import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = "openai/gpt-5-nano"
    # Accepts LLM_API_KEY or falls back to provider-standard env vars (OPENAI_API_KEY, etc.)
    llm_api_key: str = ""
    llm_base_url: str = ""

    vision_model: str = "openai/gpt-5-nano"
    vision_api_key: str = ""

    embedding_model: str = "openai/text-embedding-3-small"
    embedding_api_key: str = ""

    chroma_persist_dir: str = "./chroma_db"
    session_db_path: str = "./sessions.db"

    max_candidates_returned: int = 5

    admin_password: str = ""

    def effective_llm_api_key(self) -> str:
        """Returns the best available API key for the configured provider."""
        if self.llm_api_key:
            return self.llm_api_key
        # Fall back to provider-standard env vars so existing .env setups work
        for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
            val = os.getenv(env_var, "")
            if val:
                return val
        return ""

    def effective_vision_api_key(self) -> str:
        return self.vision_api_key or self.effective_llm_api_key()

    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.effective_llm_api_key()
