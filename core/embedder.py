import litellm
from config.settings import Settings


class Embedder:
    def __init__(self, settings: Settings):
        self._model = settings.embedding_model
        self._api_key = settings.effective_embedding_api_key()
        self._base_url = settings.llm_base_url or None

    def embed(self, text: str) -> list[float]:
        resp = litellm.embedding(
            model=self._model,
            input=[text],
            api_key=self._api_key or None,
            base_url=self._base_url,
        )
        return resp.data[0]["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = litellm.embedding(
            model=self._model,
            input=texts,
            api_key=self._api_key or None,
            base_url=self._base_url,
        )
        return [item["embedding"] for item in resp.data]
