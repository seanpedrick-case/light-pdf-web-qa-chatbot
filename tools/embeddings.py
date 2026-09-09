"""
Custom embeddings wrapper using sentence-transformers to replace langchain HuggingFaceEmbeddings.
"""

from typing import List

from sentence_transformers import SentenceTransformer


class HuggingFaceEmbeddings:
    """Wrapper around SentenceTransformer to match langchain interface."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(
            [text], convert_to_numpy=True, show_progress_bar=False
        )
        return embedding[0].tolist()
