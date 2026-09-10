"""
Custom embeddings wrapper using sentence-transformers to replace langchain HuggingFaceEmbeddings.
"""

from typing import List

from sentence_transformers import SentenceTransformer


class HuggingFaceEmbeddings:
    """Wrapper around SentenceTransformer to match langchain interface.

    Embeddings are L2-normalised so FAISS IndexFlatIP scores equal cosine
    similarity (intuitively roughly in [-1, 1], often [0, 1] for these models).
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (L2-normalised)."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (L2-normalised)."""
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding[0].tolist()
