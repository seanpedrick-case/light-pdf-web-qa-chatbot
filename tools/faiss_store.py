"""
Custom FAISS vectorstore to replace langchain FAISS.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import faiss
import numpy as np

from tools.document import Document


class InMemoryDocstore:
    """Simple in-memory document store."""

    def __init__(self):
        self._dict: Dict[str, Document] = {}

    def add(self, mapping: Dict[str, Document]):
        """Add documents to the store."""
        if not isinstance(self._dict, dict):
            # Ensure _dict is a dictionary
            if hasattr(self._dict, "_dict"):
                self._dict = self._dict._dict
            else:
                self._dict = {}
        self._dict.update(mapping)

    def get(self, key: str) -> Optional[Document]:
        """Get a document by key."""
        if not isinstance(self._dict, dict):
            # Ensure _dict is a dictionary
            if hasattr(self._dict, "_dict"):
                self._dict = self._dict._dict
            else:
                self._dict = {}
        return self._dict.get(key)


class FAISS:
    """FAISS vectorstore wrapper."""

    def __init__(
        self,
        embedding_function,
        index: Optional[faiss.Index] = None,
        docstore: Optional[InMemoryDocstore] = None,
        index_to_docstore_id: Optional[Dict[int, str]] = None,
    ):
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore if docstore else InMemoryDocstore()
        self.index_to_docstore_id = index_to_docstore_id if index_to_docstore_id else {}

    @classmethod
    def from_documents(cls, documents: List[Document], embedding) -> "FAISS":
        """Create a FAISS index from documents."""
        if not documents:
            raise ValueError("No documents provided")

        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = embedding.embed_documents(texts)
        embeddings_np = np.array(embeddings).astype("float32")

        # Create FAISS index
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_np)

        # Create docstore
        docstore = InMemoryDocstore()
        index_to_docstore_id = {}

        for i, doc in enumerate(documents):
            doc_id = str(uuid4())
            docstore.add({doc_id: doc})
            index_to_docstore_id[i] = doc_id

        return cls(
            embedding_function=embedding.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores."""
        if self.index is None:
            return []

        # Get query embedding
        query_embedding = self.embedding_function(query)
        query_vector = np.array([query_embedding]).astype("float32")

        # Search
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue
            doc_id = self.index_to_docstore_id.get(idx)
            if doc_id:
                doc = self.docstore.get(doc_id)
                if doc:
                    results.append((doc, float(score)))

        return results

    def save_local(self, folder_path: str):
        """Save the FAISS index and docstore to disk."""
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(folder / "index.faiss"))

        # Save docstore and mapping
        save_dict = {
            "docstore": self.docstore._dict,
            "index_to_docstore_id": self.index_to_docstore_id,
        }
        with open(folder / "index.pkl", "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load_local(
        cls, folder_path: str, embeddings, allow_dangerous_deserialization: bool = False
    ) -> "FAISS":
        """Load a FAISS index from disk."""
        if not allow_dangerous_deserialization:
            raise ValueError(
                "allow_dangerous_deserialization must be True to load pickled files"
            )

        folder = Path(folder_path)

        # Load FAISS index
        index = faiss.read_index(str(folder / "index.faiss"))

        # Load docstore and mapping
        with open(folder / "index.pkl", "rb") as f:
            save_dict = pickle.load(f)

        # Handle different pickle formats (dict or tuple)
        if isinstance(save_dict, dict):
            # Expected format: dictionary with keys
            docstore_data = save_dict.get("docstore", {})
            index_to_docstore_id = save_dict.get("index_to_docstore_id", {})
        elif isinstance(save_dict, tuple):
            # Legacy format: might be a tuple, try to unpack
            # If tuple has 2 elements, assume (docstore_dict, index_to_docstore_id)
            if len(save_dict) == 2:
                docstore_data, index_to_docstore_id = save_dict
            else:
                raise ValueError(
                    f"Unexpected pickle format: tuple with {len(save_dict)} elements. "
                    f"Expected dictionary or tuple with 2 elements."
                )
        else:
            raise TypeError(
                f"Unexpected pickle format: {type(save_dict)}. "
                f"Expected dictionary or tuple."
            )

        # Handle docstore_data - could be a dict or InMemoryDocstore object
        docstore = InMemoryDocstore()
        if isinstance(docstore_data, dict):
            # It's a dictionary, use it directly
            docstore._dict = docstore_data
        elif isinstance(docstore_data, InMemoryDocstore):
            # It's already an InMemoryDocstore object, copy its _dict
            docstore._dict = docstore_data._dict.copy()
        else:
            # Try to convert to dict or raise error
            raise TypeError(
                f"Unexpected docstore format: {type(docstore_data)}. "
                f"Expected dictionary or InMemoryDocstore object."
            )

        return cls(
            embedding_function=embeddings.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
