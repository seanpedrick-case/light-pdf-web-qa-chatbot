"""Optional Jina listwise reranker for hybrid retrieval."""

import logging
from typing import List, Optional, Tuple

import torch
from transformers import AutoModel

from tools.config import ENABLE_RERANKER, HF_TOKEN, RERANKER_MODEL_NAME
from tools.document import Document
from tools.model_load import torch_device

logger = logging.getLogger(__name__)

_reranker_model: Optional[object] = None


def load_reranker_model(model_name: str, device: str):
    """Load the Jina reranker with trust_remote_code."""
    kwargs = {
        "dtype": "auto",
        "trust_remote_code": True,
    }
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.eval()
    if device == "cuda" and hasattr(model, "to"):
        model = model.to(device)
    return model


def get_reranker_model():
    """Return a cached reranker instance, loading on first use."""
    global _reranker_model
    if not ENABLE_RERANKER:
        return None
    if _reranker_model is None:
        logger.info("Loading reranker model: %s", RERANKER_MODEL_NAME)
        _reranker_model = load_reranker_model(RERANKER_MODEL_NAME, torch_device)
    return _reranker_model


def rerank_passages(
    query: str,
    docs_keep_out: List[Tuple[Document, float]],
    top_n: Optional[int] = None,
) -> Tuple[List[Tuple[Document, float]], dict]:
    """
    Rerank (document, score) tuples with the Jina listwise reranker.

    On failure, returns the input list unchanged and an error entry in the meta dict.
    """
    meta: dict = {
        "enabled": bool(ENABLE_RERANKER),
        "model": RERANKER_MODEL_NAME,
        "candidates": len(docs_keep_out),
        "results": [],
    }
    if not docs_keep_out or not ENABLE_RERANKER:
        if not ENABLE_RERANKER:
            meta["skipped"] = True
            meta["reason"] = "ENABLE_RERANKER=False"
        return docs_keep_out, meta

    try:
        model = get_reranker_model()
        if model is None:
            meta["skipped"] = True
            meta["reason"] = "Reranker model could not be loaded"
            return docs_keep_out, meta

        documents = [item[0].page_content for item in docs_keep_out]
        effective_top_n = top_n if top_n is not None else len(documents)

        with torch.inference_mode():
            results = model.rerank(query, documents, top_n=effective_top_n)

        reranked: List[Tuple[Document, float]] = []
        for rank, entry in enumerate(results, 1):
            idx = entry["index"]
            score = float(entry["relevance_score"])
            doc = docs_keep_out[idx][0]
            reranked.append((doc, score))
            meta["results"].append(
                {
                    "rank": rank,
                    "relevance_score": score,
                    "index": idx,
                    "source": (getattr(doc, "metadata", None) or {}).get("source", ""),
                }
            )
        return reranked, meta
    except Exception:
        logger.warning("Reranker failed; keeping hybrid retrieval order", exc_info=True)
        meta["error"] = "rerank_failed"
        return docs_keep_out, meta
