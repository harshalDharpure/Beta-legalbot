"""Legal RAG pipeline: hybrid retrieval, citation prompts, best-of-N selection."""

from .retrieve import HybridRetriever, load_index_bundle

__all__ = ["HybridRetriever", "load_index_bundle"]
