from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from app.clustering import FuzzyClustering
from app.config import CACHE_TOP_K_CLUSTERS, SEARCH_TOP_K
from app.embeddings import EmbeddingModel
from app.semantic_cache import CacheHitResult, SemanticCache
from app.vector_store import VectorStore


@dataclass
class QueryResult:
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: Any
    dominant_cluster: int


class SearchEngine:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        clustering: FuzzyClustering,
        cache: SemanticCache,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.clustering = clustering
        self.cache = cache

    def query(self, query_text: str) -> QueryResult:
        query_embedding = self.embedding_model.encode(
            [query_text], show_progress=False
        )[0]

        top_clusters = self.clustering.top_k_clusters(
            query_embedding, k=CACHE_TOP_K_CLUSTERS
        )
        dominant_cluster = top_clusters[0]

        hit: Optional[CacheHitResult] = self.cache.lookup(
            query_embedding, cluster_ids=top_clusters
        )

        if hit is not None:
            return QueryResult(
                query=query_text,
                cache_hit=True,
                matched_query=hit.matched_query,
                similarity_score=hit.similarity_score,
                result=hit.result,
                dominant_cluster=hit.dominant_cluster,
            )

        search_results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=SEARCH_TOP_K,
        )

        result_docs: List[Dict[str, Any]] = []
        if search_results["documents"] and search_results["documents"][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                search_results["documents"][0],
                search_results["metadatas"][0],
                search_results["distances"][0],
            )):
                result_docs.append({
                    "rank": i + 1,
                    "text": doc[:500],
                    "category": meta.get("category", "unknown"),
                    "similarity": round(1 - dist, 4),
                })

        self.cache.put(
            query_text=query_text,
            query_embedding=query_embedding,
            result=result_docs,
            dominant_cluster=dominant_cluster,
        )

        return QueryResult(
            query=query_text,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result=result_docs,
            dominant_cluster=dominant_cluster,
        )
