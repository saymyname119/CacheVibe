import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import (
    CACHE_MAX_ENTRIES,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_TOP_K_CLUSTERS,
)


@dataclass
class CacheEntry:
    query_text: str
    query_embedding: np.ndarray
    result: Any
    dominant_cluster: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class CacheHitResult:
    matched_query: str
    similarity_score: float
    result: Any
    dominant_cluster: int


class SemanticCache:

    def __init__(
        self,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
        top_k_clusters: int = CACHE_TOP_K_CLUSTERS,
        max_entries: int = CACHE_MAX_ENTRIES,
    ):
        self.similarity_threshold = similarity_threshold
        self.top_k_clusters = top_k_clusters
        self.max_entries = max_entries

        self._buckets: Dict[int, OrderedDict[str, CacheEntry]] = {}

        self._hit_count = 0
        self._miss_count = 0

        self._lock = threading.RLock()

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_ids: List[int],
    ) -> Optional[CacheHitResult]:
        with self._lock:
            best_sim = -1.0
            best_entry: Optional[CacheEntry] = None

            for cid in cluster_ids:
                bucket = self._buckets.get(cid)
                if bucket is None:
                    continue

                for entry in bucket.values():
                    sim = float(np.dot(query_embedding, entry.query_embedding))
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry

            if best_entry is not None and best_sim >= self.similarity_threshold:
                self._hit_count += 1
                cid = best_entry.dominant_cluster
                bucket = self._buckets[cid]
                bucket.move_to_end(best_entry.query_text)
                return CacheHitResult(
                    matched_query=best_entry.query_text,
                    similarity_score=round(best_sim, 4),
                    result=best_entry.result,
                    dominant_cluster=best_entry.dominant_cluster,
                )

            self._miss_count += 1
            return None

    def put(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        result: Any,
        dominant_cluster: int,
    ) -> None:
        with self._lock:
            entry = CacheEntry(
                query_text=query_text,
                query_embedding=query_embedding,
                result=result,
                dominant_cluster=dominant_cluster,
            )

            cid = dominant_cluster
            if cid not in self._buckets:
                self._buckets[cid] = OrderedDict()

            self._buckets[cid][query_text] = entry

            if self._total_entries() > self.max_entries:
                self._evict_lru()

    def _total_entries(self) -> int:
        return sum(len(b) for b in self._buckets.values())

    def _evict_lru(self) -> None:
        oldest_time = float("inf")
        oldest_cid = None
        oldest_key = None

        for cid, bucket in self._buckets.items():
            if bucket:
                key = next(iter(bucket))
                entry = bucket[key]
                if entry.timestamp < oldest_time:
                    oldest_time = entry.timestamp
                    oldest_cid = cid
                    oldest_key = key

        if oldest_cid is not None and oldest_key is not None:
            del self._buckets[oldest_cid][oldest_key]

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()
            self._hit_count = 0
            self._miss_count = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._total_entries()
            total_queries = self._hit_count + self._miss_count
            return {
                "total_entries": total,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(self._hit_count / total_queries, 3) if total_queries > 0 else 0.0,
            }
