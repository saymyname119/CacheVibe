from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app.clustering import FuzzyClustering
from app.config import CACHE_SIMILARITY_THRESHOLD, CLUSTERING_MODEL_DIR, CHROMA_PERSIST_DIR
from app.embeddings import EmbeddingModel
from app.search import SearchEngine
from app.semantic_cache import SemanticCache
from app.vector_store import VectorStore


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: Any
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


search_engine: Optional[SearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_engine

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(persist_dir=CHROMA_PERSIST_DIR)
    doc_count = vector_store.count()
    print(f"Vector store loaded: {doc_count} documents")

    clustering = FuzzyClustering()
    clustering.load(CLUSTERING_MODEL_DIR)

    cache = SemanticCache(similarity_threshold=CACHE_SIMILARITY_THRESHOLD)

    search_engine = SearchEngine(
        embedding_model=embedding_model,
        vector_store=vector_store,
        clustering=clustering,
        cache=cache,
    )

    print("Service ready!")
    yield
    print("Shutting down")


app = FastAPI(
    title="Semantic Search & Cache — 20 Newsgroups",
    description=(
        "A lightweight semantic search system with fuzzy clustering and "
        "a custom semantic cache layer. Built for the Trademarkia AI/ML "
        "engineer assignment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    result = search_engine.query(request.query)
    return QueryResponse(
        query=result.query,
        cache_hit=result.cache_hit,
        matched_query=result.matched_query,
        similarity_score=result.similarity_score,
        result=result.result,
        dominant_cluster=result.dominant_cluster,
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats_endpoint():
    stats = search_engine.cache.stats()
    return CacheStatsResponse(**stats)


@app.delete("/cache")
async def cache_clear_endpoint():
    search_engine.cache.clear()
    return {"message": "Cache cleared", "status": "ok"}
