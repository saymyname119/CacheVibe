# Semantic Search & Cache — 20 Newsgroups

A lightweight semantic search system with fuzzy clustering and a custom semantic cache layer, built on the 20 Newsgroups dataset.

## Architecture

```
Query → Embed (all-MiniLM-L6-v2) → GMM cluster → Semantic cache check
                                                     ├─ HIT  → return cached result
                                                     └─ MISS → ChromaDB vector search
                                                                → cache result → return
```

**Key components:**

| Component           | Implementation                                 |
|---------------------|-------------------------------------------------|
| Embedding model     | `all-MiniLM-L6-v2` (384-dim, sentence-transformers) |
| Vector store        | ChromaDB (local, persistent)                    |
| Fuzzy clustering    | PCA (50-dim) + Gaussian Mixture Model           |
| Semantic cache      | Custom dict+OrderedDict, cluster-bucketed, LRU  |
| API                 | FastAPI + uvicorn                               |

## Quick Start

### 1. Set up the environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run data setup (one-time)

This downloads the 20 Newsgroups dataset, computes embeddings, indexes them in ChromaDB, and fits the clustering model.

```bash
python -m scripts.setup_data
```

### 3. Start the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Use the API

```bash
# Search
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best graphics cards for gaming?"}'

# Similar query (should be a cache hit)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which GPU is best for playing video games?"}'

# Cache statistics
curl http://localhost:8000/cache/stats

# Clear cache
curl -X DELETE http://localhost:8000/cache
```

## Docker

```bash
# Build and run
docker build -t semcache .
docker run -p 8000:8000 semcache

# Or with docker-compose
docker-compose up --build
```

> **Note:** Run `python -m scripts.setup_data` before building the Docker image so the `data/` directory contains the pre-computed vector store and clustering models.

## Design Decisions

### Embedding Model — `all-MiniLM-L6-v2`
384-dim vectors. Chosen over larger models (e.g. `all-mpnet-base-v2` at 768-dim) for speed: the system needs low-latency query embedding, and smaller vectors reduce storage and clustering compute.

### Fuzzy Clustering — GMM
Gaussian Mixture Models produce genuine probability distributions over clusters (`predict_proba()`)—not just hard labels. A post about "gun legislation" gets significant probability mass in both a politics cluster and a firearms cluster. BIC scoring selects the optimal number of clusters without manual elbow-plot inspection.

### Semantic Cache — The Similarity Threshold
The central tunable parameter is `CACHE_SIMILARITY_THRESHOLD` (default: 0.85). This single value controls what "close enough" means:

| Threshold | Behaviour | Hit Rate | Quality |
|-----------|-----------|----------|---------|
| 0.95 | Near-exact paraphrases only | Very low | Very high |
| 0.90 | Close paraphrases | Moderate | High |
| **0.85** | **Semantically equivalent queries** | **Good** | **High** |
| 0.80 | Broadly similar topics | High | Moderate |
| 0.70 | Loosely related queries | Very high | Low |

The insight: changing the threshold doesn't just adjust hit-rate—it changes *what* the cache fundamentally is. At 0.95 it's a deduplication engine; at 0.85 a semantic equivalence detector; at 0.70 a topic classifier.

### Cluster-Aware Cache Lookup
Instead of comparing a query against ALL cached entries (O(n)), we only check the top-2 GMM cluster candidates (O(n/k)). This is where clustering does "real work" for the cache.

### Data Cleaning
Headers, quoted replies, signatures, emails, URLs, and file paths are stripped. Posts under 50 characters after cleaning are dropped—too little signal for meaningful embeddings.

## Clustering Analysis

After running `python -m scripts.setup_data`, additional analysis is available:

```bash
python -m analysis.clustering_analysis
```

This produces:
- Cluster composition with purity scores
- Category ↔ cluster cross-tabulation
- Fuzzy membership statistics
- Boundary documents (highest entropy)
- Cluster overlap matrix

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI endpoints
│   ├── config.py             # Tunable parameters
│   ├── data_loader.py        # Dataset cleaning
│   ├── embeddings.py         # Embedding model wrapper
│   ├── vector_store.py       # ChromaDB wrapper
│   ├── clustering.py         # GMM fuzzy clustering
│   ├── semantic_cache.py     # Custom cache (no libraries)
│   └── search.py             # Query orchestration
├── scripts/
│   └── setup_data.py         # One-time data ingestion
├── analysis/
│   └── clustering_analysis.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```
# SemCche
