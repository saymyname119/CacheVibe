# CacheVibe

### Intelligent Semantic Search & Adaptive Caching Platform

CacheVibe is an intelligent semantic search platform that combines dense vector retrieval, fuzzy clustering, and an adaptive semantic cache to deliver low-latency, context-aware information retrieval.

Instead of performing a full vector search for every query, the platform understands semantic similarity between user requests and reuses previous results whenever appropriate. This significantly reduces response latency while maintaining retrieval quality.

The system demonstrates practical applications of:

- Semantic embeddings
- Vector databases
- Fuzzy clustering
- Similarity search
- Intelligent caching strategies
- Production-ready API design

---

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

---

## Docker

### Build and run

```bash
docker build -t semcache .
docker run -p 8000:8000 semcache
```

### Or with docker-compose

```bash
docker-compose up --build
```

**Note:** Run `python -m scripts.setup_data` before building the Docker image so the `data/` directory contains the pre-computed vector store and clustering models.

---

## Problem Statement

Traditional search systems repeatedly execute expensive retrieval operations even when users ask semantically equivalent questions.

Examples:

- "What are the best graphics cards for gaming?"
- "Which GPU is ideal for playing modern video games?"

Although both questions seek the same information, conventional systems process them independently.

CacheVibe addresses this challenge by:

1. Understanding semantic meaning
2. Identifying related queries
3. Reusing previously computed results
4. Reducing computational overhead
5. Improving response latency

---

## System Architecture

```text
User Query
     ↓
Sentence Embedding Generation
     ↓
Fuzzy Cluster Prediction
     ↓
Semantic Cache Lookup
     ├── Cache Hit  → Return Cached Results
     └── Cache Miss → Vector Database Retrieval
                           ↓
                    Store Result in Cache
                           ↓
                    Return Response
```

---

## Technology Stack

### Backend

- FastAPI
- Uvicorn
- Python

### Machine Learning

- Sentence Transformers
- PCA
- Gaussian Mixture Models
- Cosine Similarity

### Data Layer

- ChromaDB
- Persistent Vector Storage
- Custom LRU Cache

### Deployment

- Docker
- Docker Compose

---

## Core Features

- Semantic Search
- Fuzzy Clustering
- Adaptive Semantic Cache
- Cluster-Aware Retrieval
- Analytics Support

---

## Project Highlights

- End-to-end semantic retrieval pipeline
- Production-grade FastAPI service
- Intelligent cache optimization
- Explainable clustering strategy
- Modular architecture
- Dockerized deployment
- Low-latency information retrieval system

---

## Future Enhancements

- Multi-user authentication
- Dashboard for cache analytics
- Query monitoring and visualization
- Streaming document ingestion
- Hybrid keyword + vector search
- Distributed semantic caching
- Retrieval-Augmented Generation integration

---

## Tagline

**CacheVibe – An Intelligent Semantic Search and Adaptive Caching Platform for Low-Latency Information Retrieval.**
