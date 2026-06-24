import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
CLUSTERING_MODEL_DIR = os.path.join(DATA_DIR, "clustering")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

COLLECTION_NAME = "newsgroups"

PCA_COMPONENTS = 50
CLUSTER_RANGE = range(10, 36, 5)
GMM_COVARIANCE_TYPE = "full"
GMM_MAX_ITER = 300
GMM_N_INIT = 3

CACHE_SIMILARITY_THRESHOLD = 0.85
CACHE_TOP_K_CLUSTERS = 2
CACHE_MAX_ENTRIES = 10_000

SEARCH_TOP_K = 5
