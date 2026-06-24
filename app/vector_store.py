from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from app.config import CHROMA_PERSIST_DIR, COLLECTION_NAME


class VectorStore:

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        doc_ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> None:
        total = len(doc_ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self.collection.upsert(
                ids=doc_ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end],
            )
        print(f"Upserted {total} documents into ChromaDB")

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        kwargs = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()

    def get_all_embeddings(self) -> np.ndarray:
        result = self.collection.get(include=["embeddings"])
        return np.array(result["embeddings"], dtype=np.float32)

    def get_all_ids(self) -> List[str]:
        return self.collection.get(include=[])["ids"]

    def update_metadata(
        self,
        doc_ids: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> None:
        for start in range(0, len(doc_ids), batch_size):
            end = min(start + batch_size, len(doc_ids))
            self.collection.update(
                ids=doc_ids[start:end],
                metadatas=metadatas[start:end],
            )
