#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CHROMA_PERSIST_DIR, CLUSTERING_MODEL_DIR, DATA_DIR
from app.data_loader import load_and_clean_dataset
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore
from app.clustering import FuzzyClustering


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    documents, categories = load_and_clean_dataset()

    model = EmbeddingModel()
    texts = [doc.text for doc in documents]
    print(f"Embedding {len(texts)} documents")
    embeddings = model.encode(texts, batch_size=256, show_progress=True)
    print(f"    Embeddings shape: {embeddings.shape}")

    store = VectorStore(persist_dir=CHROMA_PERSIST_DIR)

    if store.count() == 0:
        doc_ids = [f"doc_{doc.doc_id}" for doc in documents]
        metadatas = [
            {
                "category": doc.category,
                "original_index": doc.original_index,
                "doc_id": doc.doc_id,
            }
            for doc in documents
        ]
        store.add_documents(
            doc_ids=doc_ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    else:
        print(f"ChromaDB already contains {store.count()} documents; skipping ingestion")

    clustering = FuzzyClustering()
    result = clustering.fit(embeddings)
    clustering.save(CLUSTERING_MODEL_DIR)

    np.save(os.path.join(CLUSTERING_MODEL_DIR, "membership_probs.npy"), result.membership_probs)
    np.save(os.path.join(CLUSTERING_MODEL_DIR, "hard_labels.npy"), result.hard_labels)

    with open(os.path.join(CLUSTERING_MODEL_DIR, "bic_scores.txt"), "w") as f:
        for k, bic in sorted(result.bic_scores.items()):
            f.write(f"k={k}\tBIC={bic:.2f}\n")
        f.write(f"\nBest k={result.n_clusters}\n")
        f.write(f"Silhouette score={result.silhouette_score:.4f}\n")

    doc_ids = [f"doc_{doc.doc_id}" for doc in documents]
    cluster_metadatas = [
        {"cluster_id": int(result.hard_labels[i])}
        for i in range(len(documents))
    ]
    store.update_metadata(doc_ids, cluster_metadatas)
    print("Cluster assignments written to ChromaDB metadata")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"  Documents indexed:     {store.count()}")
    print(f"  Embedding dimensions:  {embeddings.shape[1]}")
    print(f"  Number of clusters:    {result.n_clusters}")
    print(f"  Silhouette score:      {result.silhouette_score:.4f}")
    print(f"  ChromaDB path:         {CHROMA_PERSIST_DIR}")
    print(f"  Clustering model path: {CLUSTERING_MODEL_DIR}")
    print("=" * 60)

    print("\nCluster distribution:")
    unique, counts = np.unique(result.hard_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"    Cluster {cluster_id:>2}: {count:>5} docs")

    print("\nCategory → Cluster mapping (top category per cluster):")
    cat_labels = [doc.category for doc in documents]
    for cid in range(result.n_clusters):
        mask = result.hard_labels == cid
        cluster_cats = [cat_labels[i] for i in range(len(cat_labels)) if mask[i]]
        if cluster_cats:
            from collections import Counter
            top_cats = Counter(cluster_cats).most_common(3)
            cats_str = ", ".join(f"{cat}({n})" for cat, n in top_cats)
            print(f"    Cluster {cid:>2}: {cats_str}")

    print("\nTop-10 boundary documents (highest cluster uncertainty):")
    probs = result.membership_probs
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
    top_boundary_idx = np.argsort(entropy)[-10:][::-1]
    for idx in top_boundary_idx:
        doc = documents[idx]
        top2 = np.argsort(probs[idx])[-2:][::-1]
        top2_probs = probs[idx][top2]
        print(f"    Doc {doc.doc_id} [{doc.category}]: "
              f"entropy={entropy[idx]:.3f}, "
              f"top clusters: {top2[0]}({top2_probs[0]:.2f}), "
              f"{top2[1]}({top2_probs[1]:.2f})")
        print(f"      Preview: {doc.text[:100]}…")


if __name__ == "__main__":
    main()
