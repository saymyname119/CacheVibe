import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from app.config import (
    CLUSTER_RANGE,
    CLUSTERING_MODEL_DIR,
    GMM_COVARIANCE_TYPE,
    GMM_MAX_ITER,
    GMM_N_INIT,
    PCA_COMPONENTS,
)


@dataclass
class ClusteringResult:
    n_clusters: int
    membership_probs: np.ndarray
    hard_labels: np.ndarray
    bic_scores: Dict[int, float] = field(default_factory=dict)
    silhouette_score: Optional[float] = None


class FuzzyClustering:

    def __init__(self):
        self.pca: Optional[PCA] = None
        self.gmm: Optional[GaussianMixture] = None
        self.n_clusters: Optional[int] = None

    def fit(self, embeddings: np.ndarray) -> ClusteringResult:
        print(f"Reducing dimensionality: {embeddings.shape[1]} → {PCA_COMPONENTS} via PCA")
        self.pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        reduced = self.pca.fit_transform(embeddings)
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"    PCA variance retained: {explained:.1%}")

        bic_scores: Dict[int, float] = {}
        best_bic = float("inf")
        best_k = None
        best_gmm = None

        for k in CLUSTER_RANGE:
            print(f"    Fitting GMM with k={k} …", end=" ", flush=True)
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=GMM_COVARIANCE_TYPE,
                max_iter=GMM_MAX_ITER,
                n_init=GMM_N_INIT,
                random_state=42,
            )
            gmm.fit(reduced)
            bic = gmm.bic(reduced)
            bic_scores[k] = bic
            print(f"BIC = {bic:,.0f}")

            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_gmm = gmm

        print(f"Best k = {best_k}  (BIC = {best_bic:,.0f})")

        self.gmm = best_gmm
        self.n_clusters = best_k

        membership_probs = self.gmm.predict_proba(reduced)
        hard_labels = membership_probs.argmax(axis=1)

        from sklearn.metrics import silhouette_score

        sample_size = min(10_000, len(reduced))
        rng = np.random.RandomState(42)
        indices = rng.choice(len(reduced), sample_size, replace=False)
        sil = silhouette_score(reduced[indices], hard_labels[indices])
        print(f"    Silhouette score (n={sample_size}): {sil:.3f}")

        result = ClusteringResult(
            n_clusters=best_k,
            membership_probs=membership_probs,
            hard_labels=hard_labels,
            bic_scores=bic_scores,
            silhouette_score=sil,
        )

        return result

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        reduced = self.pca.transform(embeddings)
        return self.gmm.predict_proba(reduced)

    def dominant_cluster(self, embeddings: np.ndarray) -> int:
        probs = self.predict_proba(embeddings.reshape(1, -1))
        return int(probs.argmax())

    def top_k_clusters(self, embeddings: np.ndarray, k: int = 2) -> List[int]:
        probs = self.predict_proba(embeddings.reshape(1, -1))[0]
        return list(np.argsort(probs)[-k:][::-1])

    def save(self, directory: str = CLUSTERING_MODEL_DIR) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "pca.pkl"), "wb") as f:
            pickle.dump(self.pca, f)
        with open(os.path.join(directory, "gmm.pkl"), "wb") as f:
            pickle.dump(self.gmm, f)
        with open(os.path.join(directory, "meta.pkl"), "wb") as f:
            pickle.dump({"n_clusters": self.n_clusters}, f)
        print(f"Clustering model saved to {directory}")

    def load(self, directory: str = CLUSTERING_MODEL_DIR) -> None:
        with open(os.path.join(directory, "pca.pkl"), "rb") as f:
            self.pca = pickle.load(f)
        with open(os.path.join(directory, "gmm.pkl"), "rb") as f:
            self.gmm = pickle.load(f)
        with open(os.path.join(directory, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
            self.n_clusters = meta["n_clusters"]
        print(f"Clustering model loaded from {directory}  (k={self.n_clusters})")
