import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME


class EmbeddingModel:

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model ready  (dim={self.dim})")

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)
