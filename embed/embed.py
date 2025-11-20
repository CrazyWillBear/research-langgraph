import numpy as np
from numpy import float32
from sentence_transformers import SentenceTransformer


class Embeder:
    """Embedder using BAAI/bge-large-en-v1.5 model."""

    DEVICE = "cuda"

    def __init__(self):
        """Initialize the BAAI/bge-large-en-v1.5 model for embedding."""

        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=self.DEVICE)


    def embed(self, text: str):
        """Embed text into a dense vector using BAAI/bge-large-en-v1.5."""

        # Model expects a list of sentences
        vector = self.model.encode([text], normalize_embeddings=True)

        return np.array(vector[0], dtype=float32).ravel().tolist()

    def embed_batch(self, texts: list[str]):
        # Embed a batch of texts into dense vectors using BAAI/bge-large-en-v1.5.
        vectors = self.model.encode(texts, normalize_embeddings=True)

        return [np.array(vec, dtype=float32).ravel().tolist() for vec in vectors]