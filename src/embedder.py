from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBED_MODEL

class Embedder:
    def __init__(self, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, normalize=True, batch_size=32):
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
        if normalize:
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        return embs

    def encode_one(self, text):
        emb = self.encode([text])
        return emb[0]
