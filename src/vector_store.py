import faiss
import numpy as np
import json
from config import TARGET_INDEX_PATH, TARGETS_JSON, TARGET_EMB_PATH

class VectorStore:
    def __init__(self, dim=None):
        self.index = None
        self.dim = dim

    def build_from_targets(self, target_names, target_embs):
        """target_names: list[str], target_embs: numpy array (N, d) normalized"""
        n, d = target_embs.shape
        self.dim = d
        self.index = faiss.IndexFlatL2(d)
        self.index.add(target_embs.astype('float32'))
        # persist
        faiss.write_index(self.index, TARGET_INDEX_PATH)
        np.save(TARGET_EMB_PATH, target_embs)
        with open(TARGETS_JSON, 'w', encoding='utf-8') as f:
            json.dump(target_names, f, ensure_ascii=False)

    def load(self):
        try:
            self.index = faiss.read_index(TARGET_INDEX_PATH)
            return True
        except Exception:
            return False

    def search(self, query_embs, k=10):
        if self.index is None:
            raise RuntimeError('Index not loaded')
        D, I = self.index.search(query_embs.astype('float32'), k)
        # return distances and indices
        return D, I
