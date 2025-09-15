import joblib
import numpy as np
from lightgbm import LGBMClassifier
from config import RERANKER_PATH

class Reranker:
    def __init__(self):
        self.model = None

    def train(self, X, y, **kwargs):
        self.model = LGBMClassifier(**kwargs)
        self.model.fit(X, y)
        joblib.dump(self.model, RERANKER_PATH)

    def load(self):
        try:
            self.model = joblib.load(RERANKER_PATH)
            return True
        except Exception:
            return False

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError('Reranker not loaded')
        return self.model.predict_proba(X)[:, 1]
