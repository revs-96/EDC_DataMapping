import os

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
TARGET_INDEX_PATH = os.path.join(DATA_DIR, 'targets.index')
TARGETS_JSON = os.path.join(DATA_DIR, 'targets.json')
RERANKER_PATH = os.path.join(DATA_DIR, 'reranker.joblib')
TARGET_EMB_PATH = os.path.join(DATA_DIR, 'target_embs.npy')

import os

# Existing configs (you already have EMBED_MODEL, RERANKER_PATH, TARGETS_JSON, TARGET_EMB_PATH, etc.)

BASE_DIR = os.path.dirname(__file__)

# Path to persist the trained reranker model
RERANKER_PATH = os.path.join(BASE_DIR, "reranker_model.pkl")

# Path to persist targets list
TARGETS_JSON = os.path.join(BASE_DIR, "targets.json")

# Path to persist target embeddings
TARGET_EMB_PATH = os.path.join(BASE_DIR, "target_embs.npy")

# ✅ NEW: Path to persist the ViewMapping XML (this is needed for human-in-the-loop updates)
VIEWMAPPING_XML_PATH = os.path.join(BASE_DIR, "ViewMapping.xml")

# Embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
