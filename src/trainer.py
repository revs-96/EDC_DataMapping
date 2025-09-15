import numpy as np
from tqdm import tqdm
from xml_loader import parse_source_xml, parse_viewmapping_xml
from embedder import Embedder
from vector_store import VectorStore
from features import build_candidate_features
from reranker import Reranker
from persist import load_targets
from config import TARGETS_JSON
import json

def build_target_list_from_viewmapping(mappings):
    targets = sorted({m['EDCAttributeID'] for m in mappings})
    return targets

def create_training_data(source_events, mappings, k=10):
    embedder = Embedder()
    targets = build_target_list_from_viewmapping(mappings)
    target_embs = embedder.encode(targets)

    vs = VectorStore()
    vs.build_from_targets(targets, target_embs)

    visit_to_targets = {}
    for m in mappings:
        visit_to_targets.setdefault(m['EDCVisitID'], []).append(m['EDCAttributeID'])

    X = []
    y = []
    for src in tqdm(source_events, desc='Generating training data'):
        src_str = ' '.join([itm['ItemOID'] for itm in src.get('Items', []) if itm.get('ItemOID')]) or src.get('StudyEventOID','')
        s_emb = embedder.encode([src_str])
        D, I = vs.search(s_emb, k)
        pos_targets = set(visit_to_targets.get(src.get('StudyEventOID'), []))

        for dist, tid in zip(D[0], I[0]):
            cosine_sim = 1 - dist / 2.0
            target_name = targets[tid]
            feat = build_candidate_features(src, target_name, cosine_sim)
            label = 1 if target_name in pos_targets else 0
            X.append(feat)
            y.append(label)

    X = np.vstack(X)
    y = np.array(y)
    return X, y, targets, target_embs

def train_pipeline(source_xml_bytes, viewmapping_xml_bytes):
    source_events = parse_source_xml(source_xml_bytes)
    mappings = parse_viewmapping_xml(viewmapping_xml_bytes)
    X, y, targets, target_embs = create_training_data(source_events, mappings, k=10)

    reranker = Reranker()
    reranker.train(X, y, n_estimators=200, learning_rate=0.1)

    with open(TARGETS_JSON, 'w', encoding='utf-8') as f:
        json.dump(targets, f, ensure_ascii=False)

    return {'n_samples': len(y), 'n_pos': int(y.sum())}
