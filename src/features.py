import re
import numpy as np
from dateutil import parser as dateparser
from Levenshtein import distance as lev_distance
from collections import Counter

# Common date formats or tokens we expect (extended in heuristics)
_DATE_KEYWORDS = {'date', 'dob', 'birth', 'day', 'mm', 'yy', 'yy-mm', 'year'}

def safe_parse_date(s):
    s = (s or '').strip()
    if not s:
        return None
    try:
        dt = dateparser.parse(s, fuzzy=False)
        return dt
    except Exception:
        return None

def infer_dtype_from_samples(values):
    vals = [v for v in values if v not in [None, '', 'NA', 'na']]
    if not vals:
        return 'string'
    n = len(vals)
    n_date = sum(1 for v in vals if safe_parse_date(v) is not None)
    n_int = 0
    n_float = 0
    unique_vals = set()
    for v in vals:
        vv = v.strip()
        unique_vals.add(vv)
        if vv.isdigit():
            n_int += 1
        else:
            try:
                float(vv.replace(',', ''))
                n_float += 1
            except Exception:
                pass
    if n_date / n > 0.6:
        return 'date'
    if (n_int + n_float) / n > 0.8 and len(unique_vals) > 5:
        return 'float' if n_float > n_int else 'int'
    if len(unique_vals) <= 10 and len(unique_vals) / n < 0.5:
        return 'enum'
    return 'string'

def normalized_levenshtein(a, b):
    a2 = (a or '').lower()
    b2 = (b or '').lower()
    if len(a2) + len(b2) == 0:
        return 0.0
    return lev_distance(a2, b2) / max(1, max(len(a2), len(b2)))

def extract_value_patterns(values, top_k=3):
    patterns = Counter()
    for v in values:
        s = (v or '').strip()
        if not s:
            patterns['<EMPTY>'] += 1
            continue
        if safe_parse_date(s):
            patterns['<DATE>'] += 1
            continue
        if re.fullmatch(r'^\d+$', s):
            patterns[f'<DIGITS:{len(s)}>' ] += 1
            continue
        if re.fullmatch(r'^\d+[.,]\d+$', s):
            patterns['<DECIMAL>'] += 1
            continue
        tokens = re.findall(r'[A-Za-z]+', s)
        if tokens:
            patterns['<ALPHA>'] += 1
        else:
            patterns['<OTHER>'] += 1
    most = [p for p,count in patterns.most_common(top_k)]
    return most

def sample_value_match_rate(values, target_tokens):
    if not values:
        return 0.0
    tks = [t.lower() for t in re.split(r'[\W_]+', target_tokens) if t]
    if not tks:
        return 0.0
    hits = 0
    for v in values:
        s = (v or '').lower()
        if any(t in s for t in tks):
            hits += 1
    return hits / len(values)

def cardinality_stats(values):
    vals = [v for v in values if v not in [None, '', 'NA', 'na']]
    total = len(values)
    non_null = len(vals)
    unique = len(set(vals))
    null_frac = 1.0 - (non_null / total) if total>0 else 1.0
    unique_frac = unique / max(1, non_null) if non_null>0 else 0.0
    return {'total': total, 'non_null': non_null, 'unique': unique, 'null_frac': null_frac, 'unique_frac': unique_frac}

def build_candidate_features(source_row, target_name, cosine_sim):
    src_item_oids = ' '.join([itm.get('ItemOID','') for itm in source_row.get('Items', [])])
    items_values = [str(it.get('Value') or '') for it in source_row.get('Items', [])]

    lev = normalized_levenshtein(src_item_oids, target_name)
    src_dtype = infer_dtype_from_samples(items_values)
    tkns = [t.lower() for t in re.split(r'[\W_]+', target_name) if t]
    tgt_dtype = 'date' if any(k in tkns for k in _DATE_KEYWORDS) else 'string'
    same_dtype = 1 if src_dtype == tgt_dtype else 0

    cstats = cardinality_stats(items_values)
    tgt_unique_frac = 0.5
    card_sim = 1 - abs(cstats['unique_frac'] - tgt_unique_frac)
    null_diff = 1 - abs(cstats['null_frac'] - 0.0)

    sample_match = sample_value_match_rate(items_values, target_name)
    src_patterns = extract_value_patterns(items_values)
    pattern_sim = len(set([p.lower() for p in src_patterns]) & set([t.lower() for t in re.split(r'[\W_]+', target_name) if t])) / max(1, len(src_patterns))

    feats = np.array([
        cosine_sim,
        lev,
        1.0 if same_dtype else 0.0,
        card_sim,
        null_diff,
        sample_match,
        pattern_sim,
    ], dtype=float)
    return feats
