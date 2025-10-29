# scoring/scoring.py
from typing import Dict, List
import numpy as np
from difflib import SequenceMatcher
from utils import normalize_text
import re

# Common aliases to improve exact/substring matches
_ALIASES = {
    "scikit learn": "sklearn",
    "scikit-learn": "sklearn",
    "node.js": "node",
    "nodejs": "node",
    "c plus plus": "c++",
    "c sharp": "c#",
    "power bi": "powerbi",
    "ms excel": "excel",
    "microsoft excel": "excel",
}

def _canonicalize_skill(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    # remove backticks and surrounding punctuation
    s = s.replace("`", " ")
    # replace common separators with space
    s = re.sub(r"[/|+]+", " ", s)
    # normalize aliases
    for k, v in _ALIASES.items():
        s = s.replace(k, v)
    # drop version numbers (e.g., python 3.10 -> python)
    s = re.sub(r"\b\d+(?:\.\d+)*\b", "", s)
    # keep letters/digits/#/+
    s = re.sub(r"[^a-z0-9#+]+", " ", s)
    s = normalize_text(s)
    return s

def _split_items(items: List[str]) -> List[str]:
    out = []
    for it in items or []:
        if not it:
            continue
        # make simple separators uniform
        tmp = it.replace(" and ", ", ")
        tmp = tmp.replace("/", ", ").replace("•", ", ").replace("|", ", ")
        parts = re.split(r"[,;\n]+", tmp)
        out.extend([p.strip() for p in parts if p and p.strip()])
    return out

def _match_tokens(a: str, b: str) -> bool:
    # exact or substring either way
    if a == b or a in b or b in a:
        return True
    # token-level Jaccard overlap
    ta, tb = set(a.split()), set(b.split())
    if ta and tb:
        jacc = len(ta & tb) / max(1, len(ta | tb))
        if jacc >= 0.5:
            return True
    # fuzzy ratio
    if SequenceMatcher(None, a, b).ratio() >= 0.8:
        return True
    return False

def keyword_match_score(jd_skills: List[str], resume_skills: List[str]) -> float:
    """Compute coverage of JD skills in resume skills using forgiving matching.

    - Splits phrases into atomic items
    - Canonicalizes names (aliases, punctuation, versions)
    - Matches by exact, substring, token Jaccard, or fuzzy similarity
    """
    if not jd_skills:
        return 1.0
    jd_items = _split_items(jd_skills) if any(
        isinstance(x, str) and ("," in x or ";" in x or " and " in x or "\n" in x or "/" in x or "|" in x)
        for x in jd_skills
    ) else list(jd_skills)
    res_items = list(resume_skills or [])

    jd_norm = [_canonicalize_skill(x) for x in jd_items if x]
    res_norm = [_canonicalize_skill(x) for x in res_items if x]
    # remove empties
    jd_norm = [x for x in jd_norm if x]
    res_norm = [x for x in res_norm if x]
    if not jd_norm:
        return 1.0

    matched_jd = 0
    for j in set(jd_norm):
        if any(_match_tokens(j, r) for r in set(res_norm)):
            matched_jd += 1
    score = matched_jd / len(set(jd_norm))
    return float(max(0.0, min(1.0, score)))

def semantic_similarity_score(jd_texts: List[str], resume_texts: List[str], embed_model) -> float:
    """
    Compute semantic similarity by averaging cosine similarities between JD and resume chunks.
    embed_model: sentence-transformers model (with .encode)
    """
    if not jd_texts or not resume_texts:
        return 0.0
    jv = embed_model.encode(jd_texts, convert_to_numpy=True)
    rv = embed_model.encode(resume_texts, convert_to_numpy=True)
    # normalize
    jv = jv / (np.linalg.norm(jv, axis=1, keepdims=True) + 1e-9)
    rv = rv / (np.linalg.norm(rv, axis=1, keepdims=True) + 1e-9)
    # pairwise similarities
    sims = np.dot(jv, rv.T)
    # take top matches per JD chunk
    top_per_jd = sims.max(axis=1)
    return float(np.mean(top_per_jd))

def experience_score(required_years: float, candidate_years: float) -> float:
    if required_years <= 0:
        return 1.0
    return float(max(0.0, min(1.0, candidate_years / required_years)))

def education_score(required_level: str, candidate_degrees: List[str]) -> float:
    """
    Map degrees to levels: highschool=0, bachelor=1, master=2, phd=3
    required_level can be 'bachelor','master','phd' etc.
    """
    level_map = {"highschool": 0, "bachelor": 1, "bachelors": 1, "bsc":1, "ba":1,
                 "master": 2, "masters":2, "msc":2, "ms":2,
                 "phd": 3, "doctor":3}
    req = level_map.get(required_level.lower(), 1) if required_level else 1
    cand_levels = [level_map.get(d.lower(), 0) for d in candidate_degrees]
    cand = max(cand_levels) if cand_levels else 0
    # score proportionally (if candidate >= required -> 1)
    if cand >= req:
        return 1.0
    if req == 0:
        return 1.0
    return float(cand / req)

def combined_global_score(keyword_s: float, semantic_s: float, w_k=0.4, w_s=0.6) -> float:
    return float(max(0.0, min(1.0, w_k*keyword_s + w_s*semantic_s)))
