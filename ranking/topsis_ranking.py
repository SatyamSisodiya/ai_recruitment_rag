# ranking/topsis_ranking.py
import numpy as np
from pymcdm.methods import TOPSIS

def topsis_rank(decision_matrix: np.ndarray, weights: np.ndarray):
    """
    decision_matrix: shape (n_candidates, n_criteria) (benefit criteria)
    weights: shape (n_criteria,) sum to 1
    returns: scores (higher is better), ranks (1 is best)
    """
    # Use pymcdm TOPSIS implementation; all criteria are treated as benefit (+1)
    try:
        method = TOPSIS()
        types = np.ones(decision_matrix.shape[1])
        closeness = np.array(method(decision_matrix, weights, types))
    except Exception:
        # fallback: compute manually (basic TOPSIS with benefit criteria)
        norm = decision_matrix / np.sqrt((decision_matrix**2).sum(axis=0))
        weighted = norm * weights
        ideal_best = weighted.max(axis=0)
        ideal_worst = weighted.min(axis=0)
        dist_best = np.linalg.norm(weighted - ideal_best, axis=1)
        dist_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
        closeness = dist_worst / (dist_best + dist_worst + 1e-9)
    ranks = (-closeness).argsort().argsort() + 1
    return closeness, ranks
