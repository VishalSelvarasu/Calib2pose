"""6D pose metrics."""

import numpy as np


def add_metric(model_pts, T_est, T_true):
    """ADD: mean distance between model points under the two poses.

    Valid for asymmetric objects only. The YCB power drill qualifies.
    """
    a = (T_est[:3, :3] @ model_pts.T).T + T_est[:3, 3]
    b = (T_true[:3, :3] @ model_pts.T).T + T_true[:3, 3]
    return float(np.linalg.norm(a - b, axis=1).mean())


def adds_metric(model_pts, T_est, T_true, max_pts=2000):
    """ADD-S: nearest-neighbour variant, for symmetric objects.

    Subsampled because the full N^2 distance matrix on 8945 points is 80M
    entries per view. Included for completeness; the drill does not need it.
    """
    from scipy.spatial import cKDTree
    idx = np.random.default_rng(0).choice(
        len(model_pts), min(max_pts, len(model_pts)), replace=False)
    p = model_pts[idx]
    a = (T_est[:3, :3] @ p.T).T + T_est[:3, 3]
    b = (T_true[:3, :3] @ p.T).T + T_true[:3, 3]
    d, _ = cKDTree(b).query(a)
    return float(d.mean())
