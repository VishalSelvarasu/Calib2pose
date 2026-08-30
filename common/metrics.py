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


def wilson_interval(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion, returned as (lo, hi).

    Wilson rather than the normal approximation, which degenerates near p = 0
    and p = 1: at 28/28 it gives [1.0, 1.0], which 28 samples do not support.
    Wilson gives [0.879, 1.0] there. Default z = 1.96 is the 95% interval.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))
