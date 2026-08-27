import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "stage2_handeye"))
sys.path.insert(0, ROOT)

import handeye_solvers as hs           # noqa: E402
from common import transforms as tf    # noqa: E402


def _rand_T(rng, max_deg=60.0, max_t=0.5):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.radians(rng.uniform(10.0, max_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return tf.make_T(R, rng.uniform(-max_t, max_t, 3))


def _synthetic(n=12, seed=0):
    """Build flange and board poses consistent with a known T_flange_cam."""
    rng = np.random.default_rng(seed)
    X = _rand_T(rng, max_deg=50.0, max_t=0.15)     # the answer
    T_base_board = _rand_T(rng, max_deg=40.0, max_t=0.6)
    flange, board = [], []
    for _ in range(n):
        T_bf = _rand_T(rng, max_deg=70.0, max_t=0.4)
        # T_base_board = T_bf @ X @ T_cam_board  =>  solve for T_cam_board
        T_cb = np.linalg.inv(X) @ np.linalg.inv(T_bf) @ T_base_board
        flange.append(T_bf)
        board.append(T_cb)
    return X, flange, board


@pytest.mark.parametrize("name", ["PARK", "TSAI", "ANDREFF"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_solver_recovers_known_transform(name, seed):
    X, flange, board = _synthetic(seed=seed)
    T_est = hs.SOLVERS[name](flange, board)
    t_m, R_deg = tf.pose_error(T_est, X)
    t_mm = t_m * 1000.0
    assert t_mm < 1e-6, f"{name}: translation off by {t_mm:.3e} mm"
    # The arccos-based angle error has a sqrt(eps) floor, so 1e-4 deg is
    # numerical precision here, not a loose tolerance.
    assert R_deg < 1e-4, f"{name}: rotation off by {R_deg:.3e} deg"


@pytest.mark.parametrize("name", ["PARK", "TSAI", "ANDREFF"])
def test_solvers_agree_with_each_other(name):
    X, flange, board = _synthetic(seed=7)
    ref = hs.SOLVERS["PARK"](flange, board)
    est = hs.SOLVERS[name](flange, board)
    t_m, R_deg = tf.pose_error(est, ref)
    assert t_m * 1000.0 < 1e-6 and R_deg < 1e-4


def test_degenerate_motion_is_detected_by_axis_spread():
    """Yaw-only motion: the spread metric must flag it, because the solvers
    themselves will not. This is the Stage 2 lesson as an assertion."""
    rng = np.random.default_rng(3)
    flange = []
    for _ in range(12):
        ang = rng.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(ang), -np.sin(ang), 0],
                      [np.sin(ang), np.cos(ang), 0],
                      [0, 0, 1.0]])
        flange.append(tf.make_T(R, rng.uniform(-0.3, 0.3, 3)))
    assert hs.motion_axis_spread(flange) < 10.0

    _, diverse, _ = _synthetic(seed=5)
    assert hs.motion_axis_spread(diverse) >= 10.0
