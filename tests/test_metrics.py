import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.metrics import add_metric, adds_metric   # noqa: E402
from common import transforms as tf                  # noqa: E402


def _pts(n=500, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 3)) * 0.1


def test_both_metrics_are_zero_for_an_exact_pose():
    pts, T = _pts(), np.eye(4)
    assert add_metric(pts, T, T) < 1e-12
    assert adds_metric(pts, T, T) < 1e-12


def test_pure_translation_gives_exactly_that_distance():
    """ADD under a pure translation is the translation magnitude, by
    definition. A unit or convention slip shows up here immediately."""
    pts = _pts()
    T_est = tf.make_T(np.eye(3), [0.005, 0.0, 0.0])
    assert abs(add_metric(pts, T_est, np.eye(4)) - 0.005) < 1e-12


def test_adds_never_exceeds_add():
    """Nearest-neighbour matching can only shorten distances. If ADD-S comes
    out larger, the KD-tree is being queried against the wrong point set."""
    pts = _pts(seed=1)
    for rpy, t in [([0.05, 0.02, 0.10], [0.003, 0.0, 0.001]),
                   ([0.30, 0.00, 0.00], [0.0, 0.010, 0.0]),
                   ([0.00, 0.50, 0.20], [0.020, 0.0, 0.005])]:
        T_est = tf.make_T(tf.rpy_to_R(rpy), t)
        assert adds_metric(pts, T_est, np.eye(4)) <= \
            add_metric(pts, T_est, np.eye(4)) + 1e-12


def test_adds_is_deterministic_under_subsampling():
    """adds_metric subsamples to max_pts. Repeated calls on the same input
    must return the same value, or the metric is not reproducible."""
    pts = _pts(n=5000, seed=2)
    T_est = tf.make_T(tf.rpy_to_R([0.1, 0.05, 0.02]), [0.004, 0.0, 0.002])
    first = adds_metric(pts, T_est, np.eye(4))
    for _ in range(3):
        assert adds_metric(pts, T_est, np.eye(4)) == first


def test_add_scales_linearly_with_translation():
    pts = _pts(seed=3)
    a = add_metric(pts, tf.make_T(np.eye(3), [0.002, 0, 0]), np.eye(4))
    b = add_metric(pts, tf.make_T(np.eye(3), [0.004, 0, 0]), np.eye(4))
    assert abs(b - 2 * a) < 1e-12
