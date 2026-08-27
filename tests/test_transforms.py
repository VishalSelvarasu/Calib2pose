import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import transforms as tf  # noqa: E402


def _rand_T(rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = rng.uniform(0.1, np.pi - 0.1)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return tf.make_T(R, rng.uniform(-1, 1, 3))


@pytest.mark.parametrize("seed", range(5))
def test_inv_T_is_a_true_inverse(seed):
    T = _rand_T(np.random.default_rng(seed))
    np.testing.assert_allclose(tf.inv_T(T) @ T, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(T @ tf.inv_T(T), np.eye(4), atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_inv_T_matches_numpy_inverse(seed):
    T = _rand_T(np.random.default_rng(seed))
    np.testing.assert_allclose(tf.inv_T(T), np.linalg.inv(T), atol=1e-12)


def test_make_T_layout():
    R = tf.rpy_to_R([0.1, 0.2, 0.3])
    t = np.array([1.0, 2.0, 3.0])
    T = tf.make_T(R, t)
    np.testing.assert_allclose(T[:3, :3], R)
    np.testing.assert_allclose(T[:3, 3], t)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_rpy_to_R_returns_a_rotation():
    R = tf.rpy_to_R([0.4, -0.2, 1.1])
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1) < 1e-12


def test_pose_error_is_zero_for_identical_poses():
    T = _rand_T(np.random.default_rng(11))
    t, r = tf.pose_error(T, T)
    assert t < 1e-12 and r < 1e-6


def test_pose_error_returns_metres_and_degrees():
    T1 = np.eye(4)
    T2 = tf.make_T(tf.rpy_to_R([0, 0, np.radians(30)]), [0.01, 0, 0])
    t, r = tf.pose_error(T2, T1)
    assert abs(t - 0.01) < 1e-12
    assert abs(r - 30.0) < 1e-9


def test_look_at_camera_points_z_at_the_target():
    eye = np.array([0.3, 0.2, 0.5])
    target = np.array([0.0, 0.0, 0.0])
    T = tf.look_at_camera(eye, target)
    z = T[:3, :3][:, 2]
    d = target - eye
    d /= np.linalg.norm(d)
    np.testing.assert_allclose(z, d, atol=1e-9)
    np.testing.assert_allclose(T[:3, 3], eye, atol=1e-12)
    assert abs(np.linalg.det(T[:3, :3]) - 1) < 1e-9
