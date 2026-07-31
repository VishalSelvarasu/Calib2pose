"""Rigid transform helpers, shared across all stages.

Previously duplicated in stage2_handeye/sim_scene.py and
stage3_pose/drill_scene.py. Divergence between copies is a real risk once a
convention changes, so they live here.
"""

import numpy as np
import mujoco

# MuJoCo/OpenGL camera frame (-Z forward, +Y up) -> OpenCV (+Z forward, +Y down)
R_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def rpy_to_R(rpy):
    """Intrinsic XYZ (roll-pitch-yaw), radians. R = Rz @ Ry @ Rx."""
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def R_to_quat_wxyz(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).ravel())
    return q


def pose_error(T_est, T_true):
    """(translation in metres, rotation in degrees)."""
    t = float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))
    c = (np.trace(T_true[:3, :3].T @ T_est[:3, :3]) - 1) / 2
    return t, float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def look_at_camera(eye, target, roll_deg=0.0):
    """Camera pose in OpenCV convention (+Z toward target, +Y down)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(z @ ref) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    if roll_deg:
        c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return make_T(R, eye)
