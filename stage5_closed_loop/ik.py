import numpy as np
import mujoco

from common import transforms as tf


def pose_error(T_current, T_target):
    """6-vector [translation; rotation] error, rotation as an axis-angle
    vector in the base frame."""
    e_pos = T_target[:3, 3] - T_current[:3, 3]
    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    rvec = np.empty(3)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R_err).ravel())
    mujoco.mju_quat2Vel(rvec, quat, 1.0)
    return np.concatenate([e_pos, rvec])


def solve_ik(scene, T_target, q_init=None, iters=200, damping=0.08,
             pos_tol=1e-4, rot_tol=1e-3, step_clip=0.3):
    """Return (q, converged, final_error).

    q_init matters: IK is local, and starting from the current configuration
    keeps the arm near where it already is rather than flipping to a distant
    but equally valid solution.
    """
    q = (scene.data.qpos[:6].copy()
         if q_init is None else np.array(q_init, float))
    lo = scene.model.jnt_range[:6, 0]
    hi = scene.model.jnt_range[:6, 1]

    for _ in range(iters):
        scene.set_q(q)
        e = pose_error(scene.flange_pose(), T_target)
        if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
            return q, True, e

        J = scene.jacobian()
        JJt = J @ J.T + (damping ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)

        n = np.linalg.norm(dq)
        if n > step_clip:
            dq *= step_clip / n
        q = np.clip(q + dq, lo, hi)

    scene.set_q(q)
    return q, False, pose_error(scene.flange_pose(), T_target)


def grasp_pose_from_object(T_base_obj, approach_offset=0.10):
    """Where the flange should go, given where the object is.

    Approaches straight down from `approach_offset` above the object origin.
    Deliberately simple: the point of stage 5 is to measure how POSE error
    propagates into placement error, not to design a grasp. Any fixed,
    deterministic function of the object pose serves, because the same function
    is applied to the estimate and to ground truth, so the difference isolates
    the perception error.
    """
    R_down = np.array([[1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0]])
    R = T_base_obj[:3, :3] @ np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    # Keep only the object's yaw; approach vertically regardless of its tilt.
    yaw = np.arctan2(R[1, 0], R[0, 0])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1.0]])
    t = T_base_obj[:3, 3] + np.array([0, 0, approach_offset])
    return tf.make_T(Rz @ R_down, t)
