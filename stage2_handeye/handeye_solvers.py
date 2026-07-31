import numpy as np


def _log_so3(R):
    # Rotation matrix -> axis-angle 3-vector
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(c)
    if theta < 1e-9:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        # Near pi the skew part vanishes; recover the axis from R + I.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > 1e-9:
            axis = A[:, k] / axis[k]
        return axis / (np.linalg.norm(axis) + 1e-12) * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta)))


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _project_to_so3(M):
    # Nearest rotation matrix in the Frobenius sense
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _inv_sqrt_spd(M):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def _motion_pairs(T_base_flange, T_cam_board, min_angle_deg=5.0):

    pairs = []
    n = len(T_base_flange)
    for i in range(n):
        for j in range(i + 1, n):
            A = np.linalg.inv(T_base_flange[j]) @ T_base_flange[i]
            B = T_cam_board[j] @ np.linalg.inv(T_cam_board[i])
            ang = np.degrees(np.linalg.norm(_log_so3(A[:3, :3])))
            if ang >= min_angle_deg:
                pairs.append((A, B))
    return pairs


def _solve_translation(pairs, R_X):
    # Least squares on (R_A - I) t_X = R_X t_B - t_A
    C, d = [], []
    for A, B in pairs:
        C.append(A[:3, :3] - np.eye(3))
        d.append(R_X @ B[:3, 3] - A[:3, 3])
    C = np.vstack(C)
    d = np.hstack(d)
    t, *_ = np.linalg.lstsq(C, d, rcond=None)
    return t


def solve_park(T_base_flange, T_cam_board):
    pairs = _motion_pairs(T_base_flange, T_cam_board)
    M = np.zeros((3, 3))
    for A, B in pairs:
        a = _log_so3(A[:3, :3])
        b = _log_so3(B[:3, :3])
        M += np.outer(b, a)
    R_X = _project_to_so3(_inv_sqrt_spd(M.T @ M) @ M.T)
    return _make_T(R_X, _solve_translation(pairs, R_X))


def solve_tsai(T_base_flange, T_cam_board):
    pairs = _motion_pairs(T_base_flange, T_cam_board)
    S, v = [], []
    for A, B in pairs:
        ra = _log_so3(A[:3, :3])
        rb = _log_so3(B[:3, :3])
        ta, tb = np.linalg.norm(ra), np.linalg.norm(rb)
        if ta < 1e-9 or tb < 1e-9:
            continue
        # Modified Rodrigues vectors
        Pa = 2 * np.sin(ta / 2) * (ra / ta)
        Pb = 2 * np.sin(tb / 2) * (rb / tb)
        S.append(_skew(Pa + Pb))
        v.append(Pb - Pa)
    S, v = np.vstack(S), np.hstack(v)
    Px, *_ = np.linalg.lstsq(S, v, rcond=None)
    # Recover the full rotation from the scaled Rodrigues vector
    n2 = float(Px @ Px)
    Px_full = 2 * Px / np.sqrt(1 + n2)
    n2f = float(Px_full @ Px_full)
    R_X = ((1 - n2f / 2) * np.eye(3)
           + 0.5 * (np.outer(Px_full, Px_full)
                    + np.sqrt(max(4 - n2f, 0.0)) * _skew(Px_full)))
    R_X = _project_to_so3(R_X)
    return _make_T(R_X, _solve_translation(pairs, R_X))


def solve_andreff(T_base_flange, T_cam_board):

    pairs = _motion_pairs(T_base_flange, T_cam_board)
    rows, rhs = [], []
    for A, B in pairs:
        RA, tA = A[:3, :3], A[:3, 3]
        RB, tB = B[:3, :3], B[:3, 3]
        # vec(R_X) block: (I9 - RB^T kron RA) vec(R_X) = 0
        rows.append(np.hstack([np.eye(9) - np.kron(RB, RA), np.zeros((9, 3))]))
        rhs.append(np.zeros(9))
        # translation block: (I - RA) t_X - kron(tB^T, I) vec(R_X) = -tA
        rows.append(np.hstack([-np.kron(tB.reshape(1, 3), np.eye(3)),
                               np.eye(3) - RA]))
        rhs.append(-tA)
    Amat, bvec = np.vstack(rows), np.hstack(rhs)
    sol, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
    R_X = _project_to_so3(sol[:9].reshape(3, 3, order="F"))
    return _make_T(R_X, _solve_translation(pairs, R_X))


def _make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def motion_axis_spread(T_base_flange, min_angle_deg=5.0):

    axes = []
    n = len(T_base_flange)
    for i in range(n):
        for j in range(i + 1, n):
            A = np.linalg.inv(T_base_flange[j]) @ T_base_flange[i]
            r = _log_so3(A[:3, :3])
            if np.degrees(np.linalg.norm(r)) >= min_angle_deg:
                axes.append(r / np.linalg.norm(r))
    if len(axes) < 2:
        return 0.0
    axes = np.array(axes)
    return float(np.degrees(np.arccos(np.clip(np.abs(axes @ axes.T).min(), -1, 1))))


SOLVERS = {"PARK": solve_park, "TSAI": solve_tsai, "ANDREFF": solve_andreff}
