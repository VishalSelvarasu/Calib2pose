import argparse
import json
import os

import cv2
import numpy as np

import board_config as bc
import sim_scene as ss
import handeye_solvers as hs


def make_flange_poses(n, degenerate, seed=0):
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        if degenerate:

            rpy = [0.0, 0.0, rng.uniform(-50, 50)]
        else:
            rpy = [rng.uniform(-22, 22), rng.uniform(-22, 22),
                   rng.uniform(-60, 60)]
        t = [rng.uniform(-0.07, 0.07), rng.uniform(-0.07, 0.07),
             rng.uniform(0.40, 0.60)]
        poses.append(ss.make_T(ss.rpy_to_R(np.radians(rpy)), t))
    return poses


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--degenerate", action="store_true",
                    help="yaw-only motion: shows the unobservable-axis failure")
    ap.add_argument("--noise-px", type=float, default=0.0,
                    help="corner noise sigma, mimics real detection error")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    board_w = bc.SQUARES_X * bc.SQUARE_LENGTH_M
    board_h = bc.SQUARES_Y * bc.SQUARE_LENGTH_M
    if not os.path.exists("board_tex.png"):
        cv2.imwrite("board_tex.png",
                    bc.get_board().generateImage((1400, 2000), marginSize=0))

    scene = ss.Scene("board_tex.png", board_w, board_h)
    board = bc.get_board()
    detector = bc.get_detector(board)
    rng = np.random.default_rng(args.seed + 99)

    flange_poses, board_poses, used = [], [], 0

    for T_base_flange in make_flange_poses(args.n, args.degenerate, args.seed):
        scene.set_flange(T_base_flange)
        gray = cv2.cvtColor(scene.render(), cv2.COLOR_RGB2GRAY)
        ch_c, ch_id, _, _ = bc.detect(detector, gray)
        if ch_id is None or len(ch_id) < 12:
            continue

        obj_p, img_p = board.matchImagePoints(ch_c, ch_id)
        if args.noise_px > 0:
            img_p = img_p + \
                rng.normal(0, args.noise_px, img_p.shape).astype(np.float32)

        ok, rvec, tvec = cv2.solvePnP(obj_p, img_p, scene.K, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue

        flange_poses.append(scene.flange_pose())
        board_poses.append(ss.make_T(cv2.Rodrigues(rvec)[0], tvec))
        used += 1

    if used < 5:
        raise SystemExit(f"Only {used} usable views. Hand-eye needs >= 5.")

    T_true = ss.gt_T_flange_cam()
    spread = hs.motion_axis_spread(flange_poses)
    sv, cond, null = hs.motion_conditioning(flange_poses)

    print(f"views used        : {used} / {args.n}")
    print(f"motion axis spread: {spread:.1f} deg   "
          f"({'DEGENERATE' if spread < 10 else 'ok'})")
    print(f"corner noise      : {args.noise_px} px")
    print("\nground truth  T_flange_camera")
    print(f"  t = {np.round(T_true[:3, 3] * 1000, 3)} mm")
    print(f"singular values   : {sv[0]:.3f}  {sv[1]:.3f}  {sv[2]:.3e}")
    print(f"condition number  : {cond:.3e}")
    print(
        f"weakest direction : [{null[0]:+.3f} {null[1]:+.3f} {null[2]:+.3f}]")

    rows, best = [], None
    print("\n--- recovered camera pose in flange frame -------------------")
    print(f"{'method':<12}{'t error':>10}{'R error':>10}   t estimate (mm)")
    for name, solver in hs.SOLVERS.items():
        try:
            T_est = solver(flange_poses, board_poses)
        except Exception as e:
            print(f"{name:<12}  failed: {type(e).__name__}: {e}")
            continue
        t_err = float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1000)
        R_err = rot_angle_deg(T_true[:3, :3].T @ T_est[:3, :3])
        rows.append({"method": name, "t_err_mm": t_err, "R_err_deg": R_err,
                     "T_est": T_est.tolist()})
        print(f"{name:<12}{t_err:>8.3f}mm{R_err:>8.3f}deg   "
              f"{np.round(T_est[:3, 3] * 1000, 2)}")
        if best is None or t_err < best["t_err_mm"]:
            best = rows[-1]

    print(f"\nbest: {best['method']}  {best['t_err_mm']:.3f} mm  "
          f"{best['R_err_deg']:.3f} deg")

    if spread < 10:
        axis_err = np.abs(np.array(best["T_est"])[
                          :3, 3] - T_true[:3, 3]) * 1000
        print("\n  >> Motion was degenerate. Per-axis translation error (mm):")
        print(
            f"     x {axis_err[0]:.3f}   y {axis_err[1]:.3f}   z {axis_err[2]:.3f}")
        print("     The rotation axis was Z throughout, so the Z offset is")
        print("     unobservable. The solver reported no error of any kind.")

    out = {
        "n_views": used, "motion_axis_spread_deg": spread,
        "singular_values": sv.tolist(),
        "condition_number": float(cond) if np.isfinite(cond) else None,
        "weakest_direction": null.tolist(),
        "degenerate": bool(args.degenerate), "noise_px": args.noise_px,
        "T_flange_cam_true": T_true.tolist(), "methods": rows,
        "best_method": best["method"],
        "best_t_err_mm": best["t_err_mm"], "best_R_err_deg": best["R_err_deg"],
    }
    tag = "degenerate" if args.degenerate else "good"
    path = os.path.join(args.out, f"handeye_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
