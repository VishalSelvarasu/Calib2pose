import argparse
import json
import os

import cv2
import numpy as np

import drill_scene as ds


def look_at_camera(eye, target, roll_deg=0.0):
    """Camera pose (OpenCV convention: +Z forward, +Y down) looking at target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(z @ ref) > 0.95:                 # nearly vertical, pick another ref
        ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    if roll_deg:
        c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return ds.make_T(R, eye)


def make_flange_poses(n, T_flange_cam, seed=0):

    rng = np.random.default_rng(seed)
    inv_fc = np.linalg.inv(T_flange_cam)
    target = ds.DRILL_POS + np.array([0.0, 0.0, 0.02])
    poses = []
    for _ in range(n):
        az = rng.uniform(0, 2 * np.pi)
        el = np.radians(rng.uniform(40, 78))       # above the horizon
        r = rng.uniform(0.42, 0.62)
        eye = target + np.array([r * np.cos(el) * np.cos(az),
                                 r * np.cos(el) * np.sin(az),
                                 r * np.sin(el)])
        T_base_cam = look_at_camera(eye, target, rng.uniform(-25, 25))
        poses.append(T_base_cam @ inv_fc)
    return poses


def add_metric(model_pts, T_est, T_true):

    a = (T_est[:3, :3] @ model_pts.T).T + T_est[:3, 3]
    b = (T_true[:3, :3] @ model_pts.T).T + T_true[:3, 3]
    return float(np.linalg.norm(a - b, axis=1).mean())


def pose_error(T_est, T_true):
    t = float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))
    c = (np.trace(T_true[:3, :3].T @ T_est[:3, :3]) - 1) / 2
    return t, float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--handeye", choices=["true", "estimated"], default="true")
    ap.add_argument("--handeye-json",
                    default="../stage2_handeye/results/handeye_good.json")
    ap.add_argument("--min-markers", type=int, default=2)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scene = ds.DrillScene()
    rng = np.random.default_rng(args.seed + 7)
    T_fc_true = ds.gt_T_flange_cam()

    T_fc = T_fc_true
    if args.handeye == "estimated":
        if not os.path.exists(args.handeye_json):
            raise SystemExit(
                f"Run stage 2 first: {args.handeye_json} not found.")
        with open(args.handeye_json) as f:
            he = json.load(f)
        best = min(he["methods"], key=lambda m: m["t_err_mm"])
        T_fc = np.array(best["T_est"])
        print(f"using stage 2 hand-eye ({best['method']}): "
              f"{best['t_err_mm']:.3f} mm / {best['R_err_deg']:.3f} deg error")

    diameter = scene.diameter
    thresh = 0.1 * diameter
    print(f"drill diameter    : {diameter*1000:.1f} mm")
    print(f"ADD-0.1d threshold: {thresh*1000:.1f} mm")
    print(f"hand-eye source   : {args.handeye}")
    print(f"corner noise      : {args.noise_px} px\n")

    rows, skipped = [], 0
    for T_base_flange in make_flange_poses(args.n, T_fc_true, args.seed):
        scene.set_flange(T_base_flange)
        gray = cv2.cvtColor(scene.render(), cv2.COLOR_RGB2GRAY)
        obj, img, seen = scene.detect(gray, args.noise_px, rng)
        if obj is None or len(seen) < args.min_markers:
            skipped += 1
            continue

        ok, rvec, tvec = cv2.solvePnP(obj, img, scene.K, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            skipped += 1
            continue
        # Refine: solvePnP's linear init leaves ~0.2 px on the table.
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, scene.K, None, rvec, tvec)

        T_cam_drill = ds.make_T(cv2.Rodrigues(rvec)[0], tvec)
        T_est = scene.flange_pose() @ T_fc @ T_cam_drill
        T_true = scene.true_T_base_drill()

        proj, _ = cv2.projectPoints(obj, rvec, tvec, scene.K, None)
        reproj = float(np.linalg.norm(
            proj.reshape(-1, 2) - img, axis=1).mean())

        t_err, R_err = pose_error(T_est, T_true)
        rows.append({"n_markers": len(seen), "markers": seen,
                     "reproj_px": reproj, "t_err_mm": t_err * 1000,
                     "R_err_deg": R_err,
                     "add_mm": add_metric(scene.model_points, T_est, T_true) * 1000})

    if not rows:
        raise SystemExit("No usable views.")

    add = np.array([r["add_mm"] for r in rows])
    te = np.array([r["t_err_mm"] for r in rows])
    re = np.array([r["R_err_deg"] for r in rows])
    rp = np.array([r["reproj_px"] for r in rows])
    nm = np.array([r["n_markers"] for r in rows])
    passed = float((add < thresh * 1000).mean() * 100)

    print(f"views used     : {len(rows)} / {args.n}  ({skipped} skipped)")
    print(f"markers/view   : mean {nm.mean():.2f}  "
          f"(2 markers: {(nm == 2).sum()}, 3 markers: {(nm == 3).sum()})")
    print(f"reprojection   : mean {rp.mean():.3f} px")
    print()
    print(f"{'metric':<18}{'mean':>10}{'median':>10}{'worst':>10}")
    print(f"{'ADD':<18}{add.mean():>8.2f}mm{np.median(add):>8.2f}mm{add.max():>8.2f}mm")
    print(f"{'translation':<18}{te.mean():>8.2f}mm{np.median(te):>8.2f}mm{te.max():>8.2f}mm")
    print(f"{'rotation':<18}{re.mean():>8.3f}d{np.median(re):>8.3f}d{re.max():>8.3f}d")
    print(
        f"\nADD-0.1d pass rate: {passed:.1f}%   ({(add < thresh*1000).sum()}/{len(rows)})")

    if nm.min() < 3:
        m2 = add[nm == 2]
        m3 = add[nm == 3]
        if len(m2) and len(m3):
            print(f"\nADD by marker count: 2 markers {m2.mean():.2f} mm   "
                  f"3 markers {m3.mean():.2f} mm")

    out = {"handeye_source": args.handeye, "noise_px": args.noise_px,
           "n_views": len(rows), "diameter_mm": diameter * 1000,
           "add_threshold_mm": thresh * 1000,
           "add_mean_mm": float(add.mean()), "add_median_mm": float(np.median(add)),
           "add_max_mm": float(add.max()), "pass_rate_pct": passed,
           "t_err_mean_mm": float(te.mean()), "R_err_mean_deg": float(re.mean()),
           "reproj_mean_px": float(rp.mean()), "views": rows}
    tag = args.handeye
    if args.handeye == "estimated":
        tag += "-" + \
            os.path.basename(args.handeye_json).replace(
                "handeye_", "").replace(".json", "")
    path = os.path.join(args.out, f"pose_{tag}_{args.noise_px}px.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
