import argparse
import glob
import json
import os

import cv2
import numpy as np

import board_config as bc

RESULTS = "results"


def collect(paths, board, detector, min_corners):
    obj_pts, img_pts, used, rejected = [], [], [], []
    size = None
    all_corners = []

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            rejected.append((p, "unreadable"))
            continue
        if size is None:
            size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != size:

            rejected.append(
                (p, f"size {img.shape[1]}x{img.shape[0]} != {size}"))
            continue

        ch_c, ch_id, _, _ = bc.detect(detector, img)
        if ch_id is None or len(ch_id) < min_corners:
            rejected.append(
                (p, f"only {0 if ch_id is None else len(ch_id)} corners"))
            continue

        o, i = board.matchImagePoints(ch_c, ch_id)
        if o is None or len(o) < min_corners:
            rejected.append((p, "matchImagePoints failed"))
            continue

        obj_pts.append(o)
        img_pts.append(i)
        used.append(p)
        all_corners.append(ch_c.reshape(-1, 2))

    return obj_pts, img_pts, used, rejected, size, all_corners


def run_calibration(obj_pts, img_pts, size, fix_k3):
    flags = cv2.CALIB_FIX_ASPECT_RATIO * 0  # keep fx, fy independent
    if fix_k3:
        # k3 is poorly conditioned on a short-focal webcam with a modest field
        # of view. Leaving it free lets it soak up noise and trade against k1.
        flags |= cv2.CALIB_FIX_K3

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, size, None, None, flags=flags, criteria=crit
    )
    return rms, K, dist, rvecs, tvecs


def per_view_errors(obj_pts, img_pts, rvecs, tvecs, K, dist):
    errs = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        e = np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1)
        errs.append(float(e.mean()))
    return np.array(errs)


def split_half_check(obj_pts, img_pts, size, fix_k3):
    """Independent fits on even and odd views. Disagreement exposes an
    underdetermined solution that RMS alone will happily hide."""
    if len(obj_pts) < 8:
        return None
    ea, eb = list(range(0, len(obj_pts), 2)), list(range(1, len(obj_pts), 2))
    out = []
    for idx in (ea, eb):
        _, K, _, _, _ = run_calibration(
            [obj_pts[i] for i in idx], [img_pts[i] for i in idx], size, fix_k3
        )
        out.append(K)
    Ka, Kb = out
    return {
        "fx": [float(Ka[0, 0]), float(Kb[0, 0])],
        "fy": [float(Ka[1, 1]), float(Kb[1, 1])],
        "cx": [float(Ka[0, 2]), float(Kb[0, 2])],
        "cy": [float(Ka[1, 2]), float(Kb[1, 2])],
        "fx_pct_diff": float(abs(Ka[0, 0] - Kb[0, 0]) / Ka[0, 0] * 100),
        "cx_px_diff": float(abs(Ka[0, 2] - Kb[0, 2])),
    }


def coverage_image(all_corners, size, path):
    w, h = size
    canvas = np.full((h, w, 3), 25, np.uint8)
    for c in all_corners:
        for x, y in c:
            cv2.circle(canvas, (int(x), int(y)), 3, (70, 220, 120), -1)
    for i in range(1, 3):
        cv2.line(canvas, (int(i * w / 3), 0),
                 (int(i * w / 3), h), (70, 70, 70), 1)
        cv2.line(canvas, (0, int(i * h / 3)),
                 (w, int(i * h / 3)), (70, 70, 70), 1)

    pts = np.vstack(all_corners)
    empty = []
    for gy in range(3):
        for gx in range(3):
            m = ((pts[:, 0] >= gx * w / 3) & (pts[:, 0] < (gx + 1) * w / 3) &
                 (pts[:, 1] >= gy * h / 3) & (pts[:, 1] < (gy + 1) * h / 3))
            n = int(m.sum())
            if n < 20:
                empty.append((gx, gy, n))
            cv2.putText(canvas, str(n), (int(gx * w / 3) + 10, int((gy + 1) * h / 3) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (90, 90, 240) if n < 20 else (200, 200, 200), 2, cv2.LINE_AA)
    cv2.imwrite(path, canvas)
    return empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=["captures"], nargs="+",
                    help="one or more capture directories")
    ap.add_argument("--square-mm", type=float, default=None,
                    help="measured printed square size; overrides board_config")
    ap.add_argument("--min-corners", type=int, default=12)
    ap.add_argument("--drop-worst", type=float, default=None,
                    help="refit after dropping views with mean error > this (px)")
    ap.add_argument("--k3", action="store_true",
                    help="let k3 float (default: fixed at 0, correct for webcams)")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    square_m = (args.square_mm /
                1000.0) if args.square_mm else bc.SQUARE_LENGTH_M
    board = bc.get_board(square_m)
    detector = bc.get_detector(board)

    paths = sorted(p for d in args.images
                   for pat in ("*.png", "*.jpg")
                   for p in glob.glob(os.path.join(d, pat)))
    if not paths:
        raise SystemExit(
            f"No images in {args.images}/. Run 02_capture.py first.")

    obj_pts, img_pts, used, rejected, size, all_corners = collect(
        paths, board, detector, args.min_corners
    )
    print(f"{len(used)} usable / {len(paths)} images   image size {size}")
    for p, why in rejected:
        print(f"  rejected {os.path.basename(p)}: {why}")
    if len(used) < 6:
        raise SystemExit("Need at least 6 usable views. Capture more.")

    fix_k3 = not args.k3
    rms, K, dist, rvecs, tvecs = run_calibration(
        obj_pts, img_pts, size, fix_k3)
    errs = per_view_errors(obj_pts, img_pts, rvecs, tvecs, K, dist)

    if args.drop_worst is not None:
        keep = [i for i in range(len(errs)) if errs[i] <= args.drop_worst]
        dropped = [used[i]
                   for i in range(len(errs)) if errs[i] > args.drop_worst]
        if dropped and len(keep) >= 6:
            print(
                f"\ndropping {len(dropped)} views over {args.drop_worst} px:")
            for d in dropped:
                print(f"  {os.path.basename(d)}")
            obj_pts = [obj_pts[i] for i in keep]
            img_pts = [img_pts[i] for i in keep]
            all_corners = [all_corners[i] for i in keep]
            used = [used[i] for i in keep]
            rms, K, dist, rvecs, tvecs = run_calibration(
                obj_pts, img_pts, size, fix_k3)
            errs = per_view_errors(obj_pts, img_pts, rvecs, tvecs, K, dist)

    fovx, fovy, focal_mm, pp, ar = cv2.calibrationMatrixValues(
        K, size, 0.0, 0.0)
    split = split_half_check(obj_pts, img_pts, size, fix_k3)
    empty_zones = coverage_image(
        all_corners, size, os.path.join(RESULTS, "coverage.png"))

    # visual proof, straight lines should end up straight
    sample = cv2.imread(used[len(used) // 2])
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, size, 1, size)
    undist = cv2.undistort(sample, K, dist, None, newK)
    cv2.imwrite(os.path.join(RESULTS, "undistorted.png"),
                np.hstack([sample, undist]))

    print("\n--- intrinsics ---------------------------------------------")
    print(f"fx {K[0, 0]:9.3f}   fy {K[1, 1]:9.3f}")
    print(f"cx {K[0, 2]:9.3f}   cy {K[1, 2]:9.3f}   (image centre "
          f"{size[0]/2:.1f}, {size[1]/2:.1f})")
    print(
        f"dist {np.array2string(dist.ravel(), precision=5, suppress_small=True)}")
    print(
        f"FOV  {fovx:.1f} x {fovy:.1f} deg    aspect fy/fx {K[1, 1]/K[0, 0]:.4f}")

    print("\n--- errors -------------------------------------------------")
    print(f"RMS reprojection : {rms:.4f} px   over {len(used)} views")
    print(f"per-view mean    : {errs.mean():.4f} px   worst {errs.max():.4f} px "
          f"({os.path.basename(used[int(errs.argmax())])})")

    print("\n--- diagnostics --------------------------------------------")
    if split:
        print(f"split-half fx    : {split['fx'][0]:.1f} vs {split['fx'][1]:.1f}  "
              f"({split['fx_pct_diff']:.2f}% apart)")
        print(f"split-half cx    : {split['cx'][0]:.1f} vs {split['cx'][1]:.1f}  "
              f"({split['cx_px_diff']:.1f} px apart)")
        if split["fx_pct_diff"] > 1.0:
            print("  >> fx unstable. Your views are too similar, or autofocus")
            print("     moved between shots. Add steeper tilts and re-shoot.")
        else:
            print("  >> fx stable. The solution is genuinely constrained.")
    else:
        print("split-half       : skipped (need >= 8 views)")

    if empty_zones:
        print(
            f"sparse zones     : {[(gx, gy, n) for gx, gy, n in empty_zones]}")
        print("  >> distortion is extrapolating there. See results/coverage.png.")
    else:
        print("coverage         : all 9 zones populated")

    if abs(K[1, 1] / K[0, 0] - 1.0) > 0.05:
        print("aspect           : fy/fx off by >5%. Suspicious on a webcam with")
        print("                   square pixels -- usually a sign of weak coverage.")

    out = {
        "image_size": list(size),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "rms_reprojection_px": float(rms),
        "per_view_error_px": errs.tolist(),
        "fov_deg": [float(fovx), float(fovy)],
        "n_views": len(used),
        "images_used": [os.path.basename(p) for p in used],
        "split_half": split,
        "sparse_zones": [[int(a), int(b), int(c)] for a, b, c in empty_zones],
        "k3_fixed": bool(fix_k3),
        "square_length_mm_used": square_m * 1000.0,
        "opencv_version": cv2.__version__,
    }
    with open(os.path.join(RESULTS, "intrinsics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS}/intrinsics.json, coverage.png, undistorted.png")


if __name__ == "__main__":
    main()
