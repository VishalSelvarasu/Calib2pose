import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

import board_config as bc

OUT = "synthetic"


def render_view(board_img, board_size_m, K, dist, rvec, tvec, img_size):
    """Warp the flat board into the image via its four physical corners,
    then apply radial/tangential distortion as a dense remap."""
    W, H = img_size
    bw_m, bh_m = board_size_m
    ph, pw = board_img.shape[:2]

    obj = np.array([[0, 0, 0], [bw_m, 0, 0], [bw_m, bh_m, 0], [0, bh_m, 0]],
                   dtype=np.float64)

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    proj = proj.reshape(-1, 2).astype(np.float32)

    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]],
                   dtype=np.float32)
    Hm = cv2.getPerspectiveTransform(src, proj)
    ideal = cv2.warpPerspective(board_img, Hm, (W, H),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=255)

    return cv2.remap(ideal, *_forward_distort_map(K, dist, W, H),
                     cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


_MAP_CACHE = {}


def _forward_distort_map(K, dist, W, H):
    key = (K.tobytes(), dist.tobytes(), W, H)
    if key not in _MAP_CACHE:
        u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32))
        grid = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
        src = cv2.undistortPoints(grid, K, dist, P=K).reshape(H, W, 2)
        _MAP_CACHE[key] = (np.ascontiguousarray(src[:, :, 0]),
                           np.ascontiguousarray(src[:, :, 1]))
    return _MAP_CACHE[key]


def make_poses(n, board_size_m, frontal_only):
    bw, bh = board_size_m
    rng = np.random.default_rng(0)
    poses = []
    for i in range(n):
        if frontal_only:
            # The bad-practice set: centred, flat, tiny angles.
            ax = np.radians(rng.uniform(-6, 6))
            ay = np.radians(rng.uniform(-6, 6))
            tx = rng.uniform(-0.02, 0.02)
            ty = rng.uniform(-0.02, 0.02)
            tz = rng.uniform(0.55, 0.65)
        else:
            ax = np.radians(rng.uniform(-38, 38))
            ay = np.radians(rng.uniform(-38, 38))
            tx = rng.uniform(-0.13, 0.13)
            ty = rng.uniform(-0.10, 0.10)
            tz = rng.uniform(0.42, 0.80)
        az = np.radians(rng.uniform(-25, 25))

        Rx = cv2.Rodrigues(np.array([ax, 0, 0]))[0]
        Ry = cv2.Rodrigues(np.array([0, ay, 0]))[0]
        Rz = cv2.Rodrigues(np.array([0, 0, az]))[0]
        R = Rz @ Ry @ Rx
        rvec = cv2.Rodrigues(R)[0]
        # Rotate about the board centre, then place it.
        c = np.array([bw / 2, bh / 2, 0.0])
        tvec = (np.array([tx, ty, tz]) - R @ c).reshape(3, 1)
        poses.append((rvec, tvec))
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--blur", type=float, default=0.8,
                    help="gaussian sigma, mimics soft webcam optics")
    ap.add_argument("--noise", type=float, default=2.0,
                    help="sensor noise sigma")
    ap.add_argument("--jpeg", type=int, default=88,
                    help="MJPG-like compression")
    ap.add_argument("--frontal-only", action="store_true",
                    help="generate the BAD pose set, to see the failure mode")
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    W, H = args.width, args.height

    # --- ground truth, chosen to look like a real 720p laptop webcam ---------
    K_true = np.array([[905.0, 0.0, 646.2],
                       [0.0, 903.5, 357.8],
                       [0.0, 0.0, 1.0]])
    dist_true = np.array([0.115, -0.226, 0.0012, -0.0008, 0.0])

    board = bc.get_board()
    board_img = board.generateImage((1400, 2000), marginSize=30)
    board_size_m = (bc.SQUARES_X * bc.SQUARE_LENGTH_M,
                    bc.SQUARES_Y * bc.SQUARE_LENGTH_M)

    poses = make_poses(args.n, board_size_m, args.frontal_only)
    rng = np.random.default_rng(1)
    kept = 0

    for i, (rvec, tvec) in enumerate(poses):
        img = render_view(board_img, board_size_m, K_true, dist_true,
                          rvec, tvec, (W, H))
        if args.blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), args.blur)
        if args.noise > 0:
            img = np.clip(img.astype(np.float32) +
                          rng.normal(0, args.noise, img.shape), 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", img,
                               [cv2.IMWRITE_JPEG_QUALITY, args.jpeg])
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(OUT, f"calib_{i:03d}.png"), img)
        kept += 1

    print(bc.summary())
    print(f"rendered {kept} views -> {OUT}/  ({W}x{H}, blur {args.blur}, "
          f"noise {args.noise}, jpeg {args.jpeg}"
          f"{', FRONTAL-ONLY' if args.frontal_only else ''})")
    print("\nground truth")
    print(f"  fx {K_true[0, 0]:.3f}  fy {K_true[1, 1]:.3f}  "
          f"cx {K_true[0, 2]:.3f}  cy {K_true[1, 2]:.3f}")
    print(f"  dist {dist_true}")

    np.savez(os.path.join(OUT, "ground_truth.npz"), K=K_true, dist=dist_true)

    if args.no_calibrate:
        return

    print("\n" + "=" * 62)
    subprocess.run([sys.executable, "03_calibrate.py",
                   "--images", OUT], check=True)

    import json
    with open("results/intrinsics.json") as f:
        res = json.load(f)
    K = np.array(res["camera_matrix"])
    d = np.array(res["dist_coeffs"])

    print("\n--- ERROR VS GROUND TRUTH ----------------------------------")
    for name, est, true in [("fx", K[0, 0], K_true[0, 0]),
                            ("fy", K[1, 1], K_true[1, 1]),
                            ("cx", K[0, 2], K_true[0, 2]),
                            ("cy", K[1, 2], K_true[1, 2])]:
        e = est - true
        print(f"  {name}  est {est:9.3f}   true {true:9.3f}   "
              f"err {e:+8.3f} px  ({abs(e)/true*100:5.2f}%)")
    for j, name in enumerate(["k1", "k2", "p1", "p2"]):
        if j < len(d):
            print(f"  {name}  est {d[j]:+9.5f}   true {dist_true[j]:+9.5f}   "
                  f"err {d[j]-dist_true[j]:+9.5f}")
    print("\nThis error, not the RMS, is what you put in the README.")


if __name__ == "__main__":
    main()
