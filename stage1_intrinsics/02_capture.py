import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

import board_config as bc

MIN_CORNERS = 14
MIN_SHARPNESS = 60.0
STILL_PX = 1.2
STILL_FRAMES = 3
COOLDOWN_S = 1.0
GRID = 3
PER_ZONE_TARGET = 2
TILT_BINS = [(0, 12), (12, 25), (25, 40), (40, 90)]
PER_TILT_TARGET = 4


def open_camera(index, width, height):
    """Open with the backend that actually lets you set properties."""
    if platform.system() == "Windows":

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif platform.system() == "Linux":
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {index}. Try --cam 1 / --cam 2, and close "
            f"Teams/Zoom/Camera app -- Windows gives exclusive access."
        )

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def lock_camera(cap):
    """
    Disable every 'auto' the driver will let us. Returns a report dict.
    Setting a property returns True even when the driver ignored it, so we
    read every value back.
    """
    report = {}

    def try_set(prop, value, name):
        cap.set(prop, value)
        time.sleep(0.15)
        got = cap.get(prop)
        report[name] = {"requested": value, "readback": got}
        return got

    try_set(cv2.CAP_PROP_AUTOFOCUS, 0, "autofocus")
    # 0.25 is the DirectShow/V4L2 magic value for manual exposure.
    try_set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "auto_exposure")
    try_set(cv2.CAP_PROP_AUTO_WB, 0, "auto_white_balance")

    # Nail focus to infinity-ish. Ignored on fixed-focus modules, harmless.
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    report["focus"] = {"requested": 0, "readback": cap.get(cv2.CAP_PROP_FOCUS)}

    # Let AE settle before we froze it, then flush stale buffered frames.
    for _ in range(20):
        cap.read()
    return report


def sharpness(gray, corners):
    """Variance of Laplacian inside the board's bounding box only.

    Measuring the whole frame is misleading: a cluttered background can carry
    the score while the board itself is smeared.
    """
    if corners is None or len(corners) < 4:
        return 0.0
    pts = corners.reshape(-1, 2)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    h, w = gray.shape
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return 0.0
    roi = gray[y0:y1, x0:x1]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def approx_tilt_deg(obj_pts, img_pts, size):
    """
    Angle between the board normal and the camera axis, using a guessed
    pinhole (f = image width, principal point = centre, no distortion).

    Approximate by construction -- we do not have intrinsics yet, that is the
    whole point of this exercise. It is accurate enough to bucket poses into
    "frontal / moderate / steep", which is all it is used for.
    """
    w, h = size
    K = np.array([[w, 0, w / 2.0], [0, w, h / 2.0],
                 [0, 0, 1.0]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        return 0.0
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]
    cosang = abs(float(normal[2])) / (np.linalg.norm(normal) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def tilt_bin(deg):
    for i, (lo, hi) in enumerate(TILT_BINS):
        if lo <= deg < hi:
            return i
    return len(TILT_BINS) - 1


def zone_of(pts, size):
    w, h = size
    cx, cy = pts.reshape(-1, 2).mean(axis=0)
    gx = min(int(cx / w * GRID), GRID - 1)
    gy = min(int(cy / h * GRID), GRID - 1)
    return gy * GRID + gx


def draw_hud(vis, state, size, msg, msg_col):
    w, h = size
    overlay = vis.copy()

    # zone grid, tinted by how many samples each cell still needs
    for gy in range(GRID):
        for gx in range(GRID):
            z = gy * GRID + gx
            x0, y0 = int(gx * w / GRID), int(gy * h / GRID)
            x1, y1 = int((gx + 1) * w / GRID), int((gy + 1) * h / GRID)
            n = state["zone_counts"][z]
            col = (60, 160, 60) if n >= PER_ZONE_TARGET else (40, 40, 130)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), col, -1)
            cv2.putText(vis, f"{n}/{PER_ZONE_TARGET}", (x0 + 8, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1,
                        cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.13, vis, 0.87, 0, vis)
    for i in range(1, GRID):
        cv2.line(vis, (int(i * w / GRID), 0), (int(i * w / GRID), h),
                 (90, 90, 90), 1)
        cv2.line(vis, (0, int(i * h / GRID)), (w, int(i * h / GRID)),
                 (90, 90, 90), 1)

    bar_h = 96
    cv2.rectangle(vis, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.putText(vis, f"captured {state['n']}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis,
                f"corners {state['n_corners']}  sharp {state['sharp']:.0f}  "
                f"tilt {state['tilt']:.0f}deg  auto {'ON' if state['auto'] else 'OFF'}",
                (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                cv2.LINE_AA)

    tilt_txt = "  ".join(
        f"{lo}-{hi if hi < 90 else '+'}:{state['tilt_counts'][i]}/{PER_TILT_TARGET}"
        for i, (lo, hi) in enumerate(TILT_BINS)
    )
    cv2.putText(vis, f"tilt bins  {tilt_txt}", (12, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 200, 255), 1, cv2.LINE_AA)

    cv2.putText(vis, msg, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                msg_col, 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default="captures")
    ap.add_argument("--min-sharpness", type=float, default=MIN_SHARPNESS)
    ap.add_argument("--min-corners", type=int, default=MIN_CORNERS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    board = bc.get_board()
    detector = bc.get_detector(board)

    cap = open_camera(args.cam, args.width, args.height)
    lock = lock_camera(cap)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera opened but returned no frame.")
    h, w = frame.shape[:2]
    size = (w, h)

    print(bc.summary())
    print(f"resolution : requested {args.width}x{args.height}, got {w}x{h}")
    if (w, h) != (args.width, args.height):
        print("             driver clamped it. Not a problem, just be aware.")
    af = lock["autofocus"]["readback"]
    ae = lock["auto_exposure"]["readback"]
    print(f"autofocus  : readback {af}  ->  "
          f"{'locked' if af in (0, 0.0) else 'DRIVER IGNORED THE LOCK'}")
    print(f"auto-expo  : readback {ae}")
    if af not in (0, 0.0):
        print("\n  Autofocus could not be disabled through OpenCV.")
        print("  Work around it: keep the board at a CONSTANT distance from the")
        print("  camera for every shot, and give the lens a second to settle")
        print("  before each capture. Move the board sideways and tilt it, but")
        print("  do not change depth. Expect ~1 px reprojection error, not 0.2.")
    print("\nSPACE capture   A auto-capture   U undo   Q quit\n")

    state = {
        "n": 0, "n_corners": 0, "sharp": 0.0, "tilt": 0.0, "auto": True,
        "zone_counts": defaultdict(int), "tilt_counts": defaultdict(int),
    }
    saved = []
    prev_by_id = {}
    still_run = 0
    last_cap_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ch_c, ch_id, mk_c, mk_id = detector.detectBoard(gray)
        vis = frame.copy()

        n_corners = 0 if ch_id is None else len(ch_id)
        state["n_corners"] = n_corners
        msg, col = "no board detected", (60, 60, 220)
        capture_now = False

        if n_corners >= 4:
            cv2.aruco.drawDetectedCornersCharuco(
                vis, ch_c, ch_id, (0, 255, 255))

            state["sharp"] = sharpness(gray, ch_c)

            # stillness: mean motion of corners matched by id
            cur = {int(i): ch_c[k, 0] for k, i in enumerate(ch_id.flatten())}
            shared = set(cur) & set(prev_by_id)
            if shared:
                d = np.mean([np.linalg.norm(cur[i] - prev_by_id[i])
                            for i in shared])
            else:
                d = 999.0
            prev_by_id = cur
            still_run = still_run + 1 if d < STILL_PX else 0

            obj_p, img_p = board.matchImagePoints(ch_c, ch_id)
            state["tilt"] = (
                approx_tilt_deg(obj_p, img_p, size) if obj_p is not None and len(
                    obj_p) >= 6 else 0.0
            )
            z = zone_of(ch_c, size)
            tb = tilt_bin(state["tilt"])

            if n_corners < args.min_corners:
                msg, col = f"too few corners ({n_corners})", (60, 140, 220)
            elif state["sharp"] < args.min_sharpness:
                msg, col = f"too blurry ({state['sharp']:.0f})", (60, 140, 220)
            elif still_run < STILL_FRAMES:
                msg, col = "hold still...", (60, 200, 220)
            else:
                need = (state["zone_counts"][z] < PER_ZONE_TARGET
                        or state["tilt_counts"][tb] < PER_TILT_TARGET)
                if need and time.time() - last_cap_t > COOLDOWN_S:
                    msg, col = "READY", (60, 220, 60)
                    capture_now = state["auto"]
                else:
                    msg, col = "zone+tilt already covered - move on", (180,
                                                                       180, 180)
        else:
            prev_by_id, still_run = {}, 0

        vis = draw_hud(vis, state, size, msg, col)
        cv2.imshow("charuco capture", vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("a"):
            state["auto"] = not state["auto"]
        if key == ord("u") and saved:
            gone = saved.pop()
            os.remove(gone["path"])
            state["zone_counts"][gone["zone"]] -= 1
            state["tilt_counts"][gone["tilt_bin"]] -= 1
            state["n"] -= 1
            print(f"undo -> {os.path.basename(gone['path'])}")
        if key == ord(" ") and n_corners >= args.min_corners:
            capture_now = True

        if capture_now:
            z = zone_of(ch_c, size)
            tb = tilt_bin(state["tilt"])
            path = os.path.join(args.out, f"calib_{state['n']:03d}.png")
            cv2.imwrite(path, frame)  # raw frame, never the annotated one
            saved.append({"path": path, "zone": z, "tilt_bin": tb,
                          "corners": int(n_corners), "sharpness": state["sharp"],
                          "tilt_deg": state["tilt"]})
            state["zone_counts"][z] += 1
            state["tilt_counts"][tb] += 1
            state["n"] += 1
            last_cap_t = time.time()
            print(f"saved {os.path.basename(path)}  corners={n_corners}  "
                  f"sharp={state['sharp']:.0f}  tilt={state['tilt']:.0f}deg  zone={z}")

    cap.release()
    cv2.destroyAllWindows()

    meta = {
        "n_images": state["n"],
        "image_size": [w, h],
        "camera_index": args.cam,
        "camera_lock_report": {k: {kk: float(vv) for kk, vv in v.items()}
                               for k, v in lock.items()},
        "board": {
            "squares_x": bc.SQUARES_X, "squares_y": bc.SQUARES_Y,
            "square_length_mm": bc.SQUARE_LENGTH_MM,
            "marker_length_mm": bc.MARKER_LENGTH_MM,
            "dictionary": "DICT_5X5_100",
        },
        "captures": saved,
    }
    with open(os.path.join(args.out, "session.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{state['n']} images -> {args.out}/")
    missing_z = [z for z in range(GRID * GRID)
                 if state["zone_counts"][z] < PER_ZONE_TARGET]
    missing_t = [i for i in range(len(TILT_BINS))
                 if state["tilt_counts"][i] < PER_TILT_TARGET]
    if missing_z:
        print(f"zones still short: {missing_z} (0=top-left, 8=bottom-right)")
    if missing_t:
        print(f"tilt bins still short: {[TILT_BINS[i] for i in missing_t]}")
    if state["n"] < 15:
        print("Under 15 images. Run again and append -- 20-30 is the target.")


if __name__ == "__main__":
    sys.exit(main())
