# Stage 1 — Camera Intrinsic Calibration, Validated Against Ground Truth

ChArUco-based intrinsic calibration for a consumer laptop webcam, with a
synthetic ground-truth harness that makes the calibration error *measurable*
rather than merely plausible.

Stage 1 of a 5-stage 6D pose estimation project
(intrinsics → hand-eye → PnP pose → learned keypoints → closed loop).

## The point

On real hardware the true intrinsics are unknown, so a low reprojection error
is unfalsifiable — it is a residual, not an error. This repo therefore
calibrates twice:

1. **Synthetic**, through a known `K` and known distortion, with webcam-like
   blur, sensor noise and MJPG compression. Ground truth is known, so the
   output is a real error in pixels.
2. **Real**, on a printed board and a laptop webcam, using the identical
   detect-and-solve path. *(Pending — the synthetic result below stands on its
   own; the real capture adds hardware validation, not accuracy.)*

## Result: reprojection error is a misleading metric

Two runs, 30 synthetic views each, identical renderer and solver. Only the
distribution of board poses differs.

| | good pose spread | frontal-only |
|---|---|---|
| RMS reprojection | 0.2494 px | **0.2089 px** |
| fx error vs truth | **0.10 %** | 1.95 % |
| fy error vs truth | 0.01 % | 1.89 % |
| cy error vs truth | 0.15 % | 1.75 % |
| k1 (truth +0.115) | +0.1146 | +0.0919 |
| k2 (truth −0.226) | −0.2147 | **+0.6267** |
| split-half fx agreement | 0.18 % | 3.12 % |

The frontal-only set has the *lower* RMS and is comprehensively wrong — k2 does
not merely drift, it changes sign. Anyone selecting on reprojection error alone
would ship the worse calibration.

The split-half check catches it: fit the even-indexed views and the
odd-indexed views independently and compare. Stable intrinsics agree to well
under 1 %. Underdetermined ones do not.

Reproduce both:

```bash
python 00_synthetic_test.py --n 30
python 00_synthetic_test.py --n 30 --frontal-only
```

## What the calibration actually does

![undistorted](results/undistorted.png)

Left: a rendered view with barrel distortion applied through the known
coefficients. Right: the same view undistorted using the coefficients the
solver recovered. Straight lines come out straight — the visual counterpart of
the 0.10 % fx error in the table.

![corner coverage](results/coverage.png)

Where the detected ChArUco corners landed across all 30 views, with a count per
image zone. Distortion is only observable where corners fall, so a sparse zone
means the distortion model is extrapolating there rather than fitting. This plot
is what the capture HUD is trying to fill in during real capture, and it is why
the frontal-only run above fails: all its corners land in the middle.

## Webcam-specific handling

Laptop webcams break three assumptions that machine-vision cameras satisfy:

**Autofocus.** If the lens refocuses between frames, the focal length changes
and the single `K` being solved for does not exist. `02_capture.py` locks
autofocus, auto-exposure and auto white balance, then *reads every property
back* — `cap.set()` returns `True` even when the driver ignored it. When the
lock fails it says so and tells you to hold a constant working distance.

**Motion blur.** Frames are rejected on sharpness (variance of Laplacian,
measured inside the board's bounding box only — whole-frame sharpness is
carried by background clutter) and on stillness (mean corner displacement
between frames, matched by corner ID).

**Pose coverage.** Distortion is only observable where corners land. The
capture HUD tracks a 3×3 image-zone grid and four tilt buckets and auto-captures
only into cells that still need samples, which is what prevents the failure
mode in the table above.

Other choices: `CAP_DSHOW` on Windows (MSMF ignores property sets), MJPG
requested before resolution (many webcams silently clamp to 640×480 under raw
YUY2), `DICT_5X5_100` for Hamming margin on a soft sensor, loosened ArUco
bit-error thresholds for MJPG ringing, and `k3` fixed at zero by default since
it is poorly conditioned at this field of view and trades against `k1`.

## OpenCV 4.x / 5.x compatibility

OpenCV 5.0 removed the vestigial middle axis from ChArUco detector output:

| | corners | ids |
|---|---|---|
| 4.x | `(N,1,2)` | `(N,1)` |
| 5.0 | `(N,2)` | `(N,)` |

Code using `.reshape(-1,2)` is unaffected. Code indexing `corners[k, 0]` is not,
and fails in the worse way: under 5.0 it returns a bare x coordinate instead of
a point, so the caller keeps running on wrong numbers rather than raising.
`board_config.detect()` normalises at the detection boundary so the rest of the
code is version-agnostic.

Verified end-to-end on OpenCV 4.13.0 and 5.0.0, identical results to within
solver noise.

## Usage

```bash
pip install -r requirements.txt

python 00_synthetic_test.py      # validate the pipeline first
python 01_generate_board.py      # -> board/charuco_A4.pdf, print at 100%
python 02_capture.py             # SPACE capture, A auto, U undo, Q quit
python 03_calibrate.py
```

After printing: verify the 100 mm ruler printed on the page, tape the board
flat to something rigid, measure a square, and set `SQUARE_LENGTH_MM` in
`board_config.py`.

Board scale does **not** affect intrinsics — it enters only the extrinsic
translation. It does affect stage 2 hand-eye, which solves for a translation in
metres, so it is worth getting right now.

## Files

| | |
|---|---|
| `board_config.py` | single source of truth for board geometry + detector tuning |
| `00_synthetic_test.py` | render from known `K`, calibrate, report error vs truth |
| `01_generate_board.py` | print-exact A4 PDF with a 100 mm verification ruler |
| `02_capture.py` | webcam capture with focus lock, blur rejection, coverage HUD |
| `03_calibrate.py` | solve, per-view errors, outlier refit, split-half check |

Outputs land in `results/`: `intrinsics.json`, `coverage.png`,
`undistorted.png`.