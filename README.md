# calib2pose

Camera calibration → hand-eye calibration → 6D object pose estimation for a
robot arm. Every stage is validated against a **known** answer, not against its
own residual.

| Stage | Result | Status |
|---|---|---|
| [1. Camera intrinsics (ChArUco)](stage1_intrinsics/) | fx recovered to 0.10 % of ground truth | synthetic done, real capture pending |
| [2. Hand-eye calibration](stage2_handeye/) | 0.591 mm / 0.077° against a known mount | done |
| [3. 6D pose, ArUco markers](stage3_pose/) | 1.40 mm mean ADD, 100 % ADD-0.1d | done |
| [4. 6D pose, learned keypoints](stage4_keypoints/) | 11.21 mm mean ADD, 93.7 % ADD-0.1d | done |
| 5. Closed loop with UR5e | — | not started |

Python, OpenCV, MuJoCo, PyTorch. Runs on Windows; no ROS or Linux required.

## Why ground truth

On real hardware the true intrinsics, the true camera-to-flange transform, and
the true object pose are all unknown. A low residual is therefore not an error
measurement — it is unfalsifiable. Every stage here runs first in simulation,
where the transform was chosen and the answer is known, so the output is an
error in millimetres and degrees.

That discipline repeatedly overturned conclusions that looked solid:

**Stage 1 — the metric preferred the broken calibration.** A deliberately bad
pose set produced a *lower* reprojection error (0.209 px vs 0.249 px) while
getting fx wrong by 1.95 % and estimating k2 at +0.627 against a true −0.226 —
wrong sign, nearly triple the magnitude. Selecting on reprojection error alone
ships the worse calibration.

**Stage 2 — an 81 mm error with a 0.05° residual.** Hand-eye calibration needs
the relative motions to rotate about at least two axes. Given yaw-only motion,
the solver recovered rotation to 0.053° and x/y to a fraction of a millimetre,
and returned z as exactly 0.000 against a true 81 mm — least squares supplies the
minimum-norm solution for an unobservable component, silently.

**Stage 3 — and that error propagates invisibly.** Feeding the degenerate
transform into the pose stage gives 81.40 mm ADD at a 0 % pass rate, while every
signal available *inside* that stage stays green: reprojection 0.533 px, rotation
error 0.385°, markers detected normally. Nothing local indicates a problem. The
drill is simply 81 mm from where the robot thinks it is.

**Stage 4 — four hypotheses, two refuted.** The predicted failure mode for
bounding-box keypoints was corner-identity confusion; measurement showed identity
was 9.3 px of a 33.8 px error and localisation was 24.5 px. RANSAC, which should
have filtered the swaps, changed the pass rate by 0.1 %. Doubling heatmap
resolution changed nothing. Data augmentation and a longer schedule — the two
least interesting options — cut the error 3.1x and took the pass rate from 64 %
to 93.7 %.

## Markers vs markerless

Stage 4 replaces stage 3's ArUco markers with a learned keypoint detector,
feeding the same `solvePnP` call so the comparison isolates the perception
front end.

| | mean ADD | ADD-0.1d pass | rotation |
|---|---|---|---|
| markers (stage 3) | 1.40 mm | 100 % | 0.39° |
| learned keypoints (stage 4) | 11.21 mm | 93.7 % | 3.36° |

An 8x accuracy cost to remove the requirement that someone stick fiducials on the
object. Under clean visibility the learned pipeline reaches 6.73 mm at a 99.1 %
pass rate; heavy occlusion is where it degrades, to 78.0 %.

## Object

The YCB power drill (035), 274.0 mm diameter. Chosen because it is asymmetric —
so plain ADD applies rather than ADD-S — and because it is a benchmark-standard
object, which makes the numbers comparable to published work.

Poses are scored with **ADD**: the mean distance between the object's 8945 model
points transformed by the estimated pose versus the true pose. A pose counts as
correct at ADD < 0.1 × diameter, here 27.4 mm.

## Notes

`stage2_handeye/handeye_solvers.py` implements Tsai-Lenz, Park-Martin and
Andreff directly in numpy, because OpenCV 5.0 removed `cv2.calibrateHandEye`
from the Python bindings. They were validated against OpenCV 4.13's
implementation during development.

Each stage folder has its own README with the full method and results.
`NOTES.md` is the working log.