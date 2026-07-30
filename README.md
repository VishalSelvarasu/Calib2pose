# calib2pose

Camera calibration → hand-eye calibration → 6D object pose estimation for a
robot arm. Every stage validated against known ground truth.

| Stage | Status |
|---|---|
| 1. Camera intrinsics (ChArUco) | synthetic validation done, real capture pending |
| 2. Hand-eye calibration | not started |
| 3. PnP 6D pose | not started |
| 4. Learned keypoints (synthetic training) | not started |
| 5. Closed loop with UR5e | not started |

## Why ground truth

On real hardware the true intrinsics are unknown, so a low reprojection error
is a residual, not an error. Each stage therefore runs first in simulation
where the answer is known.

Stage 1 demonstrates why this matters: a deliberately bad pose set produces a
*lower* reprojection error (0.209 px vs 0.249 px) while getting fx wrong by
1.95 % and estimating k2 at +0.627 against a true value of −0.226 — wrong sign,
triple the magnitude. Selecting on reprojection error alone ships the broken
calibration.

See [`stage1_intrinsics/`](stage1_intrinsics/) for details.