# Stage 2 — Hand-Eye Calibration, Validated Against Ground Truth

Solves for the camera's pose in the robot flange frame (`T_flange_camera`) —
the transform you cannot measure with a ruler and cannot verify on real
hardware.

Here the camera is mounted in simulation at a transform we choose, so the
solver's output is a real error in millimetres and degrees, not a residual.

## Result

18 commanded flange poses, 16 with a usable board view, no detection noise:

| method | translation error | rotation error |
|---|---|---|
| PARK | 0.593 mm | **0.076°** |
| TSAI | 0.595 mm | **0.076°** |
| ANDREFF | **0.591 mm** | 0.077° |

The noise-free figure is renderer-dependent — it measures sub-pixel
rasterization detail as much as solver quality, and only demonstrates that the
solver is correct. With 0.3 px corner noise, roughly what a real detector gives,
the best method lands at about 6 mm / 0.8°. That is the honest headline number.

## Solvers are implemented here, not called

OpenCV 5.0 removed `cv2.calibrateHandEye` from the Python bindings — the
`CALIB_HAND_EYE_*` constants survive, the function does not. Rather than pin the
project to OpenCV 4, `handeye_solvers.py` implements Tsai-Lenz (1989),
Park & Martin (1994), and Andreff (1999) directly in numpy.

They were validated against OpenCV 4.13's implementation during development:
identical to machine precision on noise-free data, agreeing to 0.04 mm under
noise, and reproducing the degenerate failure identically.

## The degenerate case

Hand-eye requires the relative motions between poses to rotate about at least
two different axes. Rotate about only one and the translation along that axis
is mathematically unobservable. The solver does not warn you.

`--degenerate` uses yaw-only motion. Result:

| axis | error |
|---|---|
| x | 0.066 mm |
| y | 0.602 mm |
| **z** | **81.000 mm** |
| rotation | **0.053°** |

The rotation looks excellent. x and y look excellent. z is returned as exactly
0.000 against a true 81 mm — the solver silently substituted the null answer for
the unobservable component.

Why exactly 81.000 mm: the translation step solves `(R_A - I) t_X = R_X t_B - t_A`.
When every relative rotation shares an axis `n`, the matrix `(R_A - I)` annihilates
`n` for every pair, so the component of `t_X` along `n` is unconstrained. Least
squares returns the minimum-norm solution, which sets it to zero — and the true
value was 81 mm.

Method choice also collapses under degeneracy: PARK ends up 180° wrong, TSAI 128°
wrong. Under good motion all three agree to within 0.005 mm.

The script computes the minimum angle between relative-motion rotation axes
*before* solving — 90.0° for the good set, 0.0° for the degenerate one. That is
the check to run, not the residual.

## The convention that breaks everyone

MuJoCo cameras look down **−Z** with **+Y up** (OpenGL). OpenCV cameras look
down **+Z** with **+Y down**. They differ by `diag(1, -1, -1)`.

Get this wrong and hand-eye still converges, returning a plausible rotation with
a wrong translation. The check used here: recover the board pose in the base
frame from several different camera poses. It must come out constant, and it
does — that constancy is what proves the conversion.

On naming: the classical literature writes `gripper2base` for the transform that
maps gripper-frame points into base coordinates — which is the flange pose
expressed in the base frame, not a transform "to" the gripper. The solvers here
take explicit `T_base_flange` and `T_cam_board` lists to remove the ambiguity.

## Usage

```bash
pip install -r requirements.txt
python 01_handeye.py --n 18
python 01_handeye.py --n 18 --degenerate
python 01_handeye.py --n 18 --noise-px 0.3
```

## Note on the robot

The flange poses are commanded directly rather than produced by an arm's
forward kinematics. The solvers cannot tell the difference — they receive a list
of flange poses and a list of board-in-camera poses either way. Swapping in a
UR5e with inverse kinematics changes where the pose list comes from and nothing
else.

`board_config.py` is a copy of the stage 1 file; keep the two in sync.