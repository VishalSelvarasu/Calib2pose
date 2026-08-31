# Stage 2 — Hand-Eye Calibration

This stage estimates the rigid transform between the robot flange and the camera,

```text
T_flange_cam
```

using synthetic observations generated from a known camera mounting transform.

The purpose of the simulation experiment is not to claim real-world calibration accuracy. Instead, it provides a controlled environment in which the estimated hand-eye transform can be compared directly against ground truth. This makes it possible to measure true translation and rotation error rather than relying only on reprojection error or solver residuals.

The stage also investigates an important failure mode of hand-eye calibration: **insufficient rotational diversity in the robot motion set**.

---

## Objectives

This stage addresses four main questions:

1. Can the flange-to-camera transform be recovered accurately from synthetic robot and calibration-board observations?
2. Do multiple classical hand-eye calibration methods produce consistent results?
3. How sensitive is the calibration to realistic image-corner noise?
4. What happens when the robot motions are geometrically degenerate?

The stage implements and compares three classical methods:

* **Tsai-Lenz**
* **Park-Martin**
* **Andreff**

It also includes an intentionally degenerate yaw-only experiment to demonstrate why good motion diversity is necessary for reliable calibration.

---

## Transform being estimated

The target quantity is:

```text
T_flange_cam
```

which maps points expressed in the camera frame into the robot flange frame according to the transform convention used throughout this project.

The calibration is formulated using the standard hand-eye relationship:

```text
A X = X B
```

where:

* `A` represents relative robot-flange motion,
* `B` represents the corresponding relative calibration-target motion observed by the camera,
* `X` is the unknown flange-to-camera transform.

Explicit transform names such as:

```text
T_base_flange
T_flange_cam
T_cam_board
T_base_board
```

are used throughout the implementation to make frame directions easier to inspect and reduce ambiguity.

---

## Experimental setup

A camera is mounted at a known transform relative to the robot flange in simulation.

For each calibration observation:

1. A flange pose is selected.
2. The camera pose follows from the known flange-to-camera mounting transform.
3. The calibration board is observed from the camera.
4. The board pose is recovered from the synthetic image observation.
5. Relative robot and board motions are constructed.
6. The hand-eye solver estimates `T_flange_cam`.
7. The estimated transform is compared directly with the known ground-truth transform.

Because the true camera mounting transform is available in simulation, the final error can be measured directly in:

* translation error in millimetres;
* rotation error in degrees.

---

# Baseline result

The baseline experiment commands **18 flange poses**, of which **16 produce usable calibration-board observations**.

With no added image-corner noise, all three implemented methods recover almost the same transform.

| Method  | Translation error | Rotation error |
| ------- | ----------------: | -------------: |
| PARK    |          0.593 mm |         0.079° |
| TSAI    |          0.595 mm |     **0.076°** |
| ANDREFF |      **0.591 mm** |         0.077° |

The agreement between the methods is important because the three solvers use different mathematical formulations.

The best translation result in this experiment is approximately:

```text
0.591 mm
```

with a rotation error of approximately:

```text
0.077°
```

These values demonstrate that the synthetic calibration pipeline is internally consistent under controlled conditions.

They should **not** be interpreted as expected real-camera accuracy.

The result depends on factors including:

* synthetic rendering;
* exact robot poses;
* accurate board geometry;
* sub-pixel corner localisation;
* the absence of real lens, lighting, synchronization, mechanical, and detection errors.

---

# Added image-noise experiment

The noise-free experiment represents an idealized case.

To examine a more realistic detection scenario, Gaussian noise can be added to the detected image corners.

For example:

```bash
python 01_handeye.py --n 18 --noise-px 0.3
```

A development run with approximately **0.3 px corner noise** produced roughly:

```text
6 mm translation error
0.8° rotation error
```

This run is not currently committed as a result file, so these figures should be treated as development-run notes rather than a manifest-backed result.

The experiment illustrates an important practical point:

> Very small errors obtained from ideal synthetic observations should not be interpreted as real-hardware calibration accuracy.

The added-noise run provides an indication of how image-measurement uncertainty can propagate into the recovered hand-eye transform, but the numerical result should be committed as an artifact before being treated as a reproducible Stage 2 result.

---

# Why the hand-eye solvers are implemented in this project

The project includes NumPy implementations of:

* Tsai-Lenz (1989)
* Park-Martin (1994)
* Andreff (1999)

inside:

```text
handeye_solvers.py
```

This avoids making the project dependent on a specific OpenCV Python API configuration.

In the OpenCV 5.0 environment used during development, `cv2.calibrateHandEye` was not available through the Python bindings, so the three methods were implemented directly.

During development, the implementations were cross-checked against OpenCV 4.13.

For noise-free synthetic data, the implementations matched the OpenCV results to machine precision.

Development notes also record that, under added corner noise, the resulting translation estimates agreed to approximately:

```text
0.04 mm
```

No committed Stage 2 result artifact currently backs the 0.04 mm figure, so it should be treated as a development-note result rather than a reproducible repository result.

The implementations also reproduce the expected failure behaviour in the deliberately degenerate motion experiment described below.

This comparison provides an independent development check that the custom implementations reproduce the established solver behaviour under the tested conditions.

---

# Degenerate-motion experiment

Hand-eye calibration requires sufficient rotational diversity in the relative robot motions.

If all robot rotations occur around the same axis, part of the translation can become unobservable.

To demonstrate this failure mode directly, the stage provides:

```bash
python 01_handeye.py --n 18 --degenerate
```

The degenerate motion set uses **yaw-only rotations**.

A representative result is:

| Component |         Error |
| --------- | ------------: |
| x         |      0.066 mm |
| y         |      0.602 mm |
| **z**     | **81.000 mm** |
| Rotation  |    **0.053°** |

At first glance, this result can appear successful.

The rotation error is very small, and the x/y translation components are close to the ground truth.

However, the z component is wrong by:

```text
81 mm
```

which corresponds to the camera offset that cannot be recovered from the degenerate motion set.

This example demonstrates why checking only rotation error, reprojection quality, or a subset of translation components can be misleading.

## Detecting the degeneracy before solving

The axis-spread check is a heuristic. The underlying problem is that
`_solve_translation` fits `(R_A - I) t_X = R_X t_B - t_A` by least squares, and
if every relative rotation shares an axis, that stacked matrix is rank 2 --
`lstsq` returns the minimum-norm answer without complaint.

`motion_conditioning()` takes the SVD of that matrix directly:

| Motion set | Singular values        | Weakest direction        |
| ---------- | ---------------------- | ------------------------ |
| Diverse    | 10.849, 10.667, 4.313  | [+0.169, -0.098, -0.981] |
| Yaw-only   | 7.716, 7.716, 1.46e-15 | [-0.000, -0.000, -1.000] |

On the yaw-only set the third singular value is at machine precision and the
null direction is the shared rotation axis. The recovered translation was wrong
by 81.000 mm in z and by less than 0.7 mm in x and y, so the SVD predicts not
just that the calibration will fail but which component will be wrong.

This needs no ground truth -- it is computed from the flange poses alone, before
solving. The 81 mm figure and the axis-spread check both require knowing the
answer already.

The diverse set is still least constrained mostly along z: its weakest direction
is `[+0.169 -0.098 -0.981]`, with `σ₃ = 4.313` versus `σ₁ = 10.849`. This is
consistent with the motion sampler using approximately ±22° roll and pitch but
±60° yaw, so z-axis rotation dominates by design. The direction is nevertheless
constrained strongly enough that the stacked system remains full rank.

For the yaw-only set, `σ₃ = 1.464e-15`, approximately fifteen orders of
magnitude below the smallest singular value of the diverse set. Reporting this
smallest singular value directly is more informative than reducing the result
to an infinite condition number.

---

# Solver behaviour under degeneracy

The different hand-eye methods do not necessarily fail in exactly the same way when the motion set is poorly conditioned.

For the deliberately degenerate yaw-only experiment:

* ANDREFF can recover a visually plausible rotation while missing the unobservable translation component;
* PARK can produce a rotation approximately **180° incorrect**;
* TSAI can produce a rotation approximately **128° incorrect**.

By contrast, when the well-conditioned motion set is used, all three methods produce closely agreeing estimates.

This illustrates an important calibration principle:

> Agreement between multiple solvers under a well-designed motion set is useful evidence of consistency, but no solver can recover information that is absent from the observations.

---

# Camera-coordinate convention

One of the easiest ways to obtain a plausible but incorrect hand-eye result is to mix camera-coordinate conventions.

MuJoCo/OpenGL and OpenCV do not use the same camera coordinate system.

In this project:

### MuJoCo / OpenGL-style camera

```text
-Z forward
+Y up
```

### OpenCV camera

```text
+Z forward
+Y down
```

The conversion used between the two conventions is:

```text
diag(1, -1, -1)
```

which flips the Y and Z axes.

Incorrectly handling this conversion can cause a transform chain to produce visually reasonable but geometrically incorrect results.

---

# Frame-consistency verification

The coordinate conversion is verified using an additional consistency test.

The calibration board is fixed in the robot base frame.

For multiple camera poses, the estimated board pose is transformed back into the base frame using the complete transform chain.

Conceptually:

```text
T_base_board
=
T_base_flange
T_flange_cam
T_cam_board
```

Because the physical board is stationary, the recovered `T_base_board` should remain approximately constant across observations.

If the reconstructed board pose changes substantially as the camera moves, then at least one transform direction, frame convention, or coordinate conversion is incorrect.

This provides a useful system-level verification beyond simply checking the output of the hand-eye solver.

---

# Why explicit transform names are used

Hand-eye calibration literature often uses argument names such as:

```text
gripper2base
target2cam
```

These names can become difficult to reason about when several transforms are inverted or chained together.

This project instead uses explicit notation such as:

```text
T_base_flange
T_flange_cam
T_cam_board
```

The notation follows the form:

```text
T_destination_source
```

so the frame relationship remains visible in the variable name.

This convention reduces the chance of accidentally applying a transform in the wrong direction.

---

# Why the flange poses are applied directly

The Stage 2 simulation applies flange poses directly rather than generating them using a complete robot model and inverse kinematics.

This is intentional.

The hand-eye calibration algorithm only requires:

* flange poses;
* calibration-target poses observed by the camera.

It does not depend on the kinematic mechanism that generated those flange poses.

Using direct flange poses therefore isolates the calibration problem from unrelated effects such as:

* inverse-kinematics convergence;
* joint limits;
* robot workspace constraints;
* trajectory generation.

This makes Stage 2 a controlled calibration experiment rather than a complete manipulation experiment.

Actual UR5e forward and inverse kinematics are introduced later in **Stage 5**.

---

# Running the stage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline calibration experiment:

```bash
python 01_handeye.py --n 18
```

Run the deliberately degenerate motion experiment:

```bash
python 01_handeye.py --n 18 --degenerate
```

Run the added image-noise experiment:

```bash
python 01_handeye.py --n 18 --noise-px 0.3
```

For reproducible comparisons, the exact command and software environment used for an experiment should be recorded together with the generated result.

---

# Interpreting the results correctly

The Stage 2 results support the following conclusions:

### Supported

* The synthetic hand-eye calibration pipeline can recover the known camera mounting transform with sub-millimetre error under ideal synthetic observations.
* Tsai-Lenz, Park-Martin, and Andreff produce closely agreeing results for the well-conditioned baseline experiment.
* Added image-corner noise significantly increases the final calibration error in the development run described above.
* Yaw-only robot motion makes the stacked translation system rank deficient before solving.
* The SVD identifies the unobservable translation direction directly from the flange motions.
* In the yaw-only experiment, the weakest direction is `[-0.000 -0.000 -1.000]`, and the recovered translation error is concentrated in z.
* Coordinate-frame consistency is critical when combining MuJoCo/OpenGL and OpenCV.

### Not supported

The current experiments do **not** establish that:

* a real robot will achieve 0.591 mm hand-eye accuracy;
* the synthetic noise model captures every real camera error source;
* the axis-spread heuristic is a complete observability test;
* the uncommitted 0.3 px noise and OpenCV cross-check figures are reproducible repository results;
* the calibration has already been validated on physical hardware.

These limitations are intentional and are documented to keep the reported conclusions within the evidence produced by the experiment.

---

# Limitations

Stage 2 remains a simulation-based validation experiment.

Important effects not yet represented by the baseline simulation include:

* real camera sensor noise;
* motion blur;
* lens-model mismatch;
* imperfect printed calibration-board geometry;
* board mounting error;
* robot encoder and kinematic calibration error;
* timestamp and synchronization error;
* mechanical compliance;
* camera-mount flex;
* lighting-dependent corner-detection behaviour;
* temperature-related mechanical drift.

For this reason, the simulation results are best interpreted as verification of the calibration methodology and software implementation rather than as a final real-world accuracy claim.

---

# Future improvements

The main extension planned for this stage is:

### Real robot hand-eye calibration

Repeat the experiment using a physical robot, camera, and calibration board.

This would make it possible to compare the simulation-derived behaviour against real measurement uncertainty and test how the SVD-based motion diagnostic behaves with real measurement noise.

---

# Current status

* [x] Synthetic hand-eye calibration against a known camera mounting transform
* [x] Tsai-Lenz NumPy implementation
* [x] Park-Martin NumPy implementation
* [x] Andreff NumPy implementation
* [x] Cross-check against OpenCV 4.13 during development
* [x] Added image-corner-noise experiment
* [x] Deliberately degenerate yaw-only motion experiment
* [x] Basic relative-motion rotation-axis diversity warning
* [x] Singular-value / conditioning-based observability diagnostic
* [x] MuJoCo/OpenGL → OpenCV camera-frame conversion
* [x] Base-frame board-pose consistency verification
* [ ] Real robot hand-eye calibration

---

# Key result

Under ideal synthetic observations, the best recovered hand-eye transform has approximately **0.591 mm translation error** and **0.077° rotation error**.

A separate development run with approximately **0.3 px image-corner noise** produced roughly **6 mm translation error** and **0.8° rotation error**, but that run is not currently committed as a Stage 2 result artifact.

The deliberately degenerate yaw-only experiment is analysed in **Detecting the degeneracy before solving** above. The important result is that the SVD exposes the missing translation direction from the flange motions alone, before the hand-eye solve is attempted.

The central result of this stage is therefore not only the baseline calibration accuracy:

> **Hand-eye calibration quality depends on both measurement quality and the geometric observability of the robot motion set. A numerically plausible result can still be physically wrong when the calibration motions are degenerate, and the singular values of the motion system can expose that failure before solving.**
