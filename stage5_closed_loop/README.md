# Stage 5 — Closed Loop: Does the Pose Error Actually Matter?

Stages 1–4 report pose error in millimetres. This stage answers the question
that error exists to serve: **if the robot acts on the estimate, where does its
hand end up?**

```
T_base_drill (estimated) → grasp pose → IK → joint command → flange pose
```

The metric is **placement error** — the distance between where the flange lands
using the estimated pose and where it would have landed using the true pose.
Everything downstream of perception is deterministic, so the difference isolates
exactly what the perception error cost.

## Results

200 trials per source. Grasp counted as successful within 15 mm, a
representative parallel-jaw margin.

| pose source | placement mean | median | p90 | max | grasp success |
|---|---|---|---|---|---|
| perfect (IK only) | 0.00 mm | 0.00 mm | 0.00 mm | 0.00 mm | 100 % |
| stage 3, markers | 1.23 mm | 1.03 mm | 2.15 mm | 3.38 mm | **100 %** |
| stage 4, learned keypoints | 10.03 mm | 6.02 mm | 22.57 mm | 90.88 mm | **84.0 %** |

The IK residual is **0.037 mm**, reported separately so kinematic error cannot be
confused with perception error. It is three orders of magnitude below the
perception error and can be ignored.

Caveat on the stage 3 row: that distribution has only **28 samples**, because
stage 3 evaluated 30 poses. Stage 4 has 1000. The stage 3 tail is therefore
poorly characterised and its true maximum is probably worse than 3.38 mm.

## The benchmark metric is more permissive than the gripper

ADD-0.1d — the standard 6D pose metric — passes at 0.1 × object diameter, which
for the 274 mm drill is **27.4 mm**. A parallel-jaw gripper wants closer to
15 mm, and a precision assembly task wants 5 mm.

Sweeping the tolerance on the same 500 trials:

| grasp tolerance | success | context |
|---|---|---|
| 27.4 mm | **95.6 %** | ADD-0.1d threshold |
| 15 mm | 84.0 % | parallel-jaw gripper |
| 10 mm | 70.0 % | tight grasp |
| 5 mm | **41.4 %** | precision assembly |

At the benchmark threshold the pipeline reads as a 95.6 % success, closely
matching stage 4's own 93.7 % ADD-0.1d figure — which is the check that the two
metrics agree when their thresholds do. At a realistic gripper margin it is
84 %. At assembly tolerance it fails more often than it succeeds.

Same model, same poses, same errors. Only the question changed.

This is the project's closing case of its running theme: a metric can be
correct, standard, and still answer a different question than the one that
matters. Stage 1's reprojection error preferred a broken calibration. Stage 2's
residual reported 0.053° on an 81 mm failure. Here, ADD-0.1d passes 93.7 % of
poses that a gripper would miss one time in six.

The tail is what does the damage: the mean placement error is 9.63 mm, but the
p90 is 18.57 mm and the worst trial lands 116 mm out — three drill-widths away.

## Where the error comes from

Rather than inventing a noise model, `--source stage3` and `--source stage4`
resample the **actual per-view errors** measured on those stages' test splits,
read straight from their results JSON. Each trial draws a real
(translation, rotation) error pair and applies it in a uniformly random
direction.

This matters because the tails decide grasp outcomes, not the mean. A
distribution fitted to a mean and standard deviation would understate the p90,
and the p90 is where grasps start to miss.

## Forward kinematics, finally

Stages 2 and 3 set the flange pose directly on a mocap body. That is legitimate
for testing solvers — the hand-eye solvers and `solvePnP` only ever see a list of
poses, and cannot tell where those poses came from. But it sidesteps what a real
robot does: accept a joint command and end up somewhere.

Here the arm is the MuJoCo Menagerie UR5e, the flange pose comes from forward
kinematics on `attachment_site`, and joint targets come from damped
least-squares IK.

The eye-in-hand camera is mounted at **exactly** the transform stage 2 solved
for, verified across arm configurations: recovering `T_flange_cam` from
`inv(FK(q)) @ T_base_cam` returns the ground-truth mount to 0.000000 mm and
0.000000° at every configuration tested.

## Self-occlusion is real here

Stages 2 and 3 moved the camera on a mocap body with no arm attached. On an
actual UR5e the wrist links sit in the eye-in-hand camera's field of view for
many configurations, because that mount transform was chosen without an arm in
the picture.

`find_observation_pose()` therefore scores candidate viewpoints by how much of
the drill is actually visible, using the same segmentation technique as stage 4's
renderer, and rejects ones where the wrist is in the way.

## Damped least-squares IK

```
dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e
```

The UR5e has a closed-form solution, but DLS is ~40 lines, takes the 6D pose
target directly, and degrades gracefully near singularities instead of returning
nothing. That matters here specifically: the targets are derived from a *noisy*
pose estimate, so some of them will be near-singular or slightly unreachable.
The damping term bounds the joint step at the cost of a small tracking error,
which is the right trade when the target came from a perception system anyway.

`grasp_pose_from_object()` is deliberately simple — approach straight down from
100 mm above the object origin, keeping only the object's yaw. The point is to
measure how pose error propagates into placement error, not to design a grasp.
Any fixed deterministic function of the object pose works, because the same
function is applied to the estimate and to ground truth.

Seeding matters: the grasp IK is seeded from the reference configuration, not
from the observation pose. Seeding from the observation pose converged to a local
minimum and produced a 245 mm miss on a run that should have been exact.

## Usage

```bash
pip install -r requirements.txt
python 00_fetch_ur5e.py
python 01_closed_loop.py --trials 200 --source perfect
python 01_closed_loop.py --trials 200 --source stage3
python 01_closed_loop.py --trials 200 --source stage4
python 01_closed_loop.py --trials 500 --source stage4 --grasp-tol-mm 5
```

`--source stage3` and `--source stage4` need those stages' evaluation JSON:

```
../stage3_pose/results/pose_true_0.0px.json
../stage4_keypoints/results/eval_test_aug100.json
```

## Notes

The UR5e model is [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
BSD-3. `ur5e/ur5e.xml` is the pristine file; `_ur5e_cam.xml` is generated with
the camera injected, because MJCF does not allow re-opening an included body to
add children and MuJoCo cameras attach to bodies rather than sites.

Two path quirks worth knowing: MuJoCo resolves `<include>` relative to the
including file but resolves the included file's `meshdir` relative to that same
outer file, so the generated scene must live *inside* `ur5e/`. And `ur5e.xml`
sets `meshdir` but not `texturedir`, so both are set explicitly in the wrapper.