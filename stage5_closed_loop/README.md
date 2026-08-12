# Stage 5 — Task-Level Error Propagation with a UR5e

Stages 1–4 measure pose error in millimetres and degrees. In this stage I ask a
more practical question: **if the robot uses an imperfect object pose, how far
from the intended grasp pose does the flange end up?**

The simulated chain is:

```text
estimated T_base_drill → fixed grasp-pose rule → IK → joint configuration → FK flange pose
```

The main metric is **placement error**: the distance between the flange pose
reached from the estimated object pose and the flange pose reached from the true
object pose.

This stage is a task-level propagation experiment. It is not yet a full live
closed-loop system: the Stage 4 network does not render and infer a new image
inside each UR5e trial, and the gripper does not close on the object with contact
dynamics. The experiment isolates how the measured perception error affects a
downstream robot target.

## Results

The stored Stage 5 result files contain 200 trials for the perfect and Stage 3
sources and 500 trials for Stage 4. A trial is counted as within tolerance when
the placement error is below 15 mm.

| Pose source | Mean | Median | p90 | Max | Within 15 mm |
|---|---:|---:|---:|---:|---:|
| Perfect pose (IK only) | 0.00 mm | 0.00 mm | 0.00 mm | 0.00 mm | 100% |
| Stage 3 markers | 1.23 mm | 1.03 mm | 2.15 mm | 3.38 mm | **100%** |
| Stage 4 learned keypoints | 9.63 mm | 6.15 mm | 18.57 mm | 116.11 mm | **84.8%** |

The mean IK residual is about **0.04 mm**, which is much smaller than the
perception-induced placement errors. For this setup, the kinematic solver is not
the main source of error.

The Stage 3 distribution should be interpreted with some caution. Its source
pose file contains only 28 evaluated object poses, so resampling those errors 200
times does not provide much information about the true tail of a marker-based
system.

## The task tolerance changes the conclusion

ADD-0.1d uses 10% of the object diameter as its success threshold. The drill is
about 226 mm across, so that threshold is 22.6 mm.

Using the same Stage 4 error trials with different placement tolerances gives:

| Placement tolerance | Fraction within tolerance | Interpretation |
|---|---:|---|
| 22.6 mm | **92.4%** | Same scale as ADD-0.1d |
| 15 mm | **84.8%** | Representative grasp-position margin |
| 10 mm | 70.0% | Tighter grasp placement |
| 5 mm | **41.4%** | Precision-placement scale |

The important point is not that 15 mm is a universal gripper threshold; it is
not. The point is that a benchmark threshold and a downstream task tolerance can
ask different questions of exactly the same perception errors.

The error distribution also has a long tail. The median Stage 4 placement error
is 6.15 mm, while the 90th percentile is 18.57 mm and the worst sampled trial is
116.11 mm. For task success, those tails matter more than the mean alone.

## How Stage 3 and Stage 4 errors are used

I did not fit a Gaussian noise model. `--source stage3` and `--source stage4`
read the per-view errors saved by the previous stages and resample measured
translation/rotation error pairs.

For each trial, the stored translation magnitude and rotation magnitude are
applied in a random direction/axis before the downstream grasp target is
computed.

This preserves the empirical magnitude distribution, including its tail, but it
also has a limitation: the original 6D error direction and correlations are not
preserved. A stronger next version would save and replay the actual per-image
SE(3) error transform instead of reconstructing one from two magnitudes.

## Bringing in a full robot model

Stages 2 and 3 apply flange poses directly to a simulated body because those
stages only need pose observations. Stage 5 uses the MuJoCo Menagerie UR5e so the
flange pose is produced by forward kinematics from a joint configuration.

Joint targets are found with damped least-squares inverse kinematics, and the
camera is attached to the flange using the same transform used in Stage 2.

As a consistency check, recovering `T_flange_cam` from
`inv(FK(q)) @ T_base_cam` across multiple arm configurations returns the known
mount transform to numerical precision in the simulation.

## Observation visibility and wrist occlusion

Once the camera is placed on a real UR5e model, the wrist itself can enter the
camera view. That did not appear in the earlier stages because they moved a
camera body without the complete robot geometry.

`find_observation_pose()` therefore scores candidate viewpoints using a
segmentation-based visibility check and rejects poses where too much of the drill
is hidden by the arm.

This observation-pose search is useful scene geometry for the simulation, even
though the current Stage 5 trials do not yet run the Stage 4 network on the
resulting camera image.

## Damped least-squares IK

The inverse-kinematics update is:

```text
dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e
```

The UR5e has closed-form IK solutions, but damped least squares is compact,
accepts a 6D pose error directly, and remains numerically well behaved near
singular configurations. That is useful here because noisy perception can move
a nominal target toward an awkward or slightly unreachable pose.

`grasp_pose_from_object()` is intentionally simple. It approaches 100 mm above
the object origin and keeps only the object's yaw when constructing the grasp
orientation. The goal of this stage is not to optimise grasp planning; it is to
apply the **same deterministic grasp rule** to the estimated pose and the true
pose and measure the difference.

I also found that IK seeding matters. Seeding from the observation pose could
converge to an unrelated local solution; using the fixed reference
configuration avoids that failure in the current experiment.

## Running the stage

```bash
pip install -r requirements.txt

python 00_fetch_ur5e.py
python 01_closed_loop.py --trials 200 --source perfect
python 01_closed_loop.py --trials 200 --source stage3
python 01_closed_loop.py --trials 500 --source stage4
python 01_closed_loop.py --trials 500 --source stage4 --grasp-tol-mm 5
```

The Stage 3 and Stage 4 sources require the evaluation JSON files from those
stages, for example:

```text
../stage3_pose/results/pose_true_0.0px.json
../stage4_keypoints/results/eval_test_aug100.json
```

## MuJoCo model notes

The UR5e model comes from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and is
BSD-3 licensed. `ur5e/ur5e.xml` is kept as the original model, while
`_ur5e_cam.xml` is generated with the eye-in-hand camera added.

There are two path details in MuJoCo worth keeping in mind. `<include>` paths
and mesh/texture asset paths are resolved in slightly different ways, so the
generated wrapper scene is kept inside the `ur5e/` directory and explicitly sets
both `meshdir` and `texturedir`.

## What I would change next

The main upgrade for this stage is to replace error resampling with actual
per-image pose predictions in the UR5e scene, then move from placement tolerance
to contact-based grasp-and-lift trials. That would turn this task-level
sensitivity study into a genuine perception-in-the-loop manipulation experiment.

## Current status

- [x] UR5e model integrated in MuJoCo
- [x] Eye-in-hand camera attached using the Stage 2 transform
- [x] Forward and inverse kinematics validation
- [x] Stage 3 marker-error propagation
- [x] Stage 4 learned-pose error propagation
- [x] Placement-tolerance sensitivity analysis
- [x] Wrist / arm visibility check for candidate observation poses
- [ ] Replay actual per-image SE(3) prediction transforms
- [ ] Run Stage 4 inference inside each UR5e trial
- [ ] Contact-based grasp-and-lift evaluation
- [ ] Physical robot validation
