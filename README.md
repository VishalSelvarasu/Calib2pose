# calib2pose

`calib2pose` is a simulation-first, end-to-end study of the geometry behind
robot vision: camera intrinsics, hand-eye calibration, 6D object pose estimation,
learned keypoints, and the effect of perception error on a robot task.

The main idea is simple: I do not want to judge a stage only by the residual it
produces. Wherever possible, I first run it in simulation, where the true
transform is known, and measure the result directly in millimetres and degrees.
That makes it possible to catch failures that would otherwise look perfectly
reasonable from inside the estimator.

Each stage constructs its own known camera model rather than consuming the
previous stage's output. That is deliberate: it isolates each error source so a
stage's result reflects only what that stage is being tested on. The exception
is Stage 5, which propagates the measured Stage 4 error distribution forward on
purpose.

## Project at a glance

| Stage | Main result | Status |
|---|---|---|
| [1. Camera intrinsics (ChArUco)](stage1_intrinsics/) | `fx` recovered within 0.10% of ground truth | Synthetic validation complete; real capture pending |
| [2. Hand-eye calibration](stage2_handeye/) | 0.591 mm / 0.077° against a known camera mount | Simulation complete |
| [3. 6D pose with ArUco markers](stage3_pose/) | 1.40 mm mean ADD, 100% ADD-0.1d | Simulation complete |
| [4. 6D pose with learned keypoints](stage4_keypoints/) | 11.21 mm mean ADD, 90.2% ADD-0.1d (90.0% without ground-truth keypoint filtering) | Synthetic evaluation complete |
| [5. Task-level UR5e evaluation](stage5_task_error/) | 84.8% of trials within a 15 mm placement tolerance | Simulation complete |

![Qualitative Stage 4 results](stage4_keypoints/results/qualitative_mesh.png)

The figure above shows held-out Stage 4 test images. **Green** is the true 3D
bounding box, **orange** is the box recovered from the network predictions after
`solvePnP`, and **magenta** shows projected mesh vertices. `visible` is the
fraction of the drill that remains visible after occlusion.

The clearest pattern in this experiment was visibility. Fully visible examples
can be accurate across very different viewpoints, while the largest errors are
concentrated in heavily occluded cases.

## Why I built the project around ground truth

On real hardware, the true camera intrinsics, camera-to-flange transform, and
object pose are usually unknown. That means a small solver residual is useful,
but it is not the same thing as knowing the estimate is physically correct.

Using simulation first gave me a way to separate those two ideas. Several of the
most useful results in the project came from cases where the local metric looked
good but the known ground truth showed that something was wrong.

## Stage 1: lower reprojection error did not mean better calibration

I compared a well-distributed ChArUco pose set with a deliberately poor,
mostly-frontal pose set. The frontal set achieved a lower RMS reprojection error
(0.209 px vs 0.249 px), but its estimated `fx` was off by 1.95% and `k2` changed
sign relative to the ground truth.

That experiment is a good reminder that reprojection error measures how well the
chosen model explains the observations. It does not, by itself, tell us whether
the calibration parameters are well constrained.

## Stage 2: a hand-eye result can look excellent while one direction is unobservable

With a well-spread set of robot motions, the custom hand-eye solvers recover the
known camera mount to about 0.6 mm and 0.08°.

I also tested a yaw-only motion set. In that case the translation along the shared
rotation axis is not observable. The solver still returned a rotation error of
only 0.053°, while the missing translation component was wrong by 81 mm.

That failure motivated an explicit motion-diversity check before solving.

## Stage 3: the upstream error propagates without changing the local PnP residual

When the degenerate Stage 2 transform is passed into Stage 3, the final object
pose error becomes 81.40 mm ADD with a 0% ADD-0.1d pass rate.

What makes this result interesting is that the Stage 3 measurements still look
normal: reprojection error stays at 0.533 px, the markers are detected, and the
rotation estimate remains close to the baseline. The problem is in the upstream
coordinate transform, so the local pose solver has no way to expose it.

## Stage 4: several reasonable hypotheses were wrong

The first learned-keypoint model produced 33.78 px mean 2D keypoint error. Its
corresponding pose result was 34.79 mm mean ADD with a 55.5% ADD-0.1d pass
rate. I initially suspected corner-identity swaps and heatmap quantisation.

The measurements pointed somewhere else. Most of the error came from
localisation rather than identity, RANSAC barely changed the result, and doubling
the heatmap resolution did not help. Stronger augmentation and a longer training
schedule did: the final model reached 10.78 px keypoint error, 11.21 mm mean ADD,
and 90.2% ADD-0.1d.

The evaluation originally used ground-truth keypoint visibility to decide which
correspondences reached PnP, which a deployed system would not have. Re-running
without it gives 90.0% instead of 90.2%, so the protocol was not what produced
the result.

## Stage 5: benchmark tolerance and task tolerance are not the same thing

For this drill, ADD-0.1d corresponds to a 22.6 mm threshold. In the UR5e
simulation I propagated the measured Stage 4 error distribution through a fixed
grasp-pose function and inverse kinematics.

On the same set of trials, 92.4% landed within 22.6 mm, while 84.8% landed within
15 mm and only 41.4% landed within 5 mm. The perception result has not changed;
only the tolerance associated with the downstream task has changed.

This is not yet a physical grasping experiment or a live perception-in-the-loop
controller. It is a task-level error propagation study, which is the next step
between pose metrics and real robot trials.

## A metric bug made the early pass rates too generous

ADD-0.1d uses 10% of the object mesh diameter as its success threshold. I
initially used the drill's 274.0 mm axis-aligned bounding-box diagonal as that
diameter. That was wrong: the actual mesh diameter is 226.3 mm, the largest
distance between any two of the 8,945 mesh vertices, so the correct ADD-0.1d
threshold is 22.6 mm.

That mistake made every pass rate it produced several points too generous. I
corrected the diameter calculation and kept the failure documented because it is
exactly the kind of plausible metric error that can survive unless the quantity
being measured is checked against its definition.

## Markers vs markerless pose

Stage 4 replaces the Stage 3 ArUco observations with learned image keypoints but
keeps the same PnP-based pose recovery. That gives a reasonably clean comparison
between the two perception front ends.

| Method | Mean ADD | ADD-0.1d pass | Mean rotation error | Within 15 mm in Stage 5 |
|---|---:|---:|---:|---:|
| Markers (Stage 3) | 1.40 mm | 100% | 0.39° | 100% |
| Learned keypoints (Stage 4) | 11.21 mm | 90.2% | 3.36° | 84.8% |

The markerless system gives up roughly an order of magnitude in pose accuracy in
exchange for removing the need to attach fiducials to the object. On clean test
images it performs much better: 6.75 mm mean ADD and a 98.2% pass rate. Heavy
occlusion is the main failure mode, where the pass rate drops to about 68.8%.

## Object and metric

The experiments use the YCB power drill (`035_power_drill`). I chose it because
it is asymmetric, so standard ADD can be used without the symmetry handling
required by ADD-S.

ADD is the mean distance between model points transformed by the estimated pose
and the same points transformed by the ground-truth pose. The model contains
8,945 points in this project. ADD-0.1d counts a pose as correct when its ADD is
below 10% of the object diameter.

The use of a YCB object and a standard pose metric makes the evaluation easier to
relate to the 6D-pose literature, although the synthetic dataset and evaluation
protocol here should not be treated as directly equivalent to a standard
real-image benchmark such as YCB-Video.

## Current limitations

The main quantitative results are still simulation or synthetic-domain results.
The project therefore demonstrates the pipeline, the validation methodology, and
the observed failure modes, but it does not yet demonstrate sim-to-real
performance on a physical robot.

The most important next steps are:

- Run the Stage 1 calibration on the real webcam.
- Perform the hand-eye and marker baseline on real hardware.
