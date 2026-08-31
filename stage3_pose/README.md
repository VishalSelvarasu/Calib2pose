# Stage 3 — 6D Object Pose with ArUco Markers

This stage estimates the 6D pose of the YCB power drill in the robot base frame.
It is where the intrinsic calibration, hand-eye transform, and image-based PnP
estimate are combined into one transform chain:

```text
T_base_drill = T_base_flange @ T_flange_cam @ T_cam_drill
               robot pose        Stage 2       solvePnP here
```

The marker-based result is also the high-accuracy baseline for the learned
markerless system in Stage 4.

## Evaluation metric

I use **ADD** (Average Distance of Model Points). The drill mesh contains 8,945
points. For each test pose, the points are transformed once with the estimated
pose and once with the known ground-truth pose, and the mean 3D distance is
reported.

The mesh diameter used for ADD is 226.3 mm: the largest distance between any two
of the 8,945 mesh vertices. That makes the ADD-0.1d threshold 22.6 mm. The
drill's axis-aligned bounding-box diagonal is 274.0 mm, which is a different
length and is not the ADD diameter. I used the bounding-box diagonal by mistake
in an earlier version of this project; every pass rate it produced was several
points too generous.

Because the drill is asymmetric, standard ADD is appropriate; a symmetric
object would require a symmetry-aware metric such as ADD-S.

## Results

I evaluated 30 camera poses on a dome above the object. Twenty-eight views had
enough visible marker corners for PnP.

| Run | Mean ADD | Median | Worst | ADD-0.1d pass |
|---|---:|---:|---:|---:|
| True hand-eye, no noise | 1.40 mm | 1.15 mm | 3.64 mm | 100% |
| Stage 2 hand-eye (0.591 mm error), no noise | 1.55 mm | 1.29 mm | 3.92 mm | 100% |
| True hand-eye, 0.5 px corner noise | 1.66 mm | 1.47 mm | 4.23 mm | 100% |
| **Degenerate Stage 2 hand-eye (~81 mm error)** | **81.40 mm** | **81.37 mm** | **84.07 mm** | **0%** |

The 100% ADD-0.1d pass rate is 28 of 28 evaluated views. With that few samples
the 95% Wilson lower bound is 87.9%, so this should be read as "no failures
observed in 28 views" rather than as evidence that the marker pipeline never
fails.

For the baseline run, mean reprojection error is 0.533 px and mean rotation
error is 0.390°. A 1.40 mm ADD on a 226.3 mm object is about 0.6% of the object
diameter.

## What the error comparison shows

Relative to the 1.40 mm baseline:

| Source | Added mean ADD |
|---|---:|
| 0.5 px corner noise | +0.26 mm |
| 0.591 mm Stage 2 hand-eye error | +0.15 mm |
| Degenerate Stage 2 hand-eye | **+80.00 mm** |

The deliberately bad hand-eye transform dominates everything else.

More importantly, that upstream error is almost invisible if Stage 3 is judged
only by its own local signals:

| Signal | Baseline | Degenerate hand-eye |
|---|---:|---:|
| Reprojection | 0.533 px | 0.533 px |
| Rotation error | 0.390° | 0.385° |
| Markers detected / view | 2.86 | 2.86 |
| **ADD** | **1.40 mm** | **81.40 mm** |

The image measurements and PnP solve have not become worse. The coordinate
system used to place the answer in the robot base frame is wrong. This is the
example that convinced me not to treat a local residual as an end-to-end
validation metric.

## A geometry bug I found while building the marker rig

The marker plates are MJCF boxes. In MuJoCo, the `pos` of a box is its centre,
but the marker texture is painted on the +Z face. My first implementation built
the marker 3D corners on the centre plane instead of the visible face.

That small offset produced an error that changed with plate tilt:

| Plate | Tilt | Reprojection before | After |
|---|---:|---:|---:|
| 0 | 0° | 0.68 px | 0.60 px |
| 1 | 40° | 1.91 px | 0.80 px |
| 2 | 40° | 3.38 px | 0.65 px |

On the test view, the overall pose error dropped from 7.91 mm to 1.14 mm after
fixing the marker plane.

The useful part of this bug was the failure pattern: all four corners on a
slanted plate shifted together. That can easily look like a camera-calibration
issue. Reprojecting known 3D points using the **true** pose made it clear that the
plate geometry, not PnP, was wrong.

## Marker layout

The drill carries three 50 mm plates with 40 mm ArUco markers:

| ID | Position in drill frame | Orientation |
|---|---|---|
| 0 | `(0, 0, 115)` mm | Flat, +Z |
| 1 | `(50, 0, 92)` mm | 40° toward +X |
| 2 | `(0, -78, 92)` mm | 40° toward −Y |

I first tried mounting markers directly on flat object faces. From many overhead
views only the top marker was usable, which leaves PnP with a mostly planar
configuration. Angling the plates keeps multiple marker planes visible from a
wider range of camera poses. In the final test, 24 of the 28 usable views see all
three markers.

## Running the stage

```bash
pip install -r requirements.txt

python 01_pose_markers.py
python 01_pose_markers.py --handeye estimated
python 01_pose_markers.py --noise-px 0.5
python 01_pose_markers.py --handeye estimated \
    --handeye-json ../stage2_handeye/results/handeye_degenerate.json
```

`--handeye estimated` expects the Stage 2 result file to exist.

The YCB drill mesh is downloaded automatically on first use from the
`vikashplus/YCB_sim` MuJoCo port (Apache 2.0). The downloaded `assets/` directory
is gitignored.

## What comes next

Stage 4 removes the markers and predicts eight 2D object keypoints from the
image. The predicted keypoints are passed to the same PnP stage, so this
1.40 mm marker result serves as the accuracy baseline for the markerless system.

## Current status

- [x] ArUco-based multi-plane pose estimation
- [x] PnP evaluation against known ground truth
- [x] ADD / ADD-0.1d evaluation
- [x] Stage 2 hand-eye error propagation test
- [x] Marker geometry debugging and corrected plate offsets
- [x] Markerless comparison baseline for Stage 4
- [ ] Real-camera fiducial baseline
