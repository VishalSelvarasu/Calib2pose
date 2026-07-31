# Stage 3 — 6D Object Pose, Scored with ADD

Estimates the 6D pose of the YCB power drill in robot base coordinates by
chaining everything the project has built:

```
T_base_drill = T_base_flange @ T_flange_cam @ T_cam_drill
               ^ forward kin    ^ stage 2      ^ solvePnP, here
```

Scored with **ADD** — the mean distance between the object's 8945 model points
transformed by the estimated pose versus the true pose. A pose counts as correct
at ADD < 0.1 × object diameter. The drill's diameter is 274.0 mm, so the
threshold is 27.4 mm.

The drill is asymmetric, so plain ADD applies. A symmetric object would require
ADD-S, which matches each point to its nearest neighbour instead.

## Results

30 camera poses on a dome above the object, 28 with enough markers visible:

| run | mean ADD | median | worst | ADD-0.1d pass |
|---|---|---|---|---|
| true hand-eye, no noise | 1.40 mm | 1.15 mm | 3.64 mm | 100 % |
| stage 2 hand-eye (0.591 mm error), no noise | 1.55 mm | 1.29 mm | 3.92 mm | 100 % |
| true hand-eye, 0.5 px corner noise | 1.66 mm | 1.47 mm | 4.23 mm | 100 % |
| **stage 2 degenerate hand-eye (81 mm error)** | **81.40 mm** | 81.37 mm | 84.07 mm | **0 %** |

Baseline run: mean reprojection 0.533 px, mean rotation error 0.390°.
1.40 mm on a 274 mm object is 0.5 % of diameter.

## Error sources are not remotely equal

Measured against the baseline, each source costs:

| source | added ADD |
|---|---|
| 0.5 px corner detection noise | +0.26 mm |
| stage 2 hand-eye error of 0.591 mm | +0.15 mm |
| stage 2 degenerate hand-eye | **+80.00 mm** |

The two sources with visible local symptoms are negligible. The one that
dominates has no local symptom at all.

## Upstream error propagates, and only translation shows it

The last row of both tables is the point of the whole project. Stage 2's
degenerate run recovered rotation to 0.053° and x/y to a fraction of a
millimetre, failing only in the mathematically unobservable z axis — by 81 mm.

Feed that transform into stage 3 and the pose error is 81.40 mm at a 0 % pass
rate, while stage 3's own quality signals stay green:

| signal | baseline | with degenerate hand-eye |
|---|---|---|
| reprojection | 0.533 px | 0.533 px |
| rotation error | 0.390° | 0.385° |
| markers detected/view | 2.86 | 2.86 |
| **ADD** | **1.40 mm** | **81.40 mm** |

Nothing measurable inside stage 3 indicates a problem. The images look right,
the markers detect cleanly, the solver converges, the residual is small. The
drill is simply 81 mm from where the robot thinks it is.

This is why every stage in this project is validated against a known answer
rather than against its own residual.

## The plate offset bug

Found during development, fixed in the shipped code. Recorded because the
failure signature is worth knowing.

Markers are painted on the +Z **face** of each plate, but `pos` in MJCF is the
**centre** of the box. Building the marker's 3D corners at the centre plane
instead of the face left an error that rotated with the plate:

| plate | tilt | reprojection before | after |
|---|---|---|---|
| 0 | 0° | 0.68 px | 0.60 px |
| 1 | 40° | 1.91 px | 0.80 px |
| 2 | 40° | 3.38 px | 0.65 px |

Overall pose error fell from 7.91 mm to 1.14 mm on the test view. The signature
was a *uniform* shift of all four corners on the tilted plates — which reads
like a camera calibration problem rather than a geometry one, and is why it was
caught by reprojecting known 3D points through the *true* pose rather than by
looking at the solver's residual.

## Marker layout

Three 50 mm plates carrying 40 mm markers, angled rather than mounted flat on
the object's faces:

| id | position in drill frame | tilt |
|---|---|---|
| 0 | (0, 0, 115) mm | flat, +Z |
| 1 | (50, 0, 92) mm | 40° toward +X |
| 2 | (0, −78, 92) mm | 40° toward −Y |

Flat-on-the-faces was the first attempt: only the top marker was ever visible,
and a single planar marker gives an ill-conditioned PnP. Angling the plates
means at least two are oblique-but-visible from anywhere above, which keeps the
point set non-planar. 24 of 28 usable views see all three markers.

## Usage

```bash
pip install -r requirements.txt
python 01_pose_markers.py                       # true hand-eye
python 01_pose_markers.py --handeye estimated   # stage 2's solved transform
python 01_pose_markers.py --noise-px 0.5
python 01_pose_markers.py --handeye estimated \
    --handeye-json ../stage2_handeye/results/handeye_degenerate.json
```

`--handeye estimated` requires stage 2 to have been run first.

The YCB drill mesh (~22 MB) downloads automatically on first run from the
`vikashplus/YCB_sim` MuJoCo port, Apache 2.0. `assets/` is gitignored.

## Next

Stage 4 removes the markers: train a keypoint detector on synthetic renders and
feed its output to the same `solvePnP` call. **1.40 mm mean ADD at a 100 % pass
rate is the baseline it has to beat.**