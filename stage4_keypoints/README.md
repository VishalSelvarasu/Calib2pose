# Stage 4 — Markerless Pose from Learned Keypoints

Replaces the ArUco markers of stage 3 with a network that predicts the drill's
8 bounding-box corners directly from the image. The predicted keypoints feed the
same `solvePnP` call, so the pose stage is unchanged and the comparison is
clean.

**Baseline to beat: 1.40 mm mean ADD, 100 % ADD-0.1d pass rate** (stage 3,
marker-based).

## Keypoints

The 8 corners of the drill's axis-aligned bounding box, in the object frame.
Mean pairwise separation 171.7 mm on a 274.0 mm object — wide spread, which
conditions the PnP solve well.

Bounding-box corners are the standard choice in the 6D pose literature (BB8,
YOLO-6D, PVNet), which makes the numbers comparable to published work. The
known cost: **no corner sits on a visible surface**, so the network has no local
evidence at any keypoint and must infer all eight from global shape. The
predicted failure mode is corner *swaps* rather than corner *drift*, which would
show up as a bimodal ADD distribution rather than smooth degradation.

## Renderer

`01_render.py` produces domain-randomized images at 640×640 with labels.

Randomized: object pose (uniform on SO(3) via quaternion sampling — sampling
Euler angles uniformly clusters at the poles), camera pose on a dome with random
roll, ambient level and 1–2 directional lights, floor colour, 0–4 YCB distractor
objects, and camera `fovy` (±3°, so the network cannot memorise one intrinsic
matrix).

Not randomized: the drill's own texture, and the 640×640 resolution.

### Occlusion is labelled, not avoided

Every image records `visible_frac` — the fraction of the drill's projected area
surviving occlusion, measured by a segmentation pass with and without the
distractors. This means one dataset supports a clean-vs-occluded ablation at
evaluation time instead of requiring two.

Four constraints keep occlusion useful rather than destructive:

- `MIN_VISIBLE_FRAC = 0.35` — the fraction of the drill's *own* area still
  visible. An early debug batch produced an image at **4 % visible**; that is
  label noise, not a hard example.
- `MIN_VISIBLE_PX = 6000` — an absolute floor, ~1.5 % of the frame.
  `MIN_VISIBLE_FRAC` alone is insufficient because it is a *ratio*: a drill far
  from the camera and 60 % occluded passes the ratio test at only a few hundred
  pixels. Visual inspection caught this; the statistics did not.
- `KP_MARGIN = 15 px`, `MAX_KP_OUTSIDE = 2` — the bbox is larger than the drill,
  so demanding all 8 corners strictly in-frame would reject good close-ups. But
  corners drifting far outside give the network a target with no visible
  evidence and nowhere to put the heatmap peak.
- `MIN_SEPARATION = 0.055 m` — stops distractors spawning inside the drill mesh,
  which renders as fused objects rather than as occlusion.

A bug worth recording: occlusion was originally measured on one distractor
arrangement and then the objects were re-randomized before the final render, so
`visible_frac` described a different image than the one saved. Fixed by storing
the measured poses and restoring exactly those.

Typical output: ~82 % mean visibility, ~55 % of images with some occlusion,
~17 % below 70 % visible. Reject rate around 14 %.

## Chunked generation

10 000 images does not fit one session, so rendering is chunked:

```bash
python 01_render.py --n 2000 --start-idx 0
python 01_render.py --n 2000 --start-idx 2000
...
python 02_build_dataset.py --val 1000 --test 1000
```

The random seed defaults to `--start-idx`, so chunks never repeat poses.
`02_build_dataset.py` checks for duplicate seeds and duplicate filenames, and
verifies every manifest entry has an image on disk.

The test split is **stratified by occlusion**, so all three splits carry the same
difficulty mix. A plain random split can hand you a test set that is easier than
training, making the final number meaningless.

## Debug first

```bash
python 01_render.py --n 12 --start-idx 0 --debug --out debug_data
```

Draws the projected wireframe on each image. If the box does not enclose the
drill, or the corner ordering is inconsistent between frames, stop — a label bug
is invisible in a training loss curve and will cost days.

This step is not optional. Both the missing absolute pixel floor and the
over-permissive frame margin passed every summary statistic and were caught only
by looking at the images.

### But the eye can also be wrong

Several debug images looked like the box had missed the drill entirely. It had
not. `03_verify_labels.py` overlays the drill's true mesh vertices (magenta)
through the same stored pose:

```bash
python 03_verify_labels.py --data debug_data --n 12 --save
```

The 3D box encloses the mesh by construction, so its projection **must** enclose
the mesh's projection. Containment is therefore a hard invariant, and the script
reports it per image. It has never failed.

What those images actually show is either heavy occlusion, or a near edge-on
view where the box projects as a narrow sliver. The script flags the latter via
the projected aspect ratio. Roughly 10-15 % of views come out below 0.3, and
they are correct labels that are simply poorly conditioned for PnP — legitimate
hard examples, not errors.

## Status

- [x] Keypoint definition, verified visually
- [x] Renderer with domain randomization and occlusion labelling
- [x] Chunked generation and stratified splits
- [ ] Model: ResNet-18 backbone, deconv heatmap head
- [ ] Training loop with per-epoch checkpointing
- [ ] Inference → PnP → ADD, compared against the stage 3 baseline

Heatmaps rather than direct coordinate regression: regression forces the network
to encode spatial position in its final activations, while heatmaps stay spatial
until the argmax and support subpixel refinement on the peak. PnP accuracy is
directly limited by keypoint precision, so that subpixel step matters.