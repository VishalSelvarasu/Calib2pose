# Stage 4 — Markerless Pose from Learned Keypoints

Replaces the ArUco markers of stage 3 with a network that predicts the drill's
8 bounding-box corners directly from the image. The predicted keypoints feed the
same `solvePnP` call, so the pose stage is unchanged and the comparison is clean.

## Results

Test split, 1000 images, never touched during training or tuning.

| run | change | epochs | keypoint err | mean ADD | ADD-0.1d pass |
|---|---|---|---|---|---|
| 1 | photometric aug, heatmap 64 | 60 | 33.78 px | 34.79 mm | 64.0 % |
| 2 | + affine + cutout | 40 | 14.13 px | 14.33 mm | 88.7 % |
| 3 | as run 2, heatmap 128 | 40 | 15.19 px | 15.26 mm | 87.8 % |
| 4 | **as run 2, longer schedule** | **100** | **10.78 px** | **11.21 mm** | **93.7 %** |
| — | stage 3, ArUco markers | — | ~0.5 px | 1.40 mm | 100 % |

Two changes account for everything, and neither touches the architecture:

**Augmentation** (run 1 → 2) cut keypoint error 2.4x and took the pass rate from
64 % to 88.7 %. The train/val loss ratio went from 72x to 0.95x — the overfitting
was eliminated, not merely reduced.

**Schedule length** (run 2 → 4) cut it a further 1.3x to 93.7 %. Run 2's
validation loss was still falling monotonically at epoch 39 and early stopping
never fired, so the 40-epoch budget was the binding constraint rather than the
model. Run 4 genuinely converged: validation loss flat at 0.00196–0.00198 across
the final ten epochs.

**Heatmap resolution** (run 2 → 3) changed nothing, at 33 % more compute.

The final train/val ratio is 0.90 — validation loss sits *below* training loss,
which is expected when augmentation applies only to the training split, and
confirms no overfitting even at 100 epochs.

Run 4 by occlusion:

| occlusion | n | ADD | pass | keypoint err |
|---|---|---|---|---|
| clean >95 % | 558 | 6.73 mm | 99.1 % | 7.07 px |
| light 70–95 % | 235 | 11.46 mm | 94.5 % | 11.17 px |
| heavy <70 % | 205 | 23.07 mm | 78.0 % | 20.45 px |

Heavy occlusion went from 22.9 % (run 1) to 78.0 % — the largest single gain, and
what cutout augmentation was chosen to target.

## What the measurements overturned

Three plausible hypotheses were tested and refuted. Each is recorded with its
mechanism, because the mechanism is the transferable part.

### 1. Corner identity was never the bottleneck

Bounding-box corners sit in empty space — no corner has local visual evidence,
and all eight look identical. The predicted failure mode was therefore *swaps*,
not drift, and the indicated fix would have been semantic keypoints on visible
geometry.

The error decomposition says otherwise. For each predicted keypoint, find the
nearest ground-truth keypoint; if it is not the intended one, that is a swap.

| run | localisation | identity | swap rate |
|---|---|---|---|
| 1 | 24.5 px | 9.3 px | 14.5 % |
| 2 | 13.2 px | 1.0 px | 3.3 % |
| 4 | 10.3 px | 0.5 px | 2.0 % |

Identity was always the minor term, and better localisation dissolved most of it
as a side effect — the swap rate fell 7x without any change to the keypoint
definition. Switching to semantic keypoints would have attacked the small term
and left the large one untouched.

### 2. RANSAC does not help

If swaps were clean exchanges of well-localised points, RANSAC should reject them
as outliers. A simulation at run 1's measured noise and swap rate predicted the
pass rate rising from 63 % to 83 %. On real data:

| | pass rate | mean ADD |
|---|---|---|
| least squares | 64.0 % | 34.79 mm |
| RANSAC, 40 px | 64.1 % | 36.31 mm |

No effect. The simulation assumed a swapped point was otherwise accurate. In
reality a swapped keypoint is *also* imprecise, so it is not a separable outlier
against a 24.5 px noise floor — the two error components are entangled.

The threshold must be set from the measured error, not from an idea of what is
accurate: an initial 12 px threshold, below the 24.5 px noise itself, rejected
good points and failed to solve on a third of the images.

### 3. Heatmap resolution was not the limit

At 64×64 each heatmap pixel is 10 image pixels, so quantisation looked like a
plausible precision floor. Doubling to 128×128 gave 15.19 px against 14.13 px —
no benefit, slightly worse, 33 % slower.

Run 1's 24.5 px was a *learning* limit, not a representational one. Run 4 later
reached 10.78 px at the original 64×64 resolution, well past where quantisation
was supposed to bind, which settles it.

**The first attempt at this ablation collapsed, and that was a bug of mine.**
Training at 128 with sigma left at 2.0 never learned: loss moved from 0.00505 to
0.00441 over ten epochs, pixel error stayed at 226 px, and confidence sat at
0.07–0.10 — indistinguishable from the initialisation value of
sigmoid(−2.19) = 0.10. The network had collapsed to predicting background
everywhere.

Sigma is expressed in *heatmap* pixels, so doubling the canvas while holding
sigma fixed shrinks the target's share of the map 4x:

| | positive fraction |
|---|---|
| 64, σ=2.0 | 0.513 % |
| 128, σ=2.0 | **0.128 %** |
| 128, σ=4.0 | 0.421 % |

`sigma_for()` scales sigma linearly with heatmap size, holding the target at
40 image px wide either way. The ablation above is the corrected run.

### And one that held

Occlusion dominates; view geometry does not.

| projected box aspect | n | ADD | pass |
|---|---|---|---|
| edge-on <0.35 | 101 | 14.28 mm | 93.1 % |
| oblique 0.35–0.6 | 352 | 10.37 mm | 95.2 % |
| broad >0.6 | 547 | 11.17 mm | 92.9 % |

Flat within noise, in every run. The near edge-on views flagged during label
verification are correct labels and are not harder in practice — which retires
the PnP-conditioning concern entirely.

Occlusion, by contrast, spans 6.73 mm to 23.07 mm and 99.1 % to 78.0 % in the
same run. It is the only difficulty axis that matters.

## Keypoints

The 8 corners of the drill's axis-aligned bounding box, in the object frame.
Mean pairwise separation 171.7 mm on a 274.0 mm object — wide spread, which
conditions the PnP solve well.

Bounding-box corners are the standard choice in the 6D pose literature (BB8,
YOLO-6D, PVNet), which makes the numbers comparable to published work.

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
distractors. One dataset therefore supports a clean-vs-occluded ablation at
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
  corners drifting far outside give the network a target with no visible evidence
  and nowhere to put the heatmap peak.
- `MIN_SEPARATION = 0.055 m` — stops distractors spawning inside the drill mesh,
  which renders as fused objects rather than as occlusion.

A bug worth recording: occlusion was originally measured on one distractor
arrangement and then the objects were re-randomized before the final render, so
`visible_frac` described a different image than the one saved. Fixed by storing
the measured poses and restoring exactly those.

Typical output: ~87 % mean visibility, ~44 % of images with some occlusion,
~20 % below 70 % visible. Reject rate around 9 %.

### Chunked generation

```bash
python 00_fetch_assets.py
python 01_render.py --n 2000 --start-idx 0
python 01_render.py --n 2000 --start-idx 2000
# ... 4000, 6000, 8000
python 02_build_dataset.py --val 1000 --test 1000
```

The random seed defaults to `--start-idx`, so chunks never repeat poses.
`02_build_dataset.py` checks for duplicate seeds and filenames, and verifies
every manifest entry has an image on disk.

The test split is **stratified by occlusion**, so all three splits carry the same
difficulty mix. A plain random split can hand you a test set that is easier than
training, making the final number meaningless.

## Label verification

```bash
python 01_render.py --n 12 --start-idx 0 --debug --out debug_data
python 03_verify_labels.py --data debug_data --n 12 --save
```

The debug flag draws the projected wireframe. **This step is not optional** —
both the missing absolute pixel floor and the over-permissive frame margin passed
every summary statistic and were caught only by looking at the images.

But the eye can also be wrong. Several debug images looked like the box had
missed the drill entirely; it had not. `03_verify_labels.py` overlays the drill's
true mesh vertices through the same stored pose. The 3D box encloses the mesh by
construction, so its projection **must** enclose the mesh's projection —
containment is a hard invariant, reported per image, and it has never failed.
What those images actually showed was heavy occlusion or a near edge-on view.

**Known limitation:** the verifier compares manifest against manifest and never
opens the image, so it cannot catch a swapped or overwritten image file. A
mistyped `--start-idx` once silently corrupted ~100 images while every check
still passed. Re-rendering the affected chunk fixes it, since the renderer is
deterministic from the seed.

## Model

ResNet-18 (ImageNet-pretrained) → deconv blocks → 8 heatmaps.

| space | size | holds |
|---|---|---|
| image | 640×640 | stored keypoints, stored `K` |
| input | 256×256 | what the network sees |
| heatmap | 64×64 | targets and predictions |

Predictions are scaled back to image space before PnP, because the stored camera
matrix describes image space. Scaling `K` instead would work but leaves two `K`s
in circulation.

### Heatmaps, not coordinate regression

Regressing (x, y) directly forces the network to encode position in the *values*
of its final activations — a poor match for a convolutional trunk, which is
translation-equivariant and naturally expresses "where" as "which unit fired".
Heatmaps keep the representation spatial until the argmax.

### Sub-pixel decoding

The argmax alone quantises to whole heatmap pixels — at 64×64 that is 10 px in
image space, enough to dominate the pose error outright.

The decoder fits a quadratic to the log of the three samples along each axis
around the peak. For a gaussian the log is exactly a parabola, so on an ideal
heatmap the fit is an exact inverse. Under noise it roughly halves the error:

| heatmap noise σ | argmax only | sub-pixel |
|---|---|---|
| 0.00 | 0.376 px | **0.000 px** |
| 0.05 | 0.511 px | **0.254 px** |
| 0.10 | 0.677 px | **0.453 px** |
| 0.20 | 1.056 px | **0.827 px** |

(heatmap pixels; ×10 for image space at 64×64)

### Loss

Masked MSE on sigmoid heatmaps with `pos_weight=8`. The target is ~99 % zeros, so
unweighted MSE lets the background dominate and the network converges to
predicting nothing everywhere — see the 128-resolution collapse above for what
that looks like when the balance tips too far.

Keypoints outside the heatmap get weight 0 rather than an all-zero target.
Forcing "no peak" on a corner that is merely off-frame teaches the network to
suppress evidence it can legitimately see near the border.

### Augmentation

Photometric — brightness, contrast, colour cast, sensor noise, defocus.

In-plane affine — scale, translate, rotation. Rotation is legitimate here: it is
exactly a roll of the camera about its optical axis, a real rigid transform, and
the keypoints rotate with it. The affine is verified to keep keypoints
synchronised with the image to 0.095 px, by transforming a synthetic image with
known blob positions and re-detecting them.

Cutout — random rectangles of random colour, targeting the measured weak spot.

**Horizontal flip is excluded.** A mirrored drill is a scene that cannot
physically exist, and the keypoint permutation it would require corresponds to no
rotation in SO(3). Flip is safe for human pose; it is not safe for 6D pose of a
chiral object.

Augmentation applies to training only. Validation and test run clean, so the
stored pose and `K` stay valid for PnP.

## Training

```bash
# run 4, the best configuration
python 04_train.py --epochs 100 --patience 15 --aug 1.0 --tag aug100

# ablations
python 04_train.py --epochs 40 --patience 8 --aug 1.0 --tag aug
python 04_train.py --epochs 40 --patience 8 --aug 1.0 --heatmap 128 --batch 16 --tag aug128

python 04_train.py --epochs 100 --aug 1.0 --tag aug100 --resume   # keep --epochs identical
```

Checkpoints every epoch to `checkpoints_<tag>/last.pt`, and to `best.pt` on
validation improvement. `--resume` restores model, optimiser, scheduler, epoch
and best score.

Validation tracks **mean keypoint error in image-space pixels**, not loss.
Heatmap MSE is not comparable across runs and does not track what matters
downstream; pixel error directly limits PnP accuracy.

Keep `--epochs` constant across resumes — `CosineAnnealingLR` uses it as `T_max`,
so changing it makes the learning rate jump. The script warns if you do.

Roughly 25–45 s/epoch at heatmap 64 on an RTX 4060 laptop with `--workers 4`,
60 s at 128. Run 4 is about 45 minutes end to end.

## Evaluation

```bash
python 05_evaluate.py --ckpt checkpoints_aug100/best.pt
```

Reads the heatmap size from the checkpoint. Reports ADD, the localisation/identity
split, and breakdowns by occlusion and projected box aspect.

## Status

- [x] Keypoint definition, verified visually and by mesh containment
- [x] Renderer with domain randomization and occlusion labelling
- [x] Chunked generation and stratified splits
- [x] ResNet-18 + deconv heatmap head with sub-pixel decoding
- [x] Training with per-epoch checkpointing, resume, early stopping
- [x] Inference → PnP → ADD against the stage 3 baseline
- [x] Augmentation ablation — 2.4x improvement
- [x] Heatmap resolution ablation — no benefit
- [x] Longer schedule — a further 1.3x, converged at 93.7 %

Every lever with evidence behind it has now been pulled. Remaining ideas, none
currently indicated by the measurements: a two-stage detect-then-crop pipeline
to raise effective resolution on the object, more training data, or a keypoint
set with local visual evidence — though the identity term is now 0.5 px, so the
last of those has almost nothing left to win.