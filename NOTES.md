## 2026-07-31
- Stages 1-3 shipped. Baseline ADD 1.40 mm, 100% pass (274 mm object, 0.5% of diameter)
- Degenerate hand-eye propagates to 81.40 mm ADD, 0% pass, with every stage-3
  signal still green — the project's central result
- Env: Python 3.14, OpenCV 5.0 (calibrateHandEye removed, own solvers in stage 2)
- Debt: board_config.py + transform helpers duplicated across 3 stages -> common/
- Pending: real-webcam calibration (needs printed board)
- Next: stage 4, synthetic keypoint training. 40-80 hrs. After the exam.

## 2026-08-04
- common/ added: transforms, metrics, bbox keypoint definition (8 AABB corners,
  mean spread 171.7 mm, ordering = x/y/z min-max product). Verified visually.
- Renderer built: domain randomization, 0-4 YCB distractors, occlusion labelled
  per image via segmentation pass. 10k images @ 640, split 8k/1k/1k stratified.
- Two label bugs caught by LOOKING at debug images, invisible in statistics:
  (a) MIN_VISIBLE_FRAC is a ratio -- a distant drill 60% occluded passed at a
      few hundred px. Added MIN_VISIBLE_PX = 6000 absolute floor.
  (b) frame margin was +-40 px, letting corners float into empty space.
      Tightened to 15 px, max 2 corners outside.
- Also: occlusion was measured on one distractor arrangement then the objects
  were re-randomised before the real render, so every visible_frac described a
  different image. Fixed by storing and restoring the measured poses.
- 03_verify_labels.py checks containment (mesh projects inside box hull) -- a
  hard invariant. But it compares manifest against manifest and never opens the
  image, so it CANNOT catch a swapped image file. A mistyped --start-idx 600
  silently corrupted ~100 images while every check still passed.
- Stages 2/3 keep their own transform copies; not migrating working code.