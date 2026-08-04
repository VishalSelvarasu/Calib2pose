## 2026-07-31
- Stages 1-3 shipped. Baseline ADD 1.40 mm, 100% pass (274 mm object, 0.5% of diameter)
- Degenerate hand-eye propagates to 81.40 mm ADD, 0% pass, with every stage-3
  signal still green — the project's central result
- Env: Python 3.14, OpenCV 5.0 (calibrateHandEye removed, own solvers in stage 2)
- Debt: board_config.py + transform helpers duplicated across 3 stages -> common/
- Pending: real-webcam calibration (needs printed board)
- Next: stage 4, synthetic keypoint training. 40-80 hrs. After the exam.

## 2026-08-05
- Stage 4 trained: 60 epochs, best val 32.95 px keypoint error (epoch 33)
- Overfit hard: val loss min at epoch 13 (0.00735), train/val ratio 72x by end.
  px error flat 33-35 from epoch 13 on, so best.pt is arbitrary within noise.
- Eval on test (1000): ADD 34.79 mm mean, 19.43 median, 64.0% ADD-0.1d pass
  vs stage 3 marker baseline 1.40 mm / 100%
- DIAGNOSIS: precision-limited, not identity-limited. 24.5 px of 33.8 px is
  localisation; only 9.3 px from corner swaps (14.5% swap rate).
  -> semantic keypoints would attack the SMALL term. Not indicated.
- RANSAC (40 px thr): 64.1% pass, mean ADD 36.31 mm. NO improvement.
  Simulation predicted 83% — it assumed swaps were clean exchanges of
  well-localised points. Real swapped points are also imprecise, so they are
  not separable outliers against a 24.5 px noise floor.
- Occlusion dominates: clean 80%, light 61%, heavy 23% pass
- View geometry does NOT matter: aspect bands 63.4/60.8/66.4% — flat
- Next: retrain with stronger augmentation + early stop ~epoch 15.
  Generalisation is the problem, not architecture or PnP.