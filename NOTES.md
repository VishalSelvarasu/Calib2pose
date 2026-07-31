## 2026-07-31
- Stages 1-3 shipped. Baseline ADD 1.40 mm, 100% pass (274 mm object, 0.5% of diameter)
- Degenerate hand-eye propagates to 81.40 mm ADD, 0% pass, with every stage-3
  signal still green — the project's central result
- Env: Python 3.14, OpenCV 5.0 (calibrateHandEye removed, own solvers in stage 2)
- Debt: board_config.py + transform helpers duplicated across 3 stages -> common/
- Pending: real-webcam calibration (needs printed board)
- Next: stage 4, synthetic keypoint training. 40-80 hrs. After the exam.