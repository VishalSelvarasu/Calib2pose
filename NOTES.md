## 2026-07-31
- Stage 1 synthetic validation: fx err 0.10%, k1 err 0.0004, split-half 0.18%
- Frontal-only control run reproduces the failure: RMS 0.2089 (lower!),
  fx err 1.95%, k2 sign-flipped +0.627 vs true -0.226
- Env: Python 3.14 venv, OpenCV 5.0.0 — shape change handled in bc.detect()
- Next: print board (2 copies), measure square, set SQUARE_LENGTH_MM