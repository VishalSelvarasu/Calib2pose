import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.keypoints import BBOX_MIN, BBOX_MAX, KEYPOINTS_BBOX  # noqa: E402

MESH_DIAMETER_MM = 226.2502759695053
AABB_DIAGONAL_MM = 274.0123867988586


def test_aabb_diagonal_is_not_the_mesh_diameter():
    diag = float(np.linalg.norm(BBOX_MAX - BBOX_MIN)) * 1000
    assert abs(diag - AABB_DIAGONAL_MM) < 0.05
    assert abs(diag - MESH_DIAMETER_MM) > 40


def test_add_threshold_uses_the_mesh_diameter():
    assert abs(0.1 * MESH_DIAMETER_MM - 22.625) < 1e-3
    assert abs(0.1 * AABB_DIAGONAL_MM - 27.401) < 1e-3


def test_evaluate_script_uses_the_mesh_diameter():
    src = open(os.path.join(ROOT, "stage4_keypoints/05_evaluate.py"),
               encoding="utf-8").read()
    assert "DRILL_DIAMETER_M = 0.2263" in src


def test_keypoint_separation_is_over_unique_pairs():
    D = np.linalg.norm(
        KEYPOINTS_BBOX[:, None] - KEYPOINTS_BBOX[None, :], axis=-1) * 1000
    iu = np.triu_indices(len(KEYPOINTS_BBOX), 1)
    assert abs(D[iu].mean() - 196.25) < 0.05   # 28 unique pairs
    assert abs(D.mean() - 171.72) < 0.05       # full 8x8, includes 8 zeros

    doc = open(os.path.join(ROOT, "common/keypoints.py"),
               encoding="utf-8").read()
    assert "196.2" in doc
    for bad in ["separation is 171.7", "separation of 171.7", "171.7 mm on a"]:
        assert bad not in doc


def test_keypoint_ordering_is_the_min_max_product():
    for k, bits in enumerate(
            [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]):
        expected = np.array([
            (BBOX_MAX if bits[0] else BBOX_MIN)[0],
            (BBOX_MAX if bits[1] else BBOX_MIN)[1],
            (BBOX_MAX if bits[2] else BBOX_MIN)[2]])
        np.testing.assert_allclose(KEYPOINTS_BBOX[k], expected)


def test_common_copies_have_not_diverged():
    for name in ["keypoints.py", "metrics.py", "transforms.py"]:
        digests = set()
        for base in ["common", "stage4_keypoints/common",
                     "stage5_task_error/common"]:
            p = os.path.join(ROOT, base, name)
            if os.path.exists(p):
                digests.add(hashlib.md5(open(p, "rb").read()).hexdigest())
        assert len(digests) == 1, f"copies of {name} differ"
