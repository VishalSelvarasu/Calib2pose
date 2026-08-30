import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(stage):
    path = os.path.join(ROOT, stage)
    sys.path.insert(0, path)
    try:
        import importlib
        import board_config
        importlib.reload(board_config)
        return board_config
    finally:
        sys.path.remove(path)


PTS = np.array([[10.5, 20.5], [30.5, 40.5], [50.5, 60.5]], dtype=np.float32)
IDS = np.array([3, 7, 11], dtype=np.int32)


@pytest.mark.parametrize("stage", ["stage1_intrinsics", "stage2_handeye"])
def test_both_conventions_normalise_identically(stage):
    bc = _load(stage)
    c4, i4 = bc.normalize_charuco(PTS.reshape(-1, 1, 2), IDS.reshape(-1, 1))
    c5, i5 = bc.normalize_charuco(PTS.reshape(-1, 2), IDS.reshape(-1))
    assert c4.shape == (3, 1, 2) and c5.shape == (3, 1, 2)
    assert i4.shape == (3, 1) and i5.shape == (3, 1)
    np.testing.assert_allclose(c4, c5)
    np.testing.assert_array_equal(i4, i5)


@pytest.mark.parametrize("stage", ["stage1_intrinsics", "stage2_handeye"])
def test_indexing_yields_2d_points(stage):
    """corners[k, 0] must be an (x, y) pair, not a scalar x."""
    bc = _load(stage)
    for corners, ids in [(PTS.reshape(-1, 1, 2), IDS.reshape(-1, 1)),
                         (PTS.reshape(-1, 2), IDS.reshape(-1))]:
        c, i = bc.normalize_charuco(corners, ids)
        for k in range(len(i)):
            assert c[k, 0].shape == (2,), f"{stage}: corners[{k}, 0] is scalar"
        np.testing.assert_allclose(c[1, 0], [30.5, 40.5])


@pytest.mark.parametrize("stage", ["stage1_intrinsics", "stage2_handeye"])
def test_dtypes_are_normalised(stage):
    bc = _load(stage)
    c, i = bc.normalize_charuco(PTS.astype(np.float64), IDS.astype(np.int64))
    assert c.dtype == np.float32 and i.dtype == np.int32


@pytest.mark.parametrize("stage", ["stage1_intrinsics", "stage2_handeye"])
def test_single_detection(stage):
    bc = _load(stage)
    c, i = bc.normalize_charuco(PTS[:1].reshape(-1, 2), IDS[:1].reshape(-1))
    assert c.shape == (1, 1, 2) and i.shape == (1, 1)


@pytest.mark.parametrize("stage", ["stage1_intrinsics", "stage2_handeye"])
def test_empty_detection(stage):
    bc = _load(stage)
    assert bc.normalize_charuco(None, None) == (None, None)
    assert bc.normalize_charuco(
        np.empty((0, 2)), np.empty((0,))) == (None, None)


@pytest.mark.parametrize("stage", ["stage1_intrinsics", "stage2_handeye"])
def test_detect_helper_exists(stage):
    """The Stage 1 README claims board_config.detect() normalises shapes."""
    bc = _load(stage)
    assert hasattr(bc, "detect") and hasattr(bc, "normalize_charuco")
