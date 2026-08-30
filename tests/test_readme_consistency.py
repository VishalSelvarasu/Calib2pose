import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MESH_DIAMETER_MM = 226.2502759695053
ADD_THRESHOLD_MM = 0.1 * MESH_DIAMETER_MM

S4 = "stage4_keypoints/results"
S5 = "stage5_task_error/results"


def EDGE(r): return r["aspect"] < 0.35
def OBLIQUE(r): return 0.35 <= r["aspect"] <= 0.6
def BROAD(r): return r["aspect"] > 0.6
def CLEAN(r): return r["visible_frac"] > 0.95
def LIGHT(r): return 0.70 <= r["visible_frac"] <= 0.95
def HEAVY(r): return r["visible_frac"] < 0.70


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _json(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def rate(rel, where=None):
    rows = _json(rel)["rows"]
    if where:
        rows = [r for r in rows if where(r)]
    return 100.0 * sum(1 for r in rows if r["add_mm"] < ADD_THRESHOLD_MM) / len(rows)


def mean_add(rel, where=None):
    rows = _json(rel)["rows"]
    if where:
        rows = [r for r in rows if where(r)]
    return sum(r["add_mm"] for r in rows) / len(rows)


def count(rel, where):
    return sum(1 for r in _json(rel)["rows"] if where(r))


def within(rel, tol):
    rows = _json(rel)["rows"]
    return 100.0 * sum(1 for r in rows if r["placement_mm"] < tol) / len(rows)


def field(rel, key):
    return _json(rel)[key]


CLAIMS = [
    ("README.md", "stage 4 mean ADD", "11.21 mm",
     lambda: f"{mean_add(f'{S4}/eval_test_aug100.json'):.2f} mm"),
    ("README.md", "stage 4 pass rate", "90.2%",
     lambda: f"{rate(f'{S4}/eval_test_aug100.json'):.1f}%"),
    ("README.md", "stage 5 at 15 mm", "84.8%",
     lambda: f"{within(f'{S5}/task_stage4.json', 15.0):.1f}%"),
    ("README.md", "ADD-0.1d threshold", "22.6 mm",
     lambda: f"{ADD_THRESHOLD_MM:.1f} mm"),
    ("README.md", "mesh diameter", "226.3 mm",
     lambda: f"{MESH_DIAMETER_MM:.1f} mm"),

    ("stage3_pose/README.md", "marker ADD", "1.40 mm",
     lambda: f"{field('stage3_pose/results/pose_true_0.0px.json', 'add_mean_mm'):.2f} mm"),
    ("stage3_pose/README.md", "threshold", "22.6 mm",
     lambda: f"{field('stage3_pose/results/pose_true_0.0px.json', 'add_threshold_mm'):.1f} mm"),
    ("stage3_pose/README.md", "degenerate ADD", "81.40 mm",
     lambda: f"{field('stage3_pose/results/pose_estimated-degenerate_0.0px.json', 'add_mean_mm'):.2f} mm"),

    ("stage4_keypoints/README.md", "run 1 mean ADD", "34.79 mm",
     lambda: f"{mean_add(f'{S4}/eval_test_base.json'):.2f} mm"),
    ("stage4_keypoints/README.md", "run 1 pass", "55.5%",
     lambda: f"{rate(f'{S4}/eval_test_base.json'):.1f}%"),
    ("stage4_keypoints/README.md", "run 2 mean ADD", "14.33 mm",
     lambda: f"{mean_add(f'{S4}/eval_test_aug.json'):.2f} mm"),
    ("stage4_keypoints/README.md", "run 2 pass", "84.1%",
     lambda: f"{rate(f'{S4}/eval_test_aug.json'):.1f}%"),
    ("stage4_keypoints/README.md", "run 3 mean ADD", "15.26 mm",
     lambda: f"{mean_add(f'{S4}/eval_test_aug128.json'):.2f} mm"),
    ("stage4_keypoints/README.md", "run 3 pass", "83.1%",
     lambda: f"{rate(f'{S4}/eval_test_aug128.json'):.1f}%"),
    ("stage4_keypoints/README.md", "run 4 mean ADD", "11.21 mm",
     lambda: f"{mean_add(f'{S4}/eval_test_aug100.json'):.2f} mm"),
    ("stage4_keypoints/README.md", "run 4 pass", "90.2%",
     lambda: f"{rate(f'{S4}/eval_test_aug100.json'):.1f}%"),

    ("stage4_keypoints/README.md", "clean n", "| 560 |",
     lambda: f"| {count(f'{S4}/eval_test_aug100.json', CLEAN)} |"),
    ("stage4_keypoints/README.md", "clean pass", "98.2%",
     lambda: f"{rate(f'{S4}/eval_test_aug100.json', CLEAN):.1f}%"),
    ("stage4_keypoints/README.md", "light pass", "89.8%",
     lambda: f"{rate(f'{S4}/eval_test_aug100.json', LIGHT):.1f}%"),
    ("stage4_keypoints/README.md", "heavy pass", "68.8%",
     lambda: f"{rate(f'{S4}/eval_test_aug100.json', HEAVY):.1f}%"),

    ("stage4_keypoints/README.md", "edge-on mean ADD", "14.28 mm",
     lambda: f"{mean_add(f'{S4}/eval_test_aug100.json', EDGE):.2f} mm"),
    ("stage4_keypoints/README.md", "edge-on pass", "87.1%",
     lambda: f"{rate(f'{S4}/eval_test_aug100.json', EDGE):.1f}%"),
    ("stage4_keypoints/README.md", "oblique mean ADD", "10.37 mm",
     lambda: f"{mean_add(f'{S4}/eval_test_aug100.json', OBLIQUE):.2f} mm"),
    ("stage4_keypoints/README.md", "oblique pass", "91.2%",
     lambda: f"{rate(f'{S4}/eval_test_aug100.json', OBLIQUE):.1f}%"),
    ("stage4_keypoints/README.md", "broad mean ADD", "11.17 mm",
     lambda: f"{mean_add(f'{S4}/eval_test_aug100.json', BROAD):.2f} mm"),
    ("stage4_keypoints/README.md", "broad pass", "90.1%",
     lambda: f"{rate(f'{S4}/eval_test_aug100.json', BROAD):.1f}%"),

    ("stage4_keypoints/README.md", "no-oracle conf0.3 pass", "90.1%",
     lambda: f"{rate(f'{S4}/eval_test_aug100_noOracle_conf0.3.json'):.1f}%"),
    ("stage4_keypoints/README.md", "unfiltered pass", "90.0%",
     lambda: f"{rate(f'{S4}/eval_test_aug100_noOracle.json'):.1f}%"),
    ("stage4_keypoints/README.md", "no-oracle mean ADD", "11.37 mm",
     lambda: f"{mean_add(f'{S4}/eval_test_aug100_noOracle_conf0.3.json'):.2f} mm"),
    ("stage4_keypoints/README.md", "unfiltered mean ADD", "11.41 mm",
     lambda: f"{mean_add(f'{S4}/eval_test_aug100_noOracle.json'):.2f} mm"),

    ("stage5_task_error/README.md", "5 mm", "41.4%",
     lambda: f"{within(f'{S5}/task_stage4.json', 5.0):.1f}%"),
    ("stage5_task_error/README.md", "10 mm", "70.0%",
     lambda: f"{within(f'{S5}/task_stage4.json', 10.0):.1f}%"),
    ("stage5_task_error/README.md", "15 mm", "84.8%",
     lambda: f"{within(f'{S5}/task_stage4.json', 15.0):.1f}%"),
    ("stage5_task_error/README.md", "22.6 mm", "92.4%",
     lambda: f"{within(f'{S5}/task_stage4.json', ADD_THRESHOLD_MM):.1f}%"),
]


@pytest.mark.parametrize(
    "readme,label,quoted,recompute", CLAIMS,
    ids=[f"{c[0].split('/')[0]}::{c[1]}" for c in CLAIMS])
def test_readme_matches_artifact(readme, label, quoted, recompute):
    actual = recompute()
    assert actual == quoted, (
        f"{readme} quotes {label} as {quoted!r}, artifact gives {actual!r}")
    assert quoted in _read(readme), (
        f"{readme} no longer contains {quoted!r}; update the manifest")


def test_stage4_results_record_their_threshold():
    for fn in ["eval_test_base.json", "eval_test_aug.json",
               "eval_test_aug128.json", "eval_test_aug100.json",
               "eval_test_aug100_noOracle.json",
               "eval_test_aug100_noOracle_conf0.3.json"]:
        d = _json(f"{S4}/{fn}")
        assert "add_threshold_mm" in d and "metric_version" in d


def test_all_stage4_results_use_one_diameter():
    """The migration stamped 226.2502759695053; the evaluator once stamped the
    rounded 0.2263 constant. Both carried the same metric_version label."""
    seen = {}
    for fn in sorted(os.listdir(os.path.join(ROOT, S4))):
        if not fn.endswith(".json"):
            continue
        d = _json(f"{S4}/{fn}")
        if "diameter_mm" in d:
            seen.setdefault(round(d["diameter_mm"], 6), []).append(fn)
    assert len(seen) == 1, f"result files disagree on the ADD diameter: {seen}"
    assert abs(next(iter(seen)) - MESH_DIAMETER_MM) < 1e-6
