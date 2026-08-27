import json
import os
import sys

import numpy as np

MESH_DIAMETER_MM = 226.2502759695053
ADD_THRESHOLD_MM = 0.1 * MESH_DIAMETER_MM
METRIC_VERSION = "ADD-0.1d_mesh_diameter_226.25mm"
TOLERANCES = [5.0, 10.0, 15.0, ADD_THRESHOLD_MM]

METRIC_NOTE = (
    "fraction of trials where IK converged and the flange landed within the "
    "tolerance of its true-pose reference position. Not grasp success: no "
    "contact, closure, friction or lift is simulated."
)


def stamp_stage3(root):
    d = os.path.join(root, "stage3_pose", "results")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(d, fn)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        was = data.get("diameter_mm")
        data["diameter_mm"] = MESH_DIAMETER_MM
        data["add_threshold_mm"] = ADD_THRESHOLD_MM
        data["metric_version"] = METRIC_VERSION
        rows = data.get("rows") or data.get("views") or []
        if rows and "add_mm" in rows[0]:
            data["add_pass_pct"] = 100.0 * sum(
                1 for r in rows if r["add_mm"] < ADD_THRESHOLD_MM) / len(rows)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        stale = "" if abs(was - MESH_DIAMETER_MM) < 0.01 else "   <- was stale"
        print(f"  {fn}: {was:.2f} -> {MESH_DIAMETER_MM:.2f} mm, "
              f"pass {data.get('add_pass_pct')}%{stale}")


def stamp_stage4(root):
    d = os.path.join(root, "stage4_keypoints", "results")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(d, fn)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        data["diameter_mm"] = MESH_DIAMETER_MM
        data["add_threshold_mm"] = ADD_THRESHOLD_MM
        data["metric_version"] = METRIC_VERSION
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  {fn}: stamped")


def migrate_stage5(root):
    d = os.path.join(root, "stage5_task_error", "results")
    pairs = [("closed_loop_perfect.json", "task_perfect.json"),
             ("closed_loop_stage3.json", "task_stage3.json"),
             ("closed_loop_stage4.json", "task_stage4.json")]
    for src, dst in pairs:
        sp = os.path.join(d, src)
        if not os.path.exists(sp):
            print(f"  {src}: not present, skipping")
            continue
        with open(sp, encoding="utf-8") as f:
            data = json.load(f)
        rows = data["rows"]
        p = np.array([r["placement_mm"] for r in rows])
        conv = np.array([r["ik_converged"] for r in rows])
        curve = {f"{t:g}": float(((p < t) & conv).mean() * 100)
                 for t in sorted(TOLERANCES)}
        out = {
            "source": data["source"],
            "trials": data["trials"],
            "seed": data.get("seed", 0),
            "metric": "within_placement_tolerance_pct",
            "metric_note": METRIC_NOTE,
            "within_tolerance_pct": curve,
            "placement_mean_mm": data["placement_mean_mm"],
            "placement_median_mm": data["placement_median_mm"],
            "placement_p90_mm": data["placement_p90_mm"],
            "placement_max_mm": data["placement_max_mm"],
            "ik_residual_mean_mm": data["ik_residual_mean_mm"],
            "ik_converged_pct": data["ik_converged_pct"],
            "observation_visibility": data["observation_visibility"],
            "rows": rows,
        }
        with open(os.path.join(d, dst), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        os.remove(sp)
        pretty = "  ".join(f"{k}mm {v:.1f}%" for k, v in curve.items())
        print(f"  {src} -> {dst}   {pretty}")


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    if not os.path.exists(os.path.join(root, "README.md")):
        raise SystemExit(f"{root} is not the repository root")
    print("stage 3")
    stamp_stage3(root)
    print("\nstage 4")
    stamp_stage4(root)
    print("\nstage 5")
    migrate_stage5(root)
    print("\ndone")


if __name__ == "__main__":
    main()
