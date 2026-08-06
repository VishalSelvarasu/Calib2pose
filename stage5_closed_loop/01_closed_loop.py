import argparse
import json
import os

import numpy as np
import cv2
import mujoco

import arm_scene as a
import ik
from common import transforms as tf


def drill_visibility(scene):
    """Fraction of the frame occupied by the drill, via a segmentation pass.

    A cheap proxy for 'could a perception system see it from here'. On a real
    arm the wrist links sit in the eye-in-hand camera's field of view for many
    configurations, so this is not always high.
    """
    if scene.renderer is None:
        scene.renderer = mujoco.Renderer(scene.model, a.IMG_H, a.IMG_W)
    gid = -1
    for i in range(scene.model.ngeom):
        b = scene.model.geom_bodyid[i]
        if mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b) == "drill":
            gid = i
            break
    scene.renderer.enable_segmentation_rendering()
    scene.renderer.update_scene(scene.data, camera="eye")
    seg = scene.renderer.render()[:, :, 0]
    scene.renderer.disable_segmentation_rendering()
    return float((seg == gid).mean())


def find_observation_pose(scene, T_obj, rng, n_try=30):
    """Search flange poses whose camera both reaches and sees the object."""
    best_q, best_v = None, 0.0
    inv_fc = np.linalg.inv(a.gt_T_flange_cam())
    for _ in range(n_try):
        az = rng.uniform(0, 2 * np.pi)
        el = np.radians(rng.uniform(45, 85))
        r = rng.uniform(0.30, 0.50)
        eye = T_obj[:3, 3] + np.array([r * np.cos(el) * np.cos(az),
                                       r * np.cos(el) * np.sin(az),
                                       r * np.sin(el)])
        T_cam = tf.look_at_camera(eye, T_obj[:3, 3], rng.uniform(-180, 180))
        q, ok, _ = ik.solve_ik(scene, T_cam @ inv_fc, q_init=a.HOME_Q)
        if not ok:
            continue
        scene.set_q(q)
        v = drill_visibility(scene)
        if v > best_v:
            best_q, best_v = q, v
        if v > 0.08:
            break
    return best_q, best_v


def load_error_distribution(path, label):
    """Per-view (translation mm, rotation deg) errors from a stage's eval JSON."""
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found -- run {label}'s evaluation first.")
    with open(path) as f:
        d = json.load(f)
    rows = d.get("rows") or d.get("views") or []
    out = [(r["t_err_mm"], r["R_err_deg"]) for r in rows
           if r.get("t_err_mm") is not None and r.get("R_err_deg") is not None
           and np.isfinite(r["t_err_mm"]) and np.isfinite(r["R_err_deg"])]
    if not out:
        raise SystemExit(f"no per-view errors found in {path}")
    return np.array(out)


def perturb_pose(T, t_err_mm, R_err_deg, rng):
    """Apply an error of the given magnitude in a uniformly random direction."""
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    dR = cv2.Rodrigues(axis * np.radians(R_err_deg))[0]
    out = T.copy()
    out[:3, :3] = dR @ T[:3, :3]
    out[:3, 3] = T[:3, 3] + d * (t_err_mm / 1000.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--source", default="stage4",
                    choices=["perfect", "stage3", "stage4"])
    ap.add_argument("--stage3-json",
                    default="../stage3_pose/results/pose_true_0.0px.json")
    ap.add_argument("--stage4-json",
                    default="../stage4_keypoints/results/eval_test_aug100.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grasp-tol-mm", type=float, default=15.0,
                    help="how far the flange may land and still grasp; 15 mm is "
                         "a representative parallel-jaw margin")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scene = a.ArmScene()
    rng = np.random.default_rng(args.seed)
    T_obj_true = scene.true_T_base_drill()

    errs = None
    if args.source == "stage3":
        errs = load_error_distribution(args.stage3_json, "stage 3")
    elif args.source == "stage4":
        errs = load_error_distribution(args.stage4_json, "stage 4")

    print(f"pose source     : {args.source}")
    if errs is not None:
        print(f"error samples   : {len(errs)}   median "
              f"{np.median(errs[:, 0]):.2f} mm / {np.median(errs[:, 1]):.2f} deg")
    print(f"grasp tolerance : {args.grasp_tol_mm} mm")

    # Reference: where the flange lands given the TRUE pose. Deterministic, so
    # it is computed once and every trial is measured against it.
    T_goal_true = ik.grasp_pose_from_object(T_obj_true)
    q_ref, ok_ref, e_ref = ik.solve_ik(scene, T_goal_true, q_init=a.HOME_Q)
    if not ok_ref:
        raise SystemExit("IK failed on the true grasp pose; move the object.")
    scene.set_q(q_ref)
    T_flange_ref = scene.flange_pose()
    print(f"reference grasp : flange at "
          f"{np.round(T_flange_ref[:3, 3]*1000, 1)} mm, "
          f"IK residual {np.linalg.norm(e_ref[:3])*1000:.3f} mm")

    q_obs, vis = find_observation_pose(scene, T_obj_true, rng)
    if q_obs is None:
        print("observation pose: none reachable")
        vis = 0.0
    else:
        print(f"observation pose: drill fills {vis*100:.1f} % of the frame"
              f"{'  (heavily self-occluded by the wrist)' if vis < 0.05 else ''}")
    print()

    rows = []
    for i in range(args.trials):
        if errs is None:
            T_obj_est = T_obj_true
        else:
            t_mm, R_deg = errs[rng.integers(len(errs))]
            T_obj_est = perturb_pose(T_obj_true, t_mm, R_deg, rng)

        T_goal = ik.grasp_pose_from_object(T_obj_est)
        q, ok, e = ik.solve_ik(scene, T_goal, q_init=q_ref)
        scene.set_q(q)
        T_flange = scene.flange_pose()

        place_mm = float(np.linalg.norm(
            T_flange[:3, 3] - T_flange_ref[:3, 3]) * 1000)
        c = (np.trace(T_flange_ref[:3, :3].T @ T_flange[:3, :3]) - 1) / 2
        rows.append({
            "trial": i,
            "ik_converged": bool(ok),
            "ik_residual_mm": float(np.linalg.norm(e[:3]) * 1000),
            "placement_mm": place_mm,
            "placement_deg": float(np.degrees(np.arccos(np.clip(c, -1, 1)))),
        })

    p = np.array([r["placement_mm"] for r in rows])
    pd = np.array([r["placement_deg"] for r in rows])
    ikr = np.array([r["ik_residual_mm"] for r in rows])
    conv = np.array([r["ik_converged"] for r in rows])
    success = float((p < args.grasp_tol_mm).mean() * 100)

    print(f"{'metric':<20}{'mean':>10}{'median':>10}{'p90':>10}{'max':>10}")
    print(f"{'placement error':<20}{p.mean():>8.2f}mm{np.median(p):>8.2f}mm"
          f"{np.percentile(p, 90):>8.2f}mm{p.max():>8.2f}mm")
    print(f"{'placement rotation':<20}{pd.mean():>9.2f}d{np.median(pd):>9.2f}d"
          f"{np.percentile(pd, 90):>9.2f}d{pd.max():>9.2f}d")
    print(f"{'IK residual':<20}{ikr.mean():>8.3f}mm{np.median(ikr):>8.3f}mm"
          f"{np.percentile(ikr, 90):>8.3f}mm{ikr.max():>8.3f}mm")
    print(f"\nIK converged  : {100*conv.mean():.1f} %")
    print(
        f"grasp success : {success:.1f} %   (placement < {args.grasp_tol_mm} mm)")

    out = {"source": args.source, "trials": args.trials,
           "grasp_tol_mm": args.grasp_tol_mm,
           "placement_mean_mm": float(p.mean()),
           "placement_median_mm": float(np.median(p)),
           "placement_p90_mm": float(np.percentile(p, 90)),
           "placement_max_mm": float(p.max()),
           "ik_residual_mean_mm": float(ikr.mean()),
           "ik_converged_pct": float(100 * conv.mean()),
           "grasp_success_pct": success,
           "observation_visibility": vis,
           "rows": rows}
    path = os.path.join(args.out, f"closed_loop_{args.source}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
