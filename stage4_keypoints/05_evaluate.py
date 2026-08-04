import argparse
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import keypoints as kpdef
from common.metrics import add_metric
from dataset import DrillKeypointDataset, collate, IMAGE_SIZE
from model import KeypointNet, decode_heatmaps, HEATMAP_SIZE

DRILL_DIAMETER_M = 0.2740
ADD_THRESHOLD_M = 0.1 * DRILL_DIAMETER_M
ASSET_DIR = "assets"


MESH_CACHE = "drill_points.npy"


def load_mesh_points():
    """The drill's 8945 mesh vertices, for the ADD metric.

    Cached to .npy so evaluation never imports MuJoCo. Importing MuJoCo and
    torch in the same process segfaults on some platforms (both load their own
    OpenGL/CUDA runtimes), and evaluation has no need for a simulator.
    """
    if os.path.exists(MESH_CACHE):
        return np.load(MESH_CACHE)

    import mujoco
    xml = (f'<mujoco><asset><mesh name="drill" '
           f'file="{ASSET_DIR}/035_power_drill.msh"/></asset>'
           f'<worldbody><geom type="mesh" mesh="drill"/></worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "drill")
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    V = np.array(m.mesh_vert[a:a + n]).reshape(-1, 3)
    np.save(MESH_CACHE, V)
    print(f"cached {len(V)} mesh vertices -> {MESH_CACHE}")
    return V


def swap_analysis(pred, gt, valid):
    """Per-keypoint: is the prediction closest to its OWN ground truth?

    Returns (n_swapped, n_valid, assigned_error). `assigned_error` is the error
    measured against the nearest ground-truth keypoint rather than the intended
    one -- if that is small while the intended-target error is large, the
    network localised a corner correctly and simply labelled it wrong.
    """
    d = np.linalg.norm(pred[:, None, :] - gt[None, :, :],
                       axis=2)   # (K_pred, K_gt)
    nearest = d.argmin(axis=1)
    swapped = (nearest != np.arange(len(pred))) & valid
    assigned = d[np.arange(len(pred)), nearest]
    return int(swapped.sum()), int(valid.sum()), assigned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="drop keypoints below this heatmap confidence before PnP")
    ap.add_argument("--ransac", action="store_true",
                    help="use solvePnPRansac, which rejects swapped keypoints as "
                         "outliers instead of fitting to them")
    ap.add_argument("--ransac-px", type=float, default=40.0,
                    help="RANSAC reprojection threshold, image px. Must be set "
                         "relative to the MEASURED keypoint error, not to an "
                         "idea of what is 'accurate'. At 24.5 px keypoint noise "
                         "a 12 px threshold rejects good points and fails to "
                         "solve; 1.5-2x the noise works.")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = KeypointNet(pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(
        f"loaded {args.ckpt} (epoch {ck['epoch']}, val {ck.get('best', 0):.2f} px)")

    ds = DrillKeypointDataset(args.data, args.split, augment=False)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, collate_fn=collate)
    print(f"{args.split}: {len(ds)} images")

    model_pts = load_mesh_points()
    kp3d = kpdef.KEYPOINTS_BBOX.astype(np.float64)

    rows = []
    n_swap_tot, n_valid_tot = 0, 0
    with torch.no_grad():
        for batch in ld:
            logits = model(batch["image"].to(device))
            coords, conf = decode_heatmaps(logits)
            pred_img = coords * (IMAGE_SIZE / HEATMAP_SIZE)
            gt_img = batch["kps_img"].numpy()
            weight = batch["weight"].numpy()

            for b in range(len(pred_img)):
                rec = ds.records[int(batch["idx"][b])]
                valid = weight[b] > 0
                use = valid & (conf[b] >= args.min_conf)

                kp_err = np.linalg.norm(pred_img[b] - gt_img[b], axis=1)
                ns, nv, assigned = swap_analysis(pred_img[b], gt_img[b], valid)
                n_swap_tot += ns
                n_valid_tot += nv

                row = {
                    "file": rec["file"],
                    "visible_frac": rec["visible_frac"],
                    "kp_err_mean": float(kp_err[valid].mean()) if valid.any() else np.nan,
                    "kp_err_assigned": float(assigned[valid].mean()) if valid.any() else np.nan,
                    "n_swapped": ns,
                    "n_valid": nv,
                    "conf_mean": float(conf[b][valid].mean()) if valid.any() else 0.0,
                }

                # projected box aspect, as a conditioning proxy
                g = gt_img[b].astype(np.float32)
                (_, (w, h), _) = cv2.minAreaRect(g)
                row["aspect"] = float(
                    min(w, h) / max(w, h)) if max(w, h) > 0 else 0.0

                if use.sum() >= 4:
                    K = np.array(rec["K"])
                    op = kp3d[use]
                    ip = pred_img[b][use].astype(np.float64)

                    if args.ransac:
                        # A swapped keypoint is a gross outlier, not noise.
                        # Least-squares PnP spreads its error across the whole
                        # fit; RANSAC discards it. With ~14 % of keypoints
                        # swapped, roughly one per image is an outlier.
                        ok, rvec, tvec, inl = cv2.solvePnPRansac(
                            op, ip, K, None, flags=cv2.SOLVEPNP_EPNP,
                            reprojectionError=args.ransac_px,
                            iterationsCount=200, confidence=0.999)
                        row["n_inliers"] = 0 if inl is None else int(len(inl))
                        if ok and inl is not None and len(inl) >= 4:
                            op, ip = op[inl.ravel()], ip[inl.ravel()]
                        elif not ok:
                            rows.append(row)
                            continue
                    else:
                        ok, rvec, tvec = cv2.solvePnP(
                            op, ip, K, None, flags=cv2.SOLVEPNP_EPNP)
                    if ok:
                        rvec, tvec = cv2.solvePnPRefineLM(op, ip, K, None,
                                                          rvec, tvec)
                        T_est = np.eye(4)
                        T_est[:3, :3] = cv2.Rodrigues(rvec)[0]
                        T_est[:3, 3] = tvec.ravel()
                        T_true = np.array(rec["T_cam_drill"])
                        row["add_mm"] = add_metric(
                            model_pts, T_est, T_true) * 1000
                        row["t_err_mm"] = float(
                            np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1000)
                        c = (np.trace(T_true[:3, :3].T @
                             T_est[:3, :3]) - 1) / 2
                        row["R_err_deg"] = float(
                            np.degrees(np.arccos(np.clip(c, -1, 1))))
                rows.append(row)

    add = np.array([r.get("add_mm", np.nan) for r in rows], float)
    ok = ~np.isnan(add)
    kpe = np.array([r["kp_err_mean"] for r in rows])
    kpa = np.array([r["kp_err_assigned"] for r in rows])
    vf = np.array([r["visible_frac"] for r in rows])
    asp = np.array([r["aspect"] for r in rows])
    thr = ADD_THRESHOLD_M * 1000

    print(f"\n--- keypoints ---------------------------------------------")
    print(f"mean error            {np.nanmean(kpe):8.2f} px")
    print(f"median error          {np.nanmedian(kpe):8.2f} px")
    print(f"mean if reassigned    {np.nanmean(kpa):8.2f} px   "
          f"(measured against the NEAREST gt keypoint)")
    print(f"swapped keypoints     {n_swap_tot}/{n_valid_tot} "
          f"({100*n_swap_tot/max(n_valid_tot, 1):.1f} %)")

    gap = np.nanmean(kpe) - np.nanmean(kpa)
    print(f"\n  Of the {np.nanmean(kpe):.1f} px mean error, {gap:.1f} px is "
          f"attributable to corner identity")
    print(f"  and {np.nanmean(kpa):.1f} px to localisation precision.")

    print(f"\n--- pose --------------------------------------------------")
    print(f"PnP solved on {ok.sum()}/{len(rows)} images")
    if ok.any():
        a = add[ok]
        print(f"{'metric':<14}{'mean':>10}{'median':>10}{'p90':>10}")
        print(
            f"{'ADD':<14}{a.mean():>8.2f}mm{np.median(a):>8.2f}mm{np.percentile(a, 90):>8.2f}mm")
        te = np.array([r.get("t_err_mm", np.nan) for r in rows])[ok]
        re = np.array([r.get("R_err_deg", np.nan) for r in rows])[ok]
        print(f"{'translation':<14}{np.nanmean(te):>8.2f}mm{np.nanmedian(te):>8.2f}mm"
              f"{np.nanpercentile(te, 90):>8.2f}mm")
        print(f"{'rotation':<14}{np.nanmean(re):>9.2f}d{np.nanmedian(re):>9.2f}d"
              f"{np.nanpercentile(re, 90):>9.2f}d")
        print(f"\nADD-0.1d ({thr:.1f} mm) pass rate: "
              f"{100*(a < thr).mean():.1f} %   ({(a < thr).sum()}/{ok.sum()})")
        print(f"stage 3 marker baseline          : 1.40 mm mean ADD, 100.0 %")

        print(f"\n--- by occlusion ------------------------------------------")
        for lo, hi, name in [(0.95, 1.01, "clean   >95%"),
                             (0.70, 0.95, "light 70-95%"),
                             (0.00, 0.70, "heavy   <70%")]:
            m = ok & (vf >= lo) & (vf < hi)
            if m.sum():
                print(f"{name}  n={m.sum():4d}  ADD {add[m].mean():7.2f} mm  "
                      f"pass {100*(add[m] < thr).mean():5.1f} %  "
                      f"kp {kpe[m].mean():6.2f} px")

        print(f"\n--- by projected box aspect (PnP conditioning) ------------")
        for lo, hi, name in [(0.0, 0.35, "edge-on <0.35"),
                             (0.35, 0.6, "oblique .35-.6"),
                             (0.6, 1.01, "broad    >0.6")]:
            m = ok & (asp >= lo) & (asp < hi)
            if m.sum():
                print(f"{name}  n={m.sum():4d}  ADD {add[m].mean():7.2f} mm  "
                      f"pass {100*(add[m] < thr).mean():5.1f} %  "
                      f"kp {kpe[m].mean():6.2f} px")

    out = {
        "ckpt": args.ckpt, "split": args.split, "n": len(rows),
        "kp_err_mean_px": float(np.nanmean(kpe)),
        "kp_err_median_px": float(np.nanmedian(kpe)),
        "kp_err_assigned_px": float(np.nanmean(kpa)),
        "swap_rate_pct": 100 * n_swap_tot / max(n_valid_tot, 1),
        "add_mean_mm": float(add[ok].mean()) if ok.any() else None,
        "add_median_mm": float(np.median(add[ok])) if ok.any() else None,
        "pass_rate_pct": float(100 * (add[ok] < thr).mean()) if ok.any() else None,
        "pnp_solved": int(ok.sum()),
        "rows": rows,
    }
    path = os.path.join(args.out, f"eval_{args.split}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
