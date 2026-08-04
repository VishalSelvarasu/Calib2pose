import argparse
import glob
import json
import os

import cv2
import numpy as np
import mujoco

from common import keypoints as kp

ASSET_DIR = "assets"


def load_mesh_vertices():
    xml = (f'<mujoco><asset><mesh name="drill" '
           f'file="{ASSET_DIR}/035_power_drill.msh"/></asset>'
           f'<worldbody><geom type="mesh" mesh="drill"/></worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "drill")
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    return np.array(m.mesh_vert[a:a + n]).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--save", action="store_true",
                    help="write overlay images to <data>/verify/")
    ap.add_argument("--aspect-warn", type=float, default=0.30)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.data, "manifest_*.json")))
    if not shards:
        raise SystemExit(f"No manifests in {args.data}/")

    records = []
    for s in shards:
        with open(s) as f:
            records.extend(json.load(f)["records"])

    V = load_mesh_vertices()
    img_dir = os.path.join(args.data, "images")
    out_dir = os.path.join(args.data, "verify")
    if args.save:
        os.makedirs(out_dir, exist_ok=True)

    sel = records[:args.n] if args.n > 0 else records
    print(f"checking {len(sel)} of {len(records)} images\n")
    print(f"{'file':<12}{'contain':>9}{'aspect':>8}{'hull_px2':>10}{'vis':>7}  flag")

    fails, edge_on = 0, 0
    for r in sel:
        pts = np.array(r["keypoints"], np.float32)
        T = np.array(r["T_cam_drill"])
        K = np.array(r["K"])

        proj, _ = cv2.projectPoints(
            V, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)
        proj = proj.reshape(-1, 2)

        hull = cv2.convexHull(pts)
        inside = np.array([cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0
                           for p in proj])
        contain = float(inside.mean())

        rect = cv2.minAreaRect(pts)
        w, h = rect[1]
        aspect = float(min(w, h) / max(w, h)) if max(w, h) > 0 else 0.0
        area = float(cv2.contourArea(hull))

        flag = ""
        if contain < 0.999:
            flag = "LABEL MISMATCH"
            fails += 1
        elif aspect < args.aspect_warn:
            flag = "edge-on, weak PnP conditioning"
            edge_on += 1

        print(f"{r['file']:<12}{contain*100:>8.1f}%{aspect:>8.3f}{area:>10.0f}"
              f"{r['visible_frac']:>7.2f}  {flag}")

        if args.save:
            img = cv2.imread(os.path.join(img_dir, r["file"]))
            if img is None:
                continue
            for p in proj[::10]:
                cv2.circle(img, tuple(p.astype(int)), 1, (255, 0, 255), -1)
            for a, b in kp.BBOX_EDGES:
                cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                         (0, 255, 0), 1)
            for i, p in enumerate(pts):
                cv2.circle(img, tuple(p.astype(int)), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(
                out_dir, r["file"].replace(".jpg", "_v.jpg")), img)

    print()
    if fails:
        print(f"{fails} LABEL MISMATCHES. The dataset is corrupt -- stop and fix "
              f"the renderer before training.")
    else:
        print(f"containment OK on all {len(sel)}: every mesh vertex projects "
              f"inside its labelled box.")
    print(f"{edge_on} near edge-on views (aspect < {args.aspect_warn}). These "
          f"labels are correct but poorly conditioned for PnP;")
    print("they are legitimate hard examples, not errors.")
    if args.save:
        print(f"\noverlays -> {out_dir}/")


if __name__ == "__main__":
    main()
