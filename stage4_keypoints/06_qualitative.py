
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
from model import KeypointNet, decode_heatmaps

GT_COLOUR = (90, 230, 120)      # BGR, green
PRED_COLOUR = (60, 130, 250)    # BGR, orange
MESH_CACHE = "drill_points.npy"
ASSET_DIR = "assets"


def load_mesh_points():
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
    return V


def draw_box(img, pts, colour, thickness=2):
    for a, b in kpdef.BBOX_EDGES:
        p, q = pts[a].astype(int), pts[b].astype(int)
        cv2.line(img, tuple(p), tuple(q), colour, thickness, cv2.LINE_AA)
    return img


def caption(img, lines, scale=0.44, pad=6, lh=17):
    """Solid banner across the top of a tile.

    Outlined text over a randomised background is unreadable at tile scale --
    the outline bleeds into the glyphs and the lines collide. A filled strip is
    plainer and legible on every background the renderer can produce.
    """
    h = pad * 2 + lh * len(lines)
    strip = (img[:h].astype(np.float32) * 0.25).astype(np.uint8)
    img[:h] = strip
    y = pad + 12
    for text, colour in lines:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    colour, 1, cv2.LINE_AA)
        y += lh
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="checkpoints_aug100/best.pt")
    ap.add_argument("--n", type=int, default=12, help="images in the grid")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--sort", default="spread",
                    choices=["spread", "best", "worst", "first"],
                    help="spread = span the ADD range, which is the honest "
                         "default; best/worst are for inspecting failures")
    ap.add_argument("--pool", type=int, default=300,
                    help="how many test images to score before selecting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mesh", action="store_true",
                    help="overlay the drill's true mesh vertices. The bounding "
                         "box is much larger than the visible drill because the "
                         "drill is L-shaped and its AABB is mostly empty; this "
                         "shows the box really does contain the object.")
    ap.add_argument("--out", default="results/qualitative.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    hs = ck.get("args", {}).get("heatmap", 64)
    model = KeypointNet(pretrained=False, heatmap_size=hs).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt} (epoch {ck['epoch']}, heatmap {hs})")

    ds = DrillKeypointDataset(args.data, args.split, augment=False,
                              heatmap_size=hs)
    # Sample the pool RANDOMLY, never as records[:N]. 02_build_dataset.py
    # assembles each split by iterating occlusion buckets, so the file is
    # ORDERED BY OCCLUSION -- taking the first N lands entirely inside the most
    # occluded bucket and produces a figure where every tile is a hard case.
    rng = np.random.default_rng(args.seed)
    if args.pool < len(ds.records):
        idx = rng.choice(len(ds.records), args.pool, replace=False)
        ds.records = [ds.records[i] for i in sorted(idx)]
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0,
                    collate_fn=collate)

    mesh = load_mesh_points()
    kp3d = kpdef.KEYPOINTS_BBOX.astype(np.float64)
    # float64 and C-contiguous: the .npy cache and the ::40 slice both leave a
    # non-contiguous view, which cv2.projectPoints rejects outright.
    mesh_sub = np.ascontiguousarray(mesh[::40], dtype=np.float64)
    scored = []

    with torch.no_grad():
        for batch in ld:
            coords, conf = decode_heatmaps(model(batch["image"].to(device)))
            pred_img = coords * (IMAGE_SIZE / hs)
            weight = batch["weight"].numpy()
            for b in range(len(pred_img)):
                rec = ds.records[int(batch["idx"][b])]
                use = weight[b] > 0
                if use.sum() < 4:
                    continue
                K = np.array(rec["K"])
                ok, rvec, tvec = cv2.solvePnP(
                    kp3d[use], pred_img[b][use].astype(np.float64), K, None,
                    flags=cv2.SOLVEPNP_EPNP)
                if not ok:
                    continue
                rvec, tvec = cv2.solvePnPRefineLM(
                    kp3d[use], pred_img[b][use].astype(np.float64), K, None,
                    rvec, tvec)
                T_est = np.eye(4)
                T_est[:3, :3] = cv2.Rodrigues(rvec)[0]
                T_est[:3, 3] = tvec.ravel()
                T_true = np.array(rec["T_cam_drill"])
                add = add_metric(mesh, T_est, T_true) * 1000

                proj, _ = cv2.projectPoints(kp3d, rvec, tvec, K, None)
                rv_t = cv2.Rodrigues(T_true[:3, :3])[0]
                mp, _ = cv2.projectPoints(
                    mesh_sub, rv_t, T_true[:3, 3], K, None)
                scored.append({
                    "file": rec["file"],
                    "gt": np.array(rec["keypoints"], np.float32),
                    "pred": proj.reshape(-1, 2),
                    "mesh": mp.reshape(-1, 2),
                    "add": add,
                    "vis": rec["visible_frac"],
                })

    if not scored:
        raise SystemExit("nothing scored -- check --data and --split")

    scored.sort(key=lambda r: r["add"])
    if args.sort == "best":
        pick = scored[:args.n]
    elif args.sort == "worst":
        pick = scored[-args.n:]
    elif args.sort == "first":
        pick = scored[:args.n]
    else:
        idx = np.linspace(0, len(scored) - 1, args.n).round().astype(int)
        pick = [scored[i] for i in idx]

    img_dir = os.path.join(args.data, "images")
    tiles = []
    for r in pick:
        img = cv2.imread(os.path.join(img_dir, r["file"]))
        if img is None:
            continue
        # Ground truth is drawn THICKER and underneath. When the prediction is
        # accurate the orange lands exactly on the green and hides it, which
        # makes a good result look like only one box was drawn. A wider green
        # line leaves a visible fringe either side.
        if args.mesh:
            for q in r["mesh"].astype(int):
                cv2.circle(img, tuple(q), 2, (255, 80, 220), -1)
        draw_box(img, r["gt"], GT_COLOUR, 5)
        draw_box(img, r["pred"], PRED_COLOUR, 2)
        img = cv2.resize(img, (args.tile, args.tile),
                         interpolation=cv2.INTER_AREA)
        ok = r["add"] < 27.4
        caption(img, [
            (f"ADD {r['add']:.1f} mm", (255, 255, 255)),
            (f"visible {r['vis']*100:.0f}%",
             (150, 230, 160) if ok else (110, 150, 250)),
        ])
        tiles.append(img)

    rows = int(np.ceil(len(tiles) / args.cols))
    while len(tiles) < rows * args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r * args.cols:(r + 1) * args.cols])
                      for r in range(rows)])

    legend = np.zeros((46, grid.shape[1], 3), np.uint8)
    cv2.line(legend, (14, 22), (44, 22), GT_COLOUR, 5, cv2.LINE_AA)
    cv2.putText(legend, "ground truth", (52, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.line(legend, (206, 22), (236, 22), PRED_COLOUR, 2, cv2.LINE_AA)
    cv2.putText(legend, "predicted (network keypoints -> solvePnP)",
                (244, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1,
                cv2.LINE_AA)
    grid = np.vstack([legend, grid])

    cv2.imwrite(args.out, grid)
    adds = np.array([r["add"] for r in pick])
    vis = np.array([r["vis"] for r in pick])
    pool_vis = np.array([r["vis"] for r in scored])
    print(f"\n{len(pick)} images, ADD {adds.min():.1f}-{adds.max():.1f} mm "
          f"(pool of {len(scored)}, sort={args.sort})")
    print(f"visibility  shown {vis.min()*100:.0f}-{vis.max()*100:.0f}%   "
          f"pool {pool_vis.min()*100:.0f}-{pool_vis.max()*100:.0f}%  "
          f"(clean >95%: {100*(pool_vis > 0.95).mean():.0f}% of pool)")
    print(f"wrote {args.out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
