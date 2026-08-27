import argparse
import json
import os

import numpy as np
import cv2
import mujoco

from common import transforms as tf
from common import keypoints as kp

ASSET_DIR = "assets"
IMG_SIZE = 640
FOVY_BASE = 45.0

DRILL = "035_power_drill"
DISTRACTORS = [
    "002_master_chef_can", "003_cracker_box", "004_sugar_box",
    "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can",
    "008_pudding_box", "009_gelatin_box", "010_potted_meat_can",
]
MAX_DISTRACTORS = 4
MIN_VISIBLE_FRAC = 0.35   # ratio: fraction of the drill's own area still visible
MIN_VISIBLE_PX = 6000     # absolute: ~1.5% of a 640x640 frame
MIN_SEPARATION = 0.055    # metres, stops distractors spawning inside the drill
KP_MARGIN = 15            # px a keypoint may sit outside the frame
MAX_KP_OUTSIDE = 2        # at most this many of the 8 may be outside


def build_xml(n_distractors, floor_rgba, ambient, lights, fovy):
    """One model per image. Rebuilding the model is ~10 ms and buys full
    freedom over lighting and object count, which is worth more than the time."""
    mesh_xml = f'<mesh name="drill" file="{ASSET_DIR}/{DRILL}.msh"/>'
    tex_xml = (f'<texture name="drilltex" type="2d" file="{ASSET_DIR}/{DRILL}.png"/>'
               f'<material name="drillmat" texture="drilltex" specular="0.1" '
               f'shininess="0.2"/>')
    for i, name in enumerate(DISTRACTORS[:n_distractors]):
        mesh_xml += f'<mesh name="d{i}" file="{ASSET_DIR}/{name}.msh"/>'
        tex_xml += (f'<texture name="dt{i}" type="2d" file="{ASSET_DIR}/{name}.png"/>'
                    f'<material name="dm{i}" texture="dt{i}" specular="0.1" '
                    f'shininess="0.2"/>')

    light_xml = ""
    for d, diff in lights:
        light_xml += (f'<light pos="{d[0]:.3f} {d[1]:.3f} {d[2]:.3f}" '
                      f'dir="{-d[0]:.3f} {-d[1]:.3f} {-d[2]:.3f}" directional="true" '
                      f'diffuse="{diff:.3f} {diff:.3f} {diff:.3f}" specular="0.05 0.05 0.05"/>')

    body_xml = '<body name="drill" mocap="true" pos="0 0 0.05">'
    body_xml += '<geom name="drillgeom" type="mesh" mesh="drill" material="drillmat" contype="0" conaffinity="0"/></body>'
    for i in range(n_distractors):
        body_xml += (f'<body name="dist{i}" mocap="true" pos="0 0 -5">'
                     f'<geom name="distgeom{i}" type="mesh" mesh="d{i}" material="dm{i}" '
                     f'contype="0" conaffinity="0"/></body>')

    return f"""
<mujoco model="kprender">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="{IMG_SIZE}" offheight="{IMG_SIZE}"/>
    <headlight ambient="{ambient:.3f} {ambient:.3f} {ambient:.3f}"
               diffuse="0.05 0.05 0.05" specular="0 0 0"/>
    <!-- offsamples=0: MuJoCo's docs note that some backends ignore the
         instruction to disable multisampling during segmentation rendering.
         With MSAA on, edge pixels get blended segmentation ids and the
         occluded/unoccluded pixel counts stop being comparable, which is how
         84/1000 samples ended up with visible_frac slightly above 1.0. -->
    <quality shadowsize="2048" offsamples="0"/>
  </visual>
  <asset>
    {mesh_xml}
    {tex_xml}
    <material name="floormat" rgba="{floor_rgba}" specular="0.05" shininess="0.1"/>
  </asset>
  <worldbody>
    {light_xml}
    <geom name="floor" type="plane" size="3 3 0.1" material="floormat"/>
    {body_xml}
    <camera name="cam" pos="0 0 0.5" fovy="{fovy:.3f}"/>
  </worldbody>
</mujoco>
"""


def random_rotation(rng):
    """Uniform on SO(3) via a random quaternion. Sampling Euler angles
    uniformly does NOT give uniform rotations -- it clusters at the poles."""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.array([q[3], q[0], q[1], q[2]]))
    return R.reshape(3, 3)


def camera_matrix(fovy):
    f = (IMG_SIZE / 2.0) / np.tan(np.radians(fovy) / 2.0)
    return np.array([[f, 0, IMG_SIZE / 2.0], [0, f, IMG_SIZE / 2.0], [0, 0, 1.0]])


def set_body(data, model, name, T):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    mid = model.body_mocapid[bid]
    data.mocap_pos[mid] = T[:3, 3]
    data.mocap_quat[mid] = tf.R_to_quat_wxyz(T[:3, :3])


def render_one(rng, debug=False):
    """Returns (image_bgr, keypoints_2d, T_cam_drill, K, visible_frac) or None
    if the drill ended up outside the frame."""
    n_dist = int(rng.integers(0, MAX_DISTRACTORS + 1))
    floor = f"{rng.uniform(0.15, 0.85):.3f} {rng.uniform(0.15, 0.85):.3f} {rng.uniform(0.15, 0.85):.3f} 1"
    ambient = rng.uniform(0.25, 0.65)
    n_lights = int(rng.integers(1, 3))
    lights = []
    for _ in range(n_lights):
        d = rng.normal(size=3)
        d[2] = abs(d[2]) + 0.5
        d = d / np.linalg.norm(d) * 3.0
        lights.append((d, rng.uniform(0.3, 0.9)))
    fovy = FOVY_BASE + rng.uniform(-3, 3)

    model = mujoco.MjModel.from_xml_string(
        build_xml(n_dist, floor, ambient, lights, fovy))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, IMG_SIZE, IMG_SIZE)
    K = camera_matrix(fovy)

    # --- drill pose ---
    R_obj = random_rotation(rng)
    t_obj = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05),
                      rng.uniform(0.04, 0.10)])
    T_base_drill = tf.make_T(R_obj, t_obj)
    set_body(data, model, "drill", T_base_drill)

    # --- distractors, biased to sit near the drill so they sometimes occlude ---
    # MIN_SEPARATION keeps them from spawning inside the drill mesh, which
    # renders as objects fused together and is not a real occlusion case.
    dist_poses, dist_rots = [], []
    for i in range(n_dist):
        for _ in range(20):
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(MIN_SEPARATION, 0.20)
            p = t_obj + np.array([r * np.cos(ang), r * np.sin(ang),
                                  rng.uniform(-0.01, 0.05)])
            if all(np.linalg.norm(p - q) > MIN_SEPARATION for q in dist_poses):
                break
        R_d = random_rotation(rng)
        dist_poses.append(p)
        dist_rots.append(R_d)
        set_body(data, model, f"dist{i}", tf.make_T(R_d, p))

    # --- camera on a dome, aimed at the drill ---
    az = rng.uniform(0, 2 * np.pi)
    el = np.radians(rng.uniform(20, 80))
    dist = rng.uniform(0.32, 0.65)
    eye = t_obj + np.array([dist * np.cos(el) * np.cos(az),
                            dist * np.cos(el) * np.sin(az),
                            dist * np.sin(el)])
    aim = t_obj + rng.normal(0, 0.012, 3)
    T_base_cam = tf.look_at_camera(eye, aim, rng.uniform(-180, 180))

    # MuJoCo wants the OpenGL convention; our pose is OpenCV.
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    model.cam_pos[cid] = T_base_cam[:3, 3]
    model.cam_quat[cid] = tf.R_to_quat_wxyz(
        T_base_cam[:3, :3] @ tf.R_GL_TO_CV.T)
    mujoco.mj_forward(model, data)

    T_cam_drill = tf.inv_T(T_base_cam) @ T_base_drill

    # --- keypoints, and reject views where they leave the frame ---
    pts = kp.project_keypoints(T_cam_drill, K)
    # The bbox is larger than the drill, so demanding all 8 corners strictly
    # inside would reject many good close-up views. But letting corners drift far
    # outside gives the network a target it cannot see any evidence for, and the
    # heatmap has nowhere to put the peak. Allow a small margin and at most two.
    outside = ((pts[:, 0] < -KP_MARGIN) | (pts[:, 0] > IMG_SIZE + KP_MARGIN) |
               (pts[:, 1] < -KP_MARGIN) | (pts[:, 1] > IMG_SIZE + KP_MARGIN))
    if outside.sum() > MAX_KP_OUTSIDE:
        return None
    if (T_cam_drill[2, 3] < 0.15) or (T_cam_drill[2, 3] > 0.9):
        return None

    # --- occlusion: segmentation pass with and without distractors ---
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="cam")
    seg = renderer.render()[:, :, 0]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "drillgeom")
    visible = int((seg == gid).sum())
    renderer.disable_segmentation_rendering()

    if visible < MIN_VISIBLE_PX:           # drill too small or too hidden
        return None

    # Unoccluded area: hide the distractors far away and re-segment.
    for i in range(n_dist):
        set_body(data, model, f"dist{i}", tf.make_T(np.eye(3), [0, 0, -50]))
    mujoco.mj_forward(model, data)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="cam")
    full = int((renderer.render()[:, :, 0] == gid).sum())
    renderer.disable_segmentation_rendering()

    # restore the distractors to the SAME poses that were measured
    for i, p in enumerate(dist_poses):
        set_body(data, model, f"dist{i}", tf.make_T(dist_rots[i], p))
    mujoco.mj_forward(model, data)
    if visible < MIN_VISIBLE_PX:
        return None

    # A visibility FRACTION above 1 means the occluded pass found more drill
    # pixels than the unoccluded one, which is impossible. Log it rather than
    # clamping silently: the clamp hides the cause, and the cause was
    # multisampled segmentation rendering.
    if visible > full:
        over = visible / max(full, 1)
        if over > 1.02:
            raise RuntimeError(
                f"visible ({visible}) exceeds unoccluded ({full}) by "
                f"{100*(over-1):.1f}%. Check <quality offsamples> and that "
                f"the distractors really moved out of frame.")
        print(f"  [warn] visible/full = {over:.4f} (>1 by "
              f"{100*(over-1):.2f}%), clamping")
    visible_frac = float(min(visible / max(full, 1), 1.0))

    # A drill at 4% visible still carries a full 8-keypoint label, which teaches
    # the network to hallucinate corners from a few pixels. Occlusion is wanted;
    # near-total occlusion is label noise.
    if visible_frac < MIN_VISIBLE_FRAC:
        return None

    renderer.update_scene(data, camera="cam")
    rgb = renderer.render()
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # sensor-like degradation: mild noise, then JPEG on write
    img = np.clip(img.astype(np.float32) +
                  rng.normal(0, rng.uniform(0, 3), img.shape), 0, 255).astype(np.uint8)

    if debug:
        for a, b in kp.BBOX_EDGES:
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     (0, 255, 0), 1)
        for i, p in enumerate(pts):
            cv2.circle(img, tuple(p.astype(int)), 3, (0, 0, 255), -1)

    return img, pts, T_cam_drill, K, visible_frac, n_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=None,
                    help="defaults to start-idx, so chunks never repeat poses")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--jpeg", type=int, default=92)
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    seed = args.start_idx if args.seed is None else args.seed
    rng = np.random.default_rng(seed)

    records, made, tries = [], 0, 0
    target = args.n
    while made < target and tries < target * 8:
        tries += 1
        r = render_one(rng, args.debug)
        if r is None:
            continue
        img, pts, T, K, vf, nd = r
        idx = args.start_idx + made
        name = f"{idx:06d}.jpg"
        cv2.imwrite(os.path.join(img_dir, name), img,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg])
        records.append({
            "file": name,
            "keypoints": pts.round(3).tolist(),
            "T_cam_drill": T.round(6).tolist(),
            "K": K.round(4).tolist(),
            "visible_frac": round(vf, 4),
            "n_distractors": nd,
        })
        made += 1
        if made % 100 == 0:
            print(f"  {made}/{target}  (reject rate "
                  f"{100*(tries-made)/tries:.0f}%)", flush=True)

    shard = os.path.join(args.out, f"manifest_{args.start_idx:06d}.json")
    with open(shard, "w") as f:
        json.dump({"start_idx": args.start_idx, "n": made, "seed": seed,
                   "img_size": IMG_SIZE, "records": records}, f)

    vfs = np.array([r["visible_frac"] for r in records])
    print(f"\nrendered {made} images -> {img_dir}/")
    print(f"reject rate     : {100*(tries-made)/max(tries, 1):.1f}%")
    print(f"occlusion       : {(vfs < 0.95).mean()*100:.1f}% of images have some, "
          f"{(vfs < 0.7).mean()*100:.1f}% below 70% visible")
    print(f"mean visible    : {vfs.mean()*100:.1f}%")
    print(f"wrote {shard}")


if __name__ == "__main__":
    main()
