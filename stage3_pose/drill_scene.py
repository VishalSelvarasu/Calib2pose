import os
import zipfile
import urllib.request

import numpy as np
import cv2
import mujoco

YCB_ZIP_URL = "https://codeload.github.com/vikashplus/YCB_sim/zip/refs/heads/main"
ASSET_DIR = "assets"

IMG_W, IMG_H = 1280, 720
FOVY_DEG = 45.0
R_GL_TO_CV = np.diag([1.0, -1.0, -1.0])

# Reused from stage 2: the camera's true pose in the flange frame.
GT_CAM_POS_IN_FLANGE = np.array([0.055, -0.032, 0.081])
GT_CAM_RPY_IN_FLANGE = np.radians([180.0, -15.0, 90.0])

PLATE_HALF = 0.025          # 50 mm plates
PLATE_HALF_THICK = 0.002    # the marker sits on the +Z FACE, not mid-plane
MARKER_HALF = 0.020         # marker spans the inner 40 mm
DRILL_POS = np.array([0.0, 0.0, 0.10])


PLATES = [
    (0, np.array([0.000, 0.000, 0.115]), np.radians([0, 0, 0])),    # flat top
    (1, np.array([0.050, 0.000, 0.092]),
     np.radians([0, 40, 0])),   # tilt to +X
    (2, np.array([0.000, -0.078, 0.092]),
     np.radians([40, 0, 0])),  # tilt to -Y
]


def rpy_to_R(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)],
                  [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [
                  0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                  [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def R_to_quat_wxyz(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).ravel())
    return q


def gt_T_flange_cam():
    return make_T(rpy_to_R(GT_CAM_RPY_IN_FLANGE), GT_CAM_POS_IN_FLANGE)


def camera_matrix():
    f = (IMG_H / 2.0) / np.tan(np.radians(FOVY_DEG) / 2.0)
    return np.array([[f, 0, IMG_W / 2.0], [0, f, IMG_H / 2.0], [0, 0, 1.0]])


def fetch_assets():
    """Download the YCB drill mesh and write the marker textures."""
    os.makedirs(ASSET_DIR, exist_ok=True)
    mesh = os.path.join(ASSET_DIR, "035_power_drill.msh")
    tex = os.path.join(ASSET_DIR, "035_power_drill.png")

    if not os.path.exists(mesh):
        zpath = os.path.join(ASSET_DIR, "ycb.zip")
        print("downloading YCB drill mesh (~22 MB, once)...")
        urllib.request.urlretrieve(YCB_ZIP_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            for member, out in [
                ("YCB_sim-main/meshes/035_power_drill.msh", mesh),
                ("YCB_sim-main/textures/035_power_drill.png", tex),
            ]:
                with z.open(member) as src, open(out, "wb") as dst:
                    dst.write(src.read())
        os.remove(zpath)
        print(f"  -> {mesh}")

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    for mid, _, _ in PLATES:
        p = os.path.join(ASSET_DIR, f"marker_{mid}.png")
        if not os.path.exists(p):
            img = cv2.aruco.generateImageMarker(d, mid, 400)
            img = cv2.copyMakeBorder(img, 50, 50, 50, 50,
                                     cv2.BORDER_CONSTANT, value=255)
            cv2.imwrite(p, img)
    return mesh, tex


def marker_object_points():

    h, z = MARKER_HALF, PLATE_HALF_THICK
    local = np.array([[-h, h, z], [h, h, z], [h, -h, z], [-h, -h, z]])
    out = {}
    for mid, pos, rpy in PLATES:
        out[mid] = (rpy_to_R(rpy) @ local.T).T + pos
    return out


def build_xml(mesh, tex):
    Rc = rpy_to_R(GT_CAM_RPY_IN_FLANGE)
    q = R_to_quat_wxyz(Rc @ R_GL_TO_CV.T)
    p = GT_CAM_POS_IN_FLANGE

    plate_xml, tex_xml = "", ""
    for mid, pos, rpy in PLATES:
        pq = R_to_quat_wxyz(rpy_to_R(rpy))
        tex_xml += (
            f'<texture name="mk{mid}" type="2d" file="{ASSET_DIR}/marker_{mid}.png"/>'
            f'<material name="mkm{mid}" texture="mk{mid}" texuniform="false" '
            f'specular="0" shininess="0" reflectance="0"/>'
        )
        plate_xml += (
            f'<geom type="box" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
            f'quat="{pq[0]:.9f} {pq[1]:.9f} {pq[2]:.9f} {pq[3]:.9f}" '
            f'size="{PLATE_HALF} {PLATE_HALF} {PLATE_HALF_THICK}" material="mkm{mid}" '
            f'contype="0" conaffinity="0"/>'
        )

    return f"""
<mujoco model="drillpose">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="{IMG_W}" offheight="{IMG_H}"/>
    <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/>
    <quality shadowsize="0" offsamples="8"/>
  </visual>
  <asset>
    <mesh name="drill" file="{mesh}"/>
    <texture name="drilltex" type="2d" file="{tex}"/>
    <material name="drillmat" texture="drilltex" specular="0" shininess="0"/>
    <material name="floormat" rgba="0.5 0.5 0.53 1" specular="0" shininess="0"/>
    {tex_xml}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" material="floormat"/>
    <body name="drill" pos="{DRILL_POS[0]} {DRILL_POS[1]} {DRILL_POS[2]}">
      <geom type="mesh" mesh="drill" material="drillmat"
            contype="0" conaffinity="0"/>
      {plate_xml}
    </body>
    <body name="flange" mocap="true" pos="0 0 0.6">
      <geom type="box" size="0.03 0.03 0.01" rgba="0.3 0.3 0.35 1"
            contype="0" conaffinity="0"/>
      <camera name="cam" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
              quat="{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
              fovy="{FOVY_DEG}"/>
    </body>
  </worldbody>
</mujoco>
"""


class DrillScene:
    def __init__(self):
        mesh, tex = fetch_assets()
        self.model = mujoco.MjModel.from_xml_string(build_xml(mesh, tex))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, IMG_H, IMG_W)
        self.K = camera_matrix()
        self.obj_pts = marker_object_points()
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        )
        mid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, "drill")
        a, n = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
        self.model_points = np.array(
            self.model.mesh_vert[a:a + n]).reshape(-1, 3)
        self.diameter = float(np.linalg.norm(
            self.model_points.max(0) - self.model_points.min(0)))

    def set_flange(self, T):
        self.data.mocap_pos[0] = T[:3, 3]
        self.data.mocap_quat[0] = R_to_quat_wxyz(T[:3, :3])
        mujoco.mj_forward(self.model, self.data)

    def flange_pose(self):
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "flange")
        return make_T(np.array(self.data.xmat[b]).reshape(3, 3), self.data.xpos[b])

    def true_T_base_drill(self):
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drill")
        return make_T(np.array(self.data.xmat[b]).reshape(3, 3), self.data.xpos[b])

    def render(self):
        self.renderer.update_scene(self.data, camera="cam")
        return self.renderer.render()

    def detect(self, gray, noise_px=0.0, rng=None):
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None, None, []
        obj, img, seen = [], [], []
        for c, i in zip(corners, ids.ravel()):
            if int(i) not in self.obj_pts:
                continue
            pts = np.asarray(c).reshape(-1, 2)
            if noise_px > 0 and rng is not None:
                pts = pts + rng.normal(0, noise_px, pts.shape)
            obj.append(self.obj_pts[int(i)])
            img.append(pts)
            seen.append(int(i))
        if not obj:
            return None, None, []
        return (np.vstack(obj).astype(np.float64),
                np.vstack(img).astype(np.float64), seen)
