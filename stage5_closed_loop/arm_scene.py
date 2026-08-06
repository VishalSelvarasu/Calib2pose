
import os

import numpy as np
import mujoco

from common import transforms as tf

UR5E_DIR = "ur5e"
UR5E_SRC = "ur5e.xml"          # pristine Menagerie model, never edited
UR5E_XML = "_ur5e_cam.xml"     # generated copy with the camera injected
ASSET_DIR = "assets"           # ditto -- meshdir from ur5e.xml already points
# here, so asset filenames are bare

# The generated scene is written INTO ur5e/ rather than beside it. MuJoCo
# resolves <include> paths relative to the including file, but resolves the
# included file's own `meshdir` relative to that same outer file -- so a scene
# one directory up makes every mesh path fail with
# "Error opening file 'assets/ur5e/base_0.obj'".
SCENE_PATH = os.path.join(UR5E_DIR, "_scene_gen.xml")

IMG_W, IMG_H = 640, 640
FOVY_DEG = 45.0

# Reused verbatim from stage 2: the camera's true pose in the flange frame.
GT_CAM_POS_IN_FLANGE = np.array([0.055, -0.032, 0.081])
GT_CAM_RPY_IN_FLANGE = np.radians([180.0, -15.0, 90.0])

# Somewhere the arm can comfortably reach, off to one side of the base.
DRILL_POS = np.array([0.50, 0.10, 0.06])
DRILL_RPY = np.radians([0.0, 0.0, 35.0])

HOME_Q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


def camera_matrix():
    f = (IMG_H / 2.0) / np.tan(np.radians(FOVY_DEG) / 2.0)
    return np.array([[f, 0, IMG_W / 2.0], [0, f, IMG_H / 2.0], [0, 0, 1.0]])


def gt_T_flange_cam():
    return tf.make_T(tf.rpy_to_R(GT_CAM_RPY_IN_FLANGE), GT_CAM_POS_IN_FLANGE)


# attachment_site as declared in ur5e.xml, expressed in wrist_3_link's frame.
# MJCF attaches cameras to BODIES, not sites, so the camera pose has to be
# composed through this offset: T_wrist_cam = T_wrist_site @ T_site_cam.
SITE_POS_IN_WRIST = np.array([0.0, 0.1, 0.0])
SITE_QUAT_IN_WRIST = np.array([-1.0, 1.0, 0.0, 0.0])   # wxyz, unnormalised


def _site_T_in_wrist():
    q = SITE_QUAT_IN_WRIST / np.linalg.norm(SITE_QUAT_IN_WRIST)
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, q)
    return tf.make_T(R.reshape(3, 3), SITE_POS_IN_WRIST)


def write_arm_with_camera():
    """Write a copy of the Menagerie UR5e with the eye-in-hand camera injected.

    MJCF does not allow re-opening an included body to add children, and MuJoCo
    cameras attach to bodies rather than sites, so the camera cannot be added
    from the wrapper scene. Injecting it into a generated copy keeps the
    pristine Menagerie file untouched while making the camera a genuine child of
    the last link -- which is what makes forward kinematics move it.

    The pose is composed through attachment_site's own offset:
        T_wrist_cam = T_wrist_site @ T_site_cam
    """
    src = os.path.join(UR5E_DIR, UR5E_SRC)
    dst = os.path.join(UR5E_DIR, UR5E_XML)
    with open(src) as f:
        xml = f.read()

    T_wrist_cam = _site_T_in_wrist() @ gt_T_flange_cam()
    q = tf.R_to_quat_wxyz(T_wrist_cam[:3, :3] @ tf.R_GL_TO_CV.T)
    p = T_wrist_cam[:3, 3]
    cam = (f'<camera name="eye" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
           f'quat="{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}" '
           f'fovy="{FOVY_DEG}"/>')

    anchor = '<site name="attachment_site"'
    i = xml.index(anchor)
    j = xml.index("/>", i) + 2
    out = xml[:j] + "\n                  " + cam + xml[j:]
    with open(dst, "w") as f:
        f.write(out)
    return dst


def build_xml():
    """Wrap the Menagerie UR5e, adding the camera, the drill and a table."""
    dq = tf.R_to_quat_wxyz(tf.rpy_to_R(DRILL_RPY))

    return f"""
<mujoco model="stage5">
  <include file="{UR5E_XML}"/>
  <!-- ur5e.xml sets meshdir="assets" but not texturedir, so textures would be
       looked up beside the scene file while meshes resolve into assets/.
       Setting both explicitly removes the asymmetry. -->
  <compiler angle="radian" meshdir="{ASSET_DIR}" texturedir="{ASSET_DIR}"/>
  <visual>
    <global offwidth="{IMG_W}" offheight="{IMG_H}"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.35 0.35 0.35"
               specular="0.05 0.05 0.05"/>
    <quality shadowsize="2048" offsamples="4"/>
  </visual>
  <asset>
    <mesh name="drill" file="035_power_drill.msh"/>
    <texture name="drilltex" type="2d" file="035_power_drill.png"/>
    <material name="drillmat" texture="drilltex" specular="0.15" shininess="0.3"/>
    <material name="tablemat" rgba="0.55 0.52 0.48 1" specular="0.05"/>
    <material name="floormat" rgba="0.30 0.32 0.36 1" specular="0.02"/>
    <!-- Markers for the demo GIF: the TRUE grasp point and the one derived
         from the perceived pose. Both are moved by 02_record_demo.py; parked
         far below the floor when unused so they never appear by accident. -->
    <material name="truemat" rgba="0.15 0.85 0.35 0.9" specular="0"/>
    <material name="estmat" rgba="0.95 0.35 0.20 0.9" specular="0"/>
    <material name="ringmat" rgba="0.20 0.75 1.0 0.55" specular="0"/>
  </asset>
  <worldbody>
    <light pos="0.6 0.3 1.6" dir="-0.3 -0.2 -1" directional="true"
           diffuse="0.7 0.7 0.7"/>
    <!-- A workbench with real thickness, not a slab floating above a plane.
         The robot base sits at z=0 on the bench top, so the bench body extends
         DOWNWARD to the floor. A thin slab 25 mm above an infinite floor reads
         as a mat, and an arm swinging past its edge then looks like it is
         leaving the scene rather than reaching over the bench edge. -->
    <geom name="floor" type="plane" pos="0 0 -0.75" size="4 4 0.1"
          material="floormat" contype="0" conaffinity="0"/>
    <!-- Sized so the arm's working volume stays over the bench. The UR5e has
         850 mm of reach; a bench smaller than that means the arm swings past
         its edge during normal motion, which is physically fine but reads as
         the robot leaving the scene. Verified: no link enters the bench volume
         at any keyframe or along the interpolated path. -->
    <geom name="bench" type="box" pos="0.30 0 -0.375" size="0.95 0.75 0.375"
          material="tablemat" contype="0" conaffinity="0"/>
    <!-- Vertical pins rather than spheres: a sphere at the grasp point is
         hidden by the drill and the wrist from most camera angles, which
         defeats the purpose. A thin pin rising above the scene stays visible
         and still reads as marking one exact point at its base. -->
    <body name="marker_true" mocap="true" pos="0 0 -5">
      <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.0035"
            material="truemat" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0 0.058" size="0.008" material="truemat"
            contype="0" conaffinity="0"/>
    </body>
    <!-- Tolerance ring: a flat disc of the grasp-tolerance radius at the true
         grasp point. The measurement this whole stage produces is 6-10 mm on a
         274 mm object; without something at that scale in frame there is
         nothing for the eye to compare against. -->
    <body name="tol_ring" mocap="true" pos="0 0 -5">
      <geom type="cylinder" size="0.015 0.0006" material="ringmat"
            contype="0" conaffinity="0"/>
    </body>
    <body name="marker_est" mocap="true" pos="0 0 -5">
      <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.0035"
            material="estmat" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0 0.058" size="0.008" material="estmat"
            contype="0" conaffinity="0"/>
    </body>
    <body name="drill" pos="{DRILL_POS[0]} {DRILL_POS[1]} {DRILL_POS[2]}"
          quat="{dq[0]:.9f} {dq[1]:.9f} {dq[2]:.9f} {dq[3]:.9f}">
      <geom type="mesh" mesh="drill" material="drillmat"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


class ArmScene:
    def __init__(self):
        # MuJoCo resolves <include> relative to the including file's directory,
        # so the model must be built from a file, not a string.
        write_arm_with_camera()
        with open(SCENE_PATH, "w") as f:
            f.write(build_xml())
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.K = camera_matrix()

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.drill_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "drill")
        self.set_q(HOME_Q)

    # --- kinematics ------------------------------------------------------
    def set_q(self, q):
        self.data.qpos[:6] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def flange_pose(self):
        """T_base_flange from forward kinematics -- what a real controller
        reads back from the robot."""
        R = np.array(self.data.site_xmat[self.site_id]).reshape(3, 3)
        return tf.make_T(R, self.data.site_xpos[self.site_id])

    def set_ring_radius(self, r):
        """Resize the tolerance disc to the grasp tolerance in metres."""
        gid = -1
        for i in range(self.model.ngeom):
            b = self.model.geom_bodyid[i]
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) == "tol_ring":
                gid = i
                break
        if gid >= 0:
            self.model.geom_size[gid][0] = r

    def set_marker(self, name, pos):
        """Move one of the demo markers. Pass None to park it out of sight."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        mid = self.model.body_mocapid[bid]
        self.data.mocap_pos[mid] = (np.array([0, 0, -5.0]) if pos is None
                                    else np.asarray(pos, float))
        mujoco.mj_forward(self.model, self.data)

    def true_T_base_drill(self):
        R = np.array(self.data.xmat[self.drill_id]).reshape(3, 3)
        return tf.make_T(R, self.data.xpos[self.drill_id])

    def jacobian(self):
        """6x6 site Jacobian: d(pos, rot) / d(q)."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return np.vstack([jacp[:, :6], jacr[:, :6]])

    # --- camera ----------------------------------------------------------
    def _ensure_camera(self):
        """Attach the camera to the flange body once the model exists.

        The camera has to be a child of the last link, but Menagerie's XML is
        included wholesale, so it is added by editing the compiled model rather
        than the source: cam_bodyid points it at wrist_3_link and cam_pos /
        cam_quat place it in that body's frame.
        """
        if getattr(self, "_cam_ready", False):
            return
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
        if cid < 0:
            raise RuntimeError("camera 'eye' missing from the model")
        self._cam_ready = True

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, IMG_H, IMG_W)
        self.renderer.update_scene(self.data, camera="eye")
        return self.renderer.render()

    def true_T_base_cam(self):
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
        R_gl = np.array(self.data.cam_xmat[cid]).reshape(3, 3)
        return tf.make_T(R_gl @ tf.R_GL_TO_CV, self.data.cam_xpos[cid])
