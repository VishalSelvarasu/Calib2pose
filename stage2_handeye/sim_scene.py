import numpy as np
import mujoco


GT_CAM_POS_IN_FLANGE = np.array([0.055, -0.032, 0.081])
GT_CAM_RPY_IN_FLANGE = np.radians([180.0, -15.0, 90.0])

R_GL_TO_CV = np.diag([1.0, -1.0, -1.0])

IMG_W, IMG_H = 1280, 720
FOVY_DEG = 45.0


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


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def R_to_quat_wxyz(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def gt_T_flange_cam():
    """Ground-truth camera pose in the flange frame, in OpenCV convention."""
    return make_T(rpy_to_R(GT_CAM_RPY_IN_FLANGE), GT_CAM_POS_IN_FLANGE)


def camera_matrix():
    """Pinhole K implied by MuJoCo's vertical FOV. No distortion by design --
    stage 2 must isolate hand-eye error from lens error."""
    f = (IMG_H / 2.0) / np.tan(np.radians(FOVY_DEG) / 2.0)
    return np.array([[f, 0.0, IMG_W / 2.0],
                     [0.0, f, IMG_H / 2.0],
                     [0.0, 0.0, 1.0]])


def build_xml(board_png, board_w_m, board_h_m):
    """
    The camera is declared as a child of `flange`, with a quaternion that
    already includes the OpenCV->OpenGL flip. So the MuJoCo camera body sits at
    GT_CAM_* but points the OpenGL way, while our ground truth is expressed the
    OpenCV way.
    """
    R_cv = rpy_to_R(GT_CAM_RPY_IN_FLANGE)
    R_gl = R_cv @ R_GL_TO_CV.T          # undo the flip for MuJoCo's convention
    q = R_to_quat_wxyz(R_gl)
    p = GT_CAM_POS_IN_FLANGE

    return f"""
<mujoco model="handeye">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="{IMG_W}" offheight="{IMG_H}"/>
    <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/>
    <quality shadowsize="0" offsamples="8"/>
  </visual>
  <asset>
    <texture name="boardtex" type="2d" file="{board_png}"/>
    <material name="boardmat" texture="boardtex" texuniform="false"
              specular="0" shininess="0" reflectance="0"/>
    <material name="floormat" rgba="0.55 0.55 0.58 1" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" material="floormat"/>
    <geom name="board" type="box" pos="0 0 0.001"
          size="{board_w_m/2:.6f} {board_h_m/2:.6f} 0.001" material="boardmat"/>
    <body name="flange" mocap="true" pos="0 0 0.5">
      <geom type="box" size="0.03 0.03 0.01" rgba="0.3 0.3 0.35 1"/>
      <camera name="cam" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
              quat="{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
              fovy="{FOVY_DEG}"/>
    </body>
  </worldbody>
</mujoco>
"""


class Scene:
    def __init__(self, board_png, board_w_m, board_h_m):
        self.model = mujoco.MjModel.from_xml_string(
            build_xml(board_png, board_w_m, board_h_m)
        )
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, IMG_H, IMG_W)
        self.K = camera_matrix()

    def set_flange(self, T_base_flange):
        """Command the flange pose. Returns the pose MuJoCo actually applied."""
        self.data.mocap_pos[0] = T_base_flange[:3, 3]
        self.data.mocap_quat[0] = R_to_quat_wxyz(T_base_flange[:3, :3])
        mujoco.mj_forward(self.model, self.data)
        return self.flange_pose()

    def flange_pose(self):
        """T_base_flange, i.e. what forward kinematics would report."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "flange")
        R = np.array(self.data.xmat[bid]).reshape(3, 3)
        return make_T(R, self.data.xpos[bid])

    def render(self):
        self.renderer.update_scene(self.data, camera="cam")
        return self.renderer.render()

    def true_T_base_cam(self):
        """Where the camera really is, straight from MuJoCo, in OpenCV
        convention. Used only to check the solver -- never fed to it."""
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        R_gl = np.array(self.data.cam_xmat[cid]).reshape(3, 3)
        return make_T(R_gl @ R_GL_TO_CV, self.data.cam_xpos[cid])
