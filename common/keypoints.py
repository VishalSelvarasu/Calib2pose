"""Keypoint definitions for stage 4.

The 8 corners of the drill's axis-aligned bounding box, in the drill's own
frame. Computed once from the mesh and hard-coded so the training set, the
network, and the PnP solve can never disagree about ordering.

Ordering is the (x, y, z) min/max product in that nesting order:

    kp0 (-x,-y,-z)   kp1 (-x,-y,+z)   kp2 (-x,+y,-z)   kp3 (-x,+y,+z)
    kp4 (+x,-y,-z)   kp5 (+x,-y,+z)   kp6 (+x,+y,-z)   kp7 (+x,+y,+z)

Bounding-box corners are the standard choice in the 6D pose literature (BB8,
YOLO-6D, PVNet all use them), which makes the numbers comparable. The cost is
that no corner sits on a visible surface -- the network must infer each one
from the object's overall shape rather than from local evidence.

Mean pairwise separation is 171.7 mm on a 274.0 mm object: wide spread, which
is what conditions the PnP solve well.
"""

import numpy as np

# metres, drill frame
BBOX_MIN = np.array([-0.0283, -0.0880, -0.0917])
BBOX_MAX = np.array([0.0289, 0.0952, 0.1039])

KEYPOINTS_BBOX = np.array([
    [BBOX_MIN[0], BBOX_MIN[1], BBOX_MIN[2]],
    [BBOX_MIN[0], BBOX_MIN[1], BBOX_MAX[2]],
    [BBOX_MIN[0], BBOX_MAX[1], BBOX_MIN[2]],
    [BBOX_MIN[0], BBOX_MAX[1], BBOX_MAX[2]],
    [BBOX_MAX[0], BBOX_MIN[1], BBOX_MIN[2]],
    [BBOX_MAX[0], BBOX_MIN[1], BBOX_MAX[2]],
    [BBOX_MAX[0], BBOX_MAX[1], BBOX_MIN[2]],
    [BBOX_MAX[0], BBOX_MAX[1], BBOX_MAX[2]],
])

N_KEYPOINTS = len(KEYPOINTS_BBOX)

# Edges, for drawing the wireframe box on debug images. Getting the projected
# box to look like a box is the fastest visual check that keypoint ordering
# survived the dataset pipeline.
BBOX_EDGES = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
]


def project_keypoints(T_cam_obj, K, kps=None):
    """Project 3D keypoints into the image. Returns (N,2) pixels."""
    import cv2
    kps = KEYPOINTS_BBOX if kps is None else kps
    rvec = cv2.Rodrigues(T_cam_obj[:3, :3])[0]
    proj, _ = cv2.projectPoints(kps, rvec, T_cam_obj[:3, 3], K, None)
    return proj.reshape(-1, 2)
