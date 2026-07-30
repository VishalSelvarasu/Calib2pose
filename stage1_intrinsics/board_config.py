"""
Single source of truth for the ChArUco board.

Both the generator and the detector import from here. The most common silent
failure in ChArUco calibration is generating with one dictionary/geometry and
detecting with another -- detection then just returns nothing, or worse,
returns partially wrong correspondences.

IMPORTANT after printing:
    Measure one printed chessboard square edge with a ruler (or calipers) and
    set SQUARE_LENGTH_MM to the MEASURED value, not the nominal one. Printers
    apply scaling you did not ask for.

    Note precisely what this affects: the intrinsics (fx, fy, cx, cy, distortion)
    are invariant to board scale -- scale only enters the extrinsic translation.
    So a wrong square size will NOT corrupt stage 1. It WILL corrupt stage 2
    (hand-eye), because that solves for a translation in metres. Fix it now.
"""

import cv2

# --- geometry (nominal, A4) ---------------------------------------------------
SQUARES_X = 7           # columns of chessboard squares
SQUARES_Y = 10          # rows of chessboard squares
SQUARE_LENGTH_MM = 25.0  # <-- OVERWRITE WITH YOUR MEASURED VALUE
MARKER_LENGTH_MM = 18.0  # 0.72 of square; keep the ratio if you change sizes

# --- dictionary ---------------------------------------------------------------
# 5x5 bits: better Hamming margin than 4x4 on a soft, noisy webcam image.
# 7*10/2 = 35 markers needed, DICT_5X5_100 has 100.
ARUCO_DICT_ID = cv2.aruco.DICT_5X5_100

# --- printing -----------------------------------------------------------------
PRINT_DPI = 200
PAGE_W_MM = 210.0   # A4
PAGE_H_MM = 297.0

SQUARE_LENGTH_M = SQUARE_LENGTH_MM / 1000.0
MARKER_LENGTH_M = MARKER_LENGTH_MM / 1000.0

# Interior chessboard corners -- these are what actually get calibrated on.
N_CORNERS = (SQUARES_X - 1) * (SQUARES_Y - 1)


def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)


def get_board(square_length_m: float = None):
    """Build the board. Pass square_length_m to override the measured size."""
    s = SQUARE_LENGTH_M if square_length_m is None else square_length_m
    m = s * (MARKER_LENGTH_MM / SQUARE_LENGTH_MM)
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), s, m, get_dictionary()
    )


def get_detector(board=None):
    """
    Detector tuned for a laptop webcam: soft optics, compression artefacts,
    low contrast. Defaults are tuned for crisp machine-vision images.
    """
    board = board if board is not None else get_board()

    det_params = cv2.aruco.DetectorParameters()
    # Subpixel refinement of the marker corners. Costs a little time, buys a
    # lot of accuracy on a blurry sensor.
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det_params.cornerRefinementWinSize = 5
    # Webcam auto-exposure swings; widen the adaptive threshold sweep.
    det_params.adaptiveThreshWinSizeMin = 3
    det_params.adaptiveThreshWinSizeMax = 43
    det_params.adaptiveThreshWinSizeStep = 8
    # Accept slightly smaller markers -- you will hold the board further away
    # than you think when covering the image corners.
    det_params.minMarkerPerimeterRate = 0.02
    # JPEG/MJPG ringing pushes bits off their nominal value; loosen the error rate.
    det_params.maxErroneousBitsInBorderRate = 0.4
    det_params.errorCorrectionRate = 0.6

    charuco_params = cv2.aruco.CharucoParameters()
    # Interpolate chessboard corners even where the neighbouring marker was
    # missed -- recovers corners at the image edge, which is exactly where you
    # need data to constrain the distortion model.
    charuco_params.tryRefineMarkers = True

    refine_params = cv2.aruco.RefineParameters()

    return cv2.aruco.CharucoDetector(
        board, charuco_params, det_params, refine_params
    )


def summary() -> str:
    return (
        f"ChArUco {SQUARES_X}x{SQUARES_Y} | square {SQUARE_LENGTH_MM:.2f} mm | "
        f"marker {MARKER_LENGTH_MM:.2f} mm | DICT_5X5_100 | "
        f"{N_CORNERS} interior corners"
    )
