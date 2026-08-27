import cv2
import numpy as np

SQUARES_X = 7           # columns of chessboard squares
SQUARES_Y = 10          # rows of chessboard squares
SQUARE_LENGTH_MM = 25.0
MARKER_LENGTH_MM = 18.0


ARUCO_DICT_ID = cv2.aruco.DICT_5X5_100

PRINT_DPI = 200
PAGE_W_MM = 210.0   # A4
PAGE_H_MM = 297.0

SQUARE_LENGTH_M = SQUARE_LENGTH_MM / 1000.0
MARKER_LENGTH_M = MARKER_LENGTH_MM / 1000.0

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
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det_params.cornerRefinementWinSize = 5
    det_params.adaptiveThreshWinSizeMin = 3
    det_params.adaptiveThreshWinSizeMax = 43
    det_params.adaptiveThreshWinSizeStep = 8
    det_params.minMarkerPerimeterRate = 0.02
    det_params.maxErroneousBitsInBorderRate = 0.4
    det_params.errorCorrectionRate = 0.6

    charuco_params = cv2.aruco.CharucoParameters()

    charuco_params.tryRefineMarkers = True

    refine_params = cv2.aruco.RefineParameters()

    return cv2.aruco.CharucoDetector(
        board, charuco_params, det_params, refine_params
    )


def normalize_charuco(charuco_corners, charuco_ids):
    """Force detector output to the OpenCV 4.x shapes: (N,1,2) and (N,1).

    OpenCV 5 returns (N,2) corners and (N,) ids. Code that indexes
    ``corners[k, 0]`` expecting a 2D point then silently gets a scalar x
    coordinate instead, with no exception. Normalising at the detector
    boundary keeps the rest of the stage version-independent.
    """
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        return None, None
    corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
    return corners, ids


def detect(detector, gray):
    """detectBoard + shape normalisation. Use this instead of calling
    detector.detectBoard directly."""
    ch_c, ch_id, mk_c, mk_id = detector.detectBoard(gray)
    ch_c, ch_id = normalize_charuco(ch_c, ch_id)
    return ch_c, ch_id, mk_c, mk_id


def summary() -> str:
    return (
        f"ChArUco {SQUARES_X}x{SQUARES_Y} | square {SQUARE_LENGTH_MM:.2f} mm | "
        f"marker {MARKER_LENGTH_MM:.2f} mm | DICT_5X5_100 | "
        f"{N_CORNERS} interior corners"
    )
