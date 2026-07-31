import os
import numpy as np
import cv2
from PIL import Image

import board_config as bc

OUT_DIR = "board"


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dpi = bc.PRINT_DPI

    board = bc.get_board()

    board_w_px = mm_to_px(bc.SQUARES_X * bc.SQUARE_LENGTH_MM, dpi)
    board_h_px = mm_to_px(bc.SQUARES_Y * bc.SQUARE_LENGTH_MM, dpi)
    page_w_px = mm_to_px(bc.PAGE_W_MM, dpi)
    page_h_px = mm_to_px(bc.PAGE_H_MM, dpi)

    if board_w_px > page_w_px or board_h_px > page_h_px:
        raise SystemExit(
            f"Board ({board_w_px}x{board_h_px} px) does not fit the page "
            f"({page_w_px}x{page_h_px} px). Reduce SQUARE_LENGTH_MM or the "
            f"square count in board_config.py."
        )

    # marginSize=0: we place the board on the page ourselves so the physical
    # offset is known and the margins are symmetric.
    board_img = board.generateImage((board_w_px, board_h_px), marginSize=0)

    page = np.full((page_h_px, page_w_px), 255, dtype=np.uint8)
    x0 = (page_w_px - board_w_px) // 2
    y0 = (page_h_px - board_h_px) // 2
    page[y0:y0 + board_h_px, x0:x0 + board_w_px] = board_img

    # Ruler ticks under the board: lets you verify print scale without trusting
    # the printer dialog. Measure tick-to-tick, it must be exactly 10 mm.
    tick_y = y0 + board_h_px + mm_to_px(8, dpi)
    if tick_y + mm_to_px(12, dpi) < page_h_px:
        for i in range(11):
            tx = x0 + mm_to_px(10 * i, dpi)
            h = mm_to_px(6 if i % 5 == 0 else 3, dpi)
            cv2.line(page, (tx, tick_y), (tx, tick_y + h), 0, 2)
        cv2.line(page, (x0, tick_y), (x0 + mm_to_px(100, dpi), tick_y), 0, 2)
        cv2.putText(
            page, "100 mm - verify with a ruler",
            (x0, tick_y + mm_to_px(13, dpi)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA,
        )

    label = (
        f"{bc.SQUARES_X}x{bc.SQUARES_Y}  square={bc.SQUARE_LENGTH_MM:g}mm  "
        f"marker={bc.MARKER_LENGTH_MM:g}mm  DICT_5X5_100"
    )
    cv2.putText(
        page, label, (x0, max(30, y0 - mm_to_px(5, dpi))),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA,
    )

    png_path = os.path.join(OUT_DIR, "charuco_A4.png")
    pdf_path = os.path.join(OUT_DIR, "charuco_A4.pdf")
    cv2.imwrite(png_path, page)

    # PIL embeds the DPI, so the PDF page comes out at exactly A4 and the
    # printer has an unambiguous physical size to honour.
    Image.fromarray(page).save(pdf_path, resolution=float(dpi))

    print(bc.summary())
    print(f"page   : {bc.PAGE_W_MM:g} x {bc.PAGE_H_MM:g} mm @ {dpi} dpi")
    print(f"board  : {bc.SQUARES_X * bc.SQUARE_LENGTH_MM:g} x "
          f"{bc.SQUARES_Y * bc.SQUARE_LENGTH_MM:g} mm")
    print(f"wrote  : {pdf_path}")
    print(f"wrote  : {png_path}")
    print("\nPrint the PDF at 100% scale, verify the 100 mm ruler, tape it flat,")
    print("then measure a square and update SQUARE_LENGTH_MM in board_config.py.")


if __name__ == "__main__":
    main()
