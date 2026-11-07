"""
Numbered thumbnail grids of the crops, for quickly spotting presses before labelling.

Usage:  python make_contact_sheets.py [session ...] [--cols 12] [--rows 6] [--every 1]

Output: data/contact_sheets/<session>/sheet_<first>-<last>.jpg
        Each thumbnail is labelled with its frame index; frames without the tag are
        shown as a grey "no tag" tile so gaps in time stay visible. If labels exist,
        a coloured bar under each thumbnail shows the current label.
"""
import argparse

import cv2
import numpy as np

import common
import config as C

THUMB_W = 120
LABEL_COLOURS = {"pressed": (0, 200, 0), "not_pressed": (200, 120, 0),
                 "ignore": (0, 200, 255), C.UNLABELLED: (90, 90, 90)}


def thumb(session, row, label, th):
    tile = np.full((th + 22, THUMB_W, 3), 40, np.uint8)
    if row["marker_found"]:
        img = cv2.imread(str(C.CROPS_DIR / session / row["crop"]))
        if img is not None:
            tile[:th] = cv2.resize(img, (THUMB_W, th), interpolation=cv2.INTER_AREA)
    else:
        cv2.putText(tile, "no tag", (28, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    tile[th:th + 6] = LABEL_COLOURS.get(label, (90, 90, 90))
    cv2.putText(tile, str(row["frame_idx"]), (4, th + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def make_sheets(session, cols, rows_per_sheet, every):
    index = common.read_index(session)[::every]
    labels = common.expand_labels(common.load_change_points(session), [r["frame_idx"] for r in index])
    xs, ys = common.crop_axes()
    th = int(round(THUMB_W * len(ys) / len(xs)))
    out_dir = C.SHEETS_DIR / session
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = cols * rows_per_sheet
    for s in range(0, len(index), per_sheet):
        chunk = list(zip(index[s:s + per_sheet], labels[s:s + per_sheet]))
        tiles = [thumb(session, r, l, th) for r, l in chunk]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
        first, last = chunk[0][0]["frame_idx"], chunk[-1][0]["frame_idx"]
        cv2.imwrite(str(out_dir / f"sheet_{first:06d}-{last:06d}.jpg"), grid,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[{session}] {(len(index) + per_sheet - 1) // per_sheet} sheets -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--every", type=int, default=1, help="use every n-th frame")
    args = ap.parse_args()
    for name in common.session_names(args.sessions):
        make_sheets(name, args.cols, args.rows, args.every)


if __name__ == "__main__":
    main()
