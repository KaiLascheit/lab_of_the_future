"""
Stack the annotated frames (top) and the abstract view (bottom) and write them
out as a video.

Usage:  python make_video.py
        (edit the CONFIG block below)
"""
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------------------- CONFIG
RESULTS_DIR = Path("tracking_results")
TOP_DIR = RESULTS_DIR / "annotated"
BOTTOM_DIR = RESULTS_DIR / "abstract"
OUTPUT_PATH = RESULTS_DIR / "tracking_stacked.mp4"

FPS = 10                 # playback frame rate of the frames you exported
SCALE = 0.5              # resize each panel (1.0 = original size); keeps the video manageable
INTERP_STEPS = 0         # extra cross-faded frames between each pair (0 = off) for smoother motion
DIVIDER_PX = 4           # thickness of the line between the two panels
DIVIDER_COLOUR = (60, 60, 60)
LABELS = ("camera", "abstract")   # panel captions; set to None to disable
SHOW_FRAME_NAME = True   # print the frame file name in the top-left corner
PREVIEW = False          # also show the video in a window while writing (press q to stop)
# ----------------------------------------------------------------------------- /CONFIG


def put_label(img, text, org, scale=0.8):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def compose(top_path, bottom_path):
    top, bottom = cv2.imread(str(top_path)), cv2.imread(str(bottom_path))
    if top is None or bottom is None:
        return None
    if bottom.shape != top.shape:          # should match already; guard anyway
        bottom = cv2.resize(bottom, (top.shape[1], top.shape[0]))
    if SCALE != 1.0:
        size = (round(top.shape[1] * SCALE), round(top.shape[0] * SCALE))
        top = cv2.resize(top, size, interpolation=cv2.INTER_AREA)
        bottom = cv2.resize(bottom, size, interpolation=cv2.INTER_AREA)

    if LABELS:
        put_label(top, LABELS[0], (12, top.shape[0] - 14))
        put_label(bottom, LABELS[1], (12, bottom.shape[0] - 14))
    if SHOW_FRAME_NAME:
        put_label(top, top_path.stem, (12, 30))

    divider = np.full((DIVIDER_PX, top.shape[1], 3), DIVIDER_COLOUR, dtype=np.uint8)
    frame = np.vstack([top, divider, bottom])
    # Most codecs need even dimensions
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


def main():
    names = sorted(p.name for p in TOP_DIR.glob("*.jpg") if (BOTTOM_DIR / p.name).exists())
    if not names:
        raise SystemExit(f"No matching frames in {TOP_DIR} and {BOTTOM_DIR}")

    writer, prev = None, None
    for name in tqdm(names, desc="Writing video", unit="frame"):
        frame = compose(TOP_DIR / name, BOTTOM_DIR / name)
        if frame is None:
            tqdm.write(f"Could not read {name}, skipping")
            continue
        if writer is None:
            h, w = frame.shape[:2]
            fps = FPS * (INTERP_STEPS + 1)
            writer = cv2.VideoWriter(str(OUTPUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise SystemExit(f"Could not open video writer for {OUTPUT_PATH}")

        # Optional cross-fade from the previous frame for smoother playback
        if prev is not None and prev.shape == frame.shape:
            for k in range(1, INTERP_STEPS + 1):
                a = k / (INTERP_STEPS + 1)
                writer.write(cv2.addWeighted(prev, 1 - a, frame, a, 0))
        writer.write(frame)
        prev = frame

        if PREVIEW:
            cv2.imshow("tracking", frame)
            if cv2.waitKey(max(1, int(1000 / FPS))) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    if PREVIEW:
        cv2.destroyAllWindows()
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()