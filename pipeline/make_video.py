"""
Stack the tracking frames (top) and the abstract view (bottom) and write them out
as a video. Called by the pipeline at the end of a run (VIS_VIDEO in settings.py),
or on its own:

  python make_video.py ../outputs/<experiment>/visus
  python make_video.py ../outputs/<experiment>/visus --slowdown 4 --scale 0.5 --interp 2
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DIVIDER_PX = 4
DIVIDER_COLOUR = (60, 60, 60)
LABELS = ("camera", "abstract")   # panel captions; None to disable
SHOW_FRAME_NAME = True


def put_label(img, text, org, scale=0.8):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def compose(top_path, bottom_path, scale):
    top, bottom = cv2.imread(str(top_path)), cv2.imread(str(bottom_path))
    if top is None or bottom is None:
        return None
    if bottom.shape != top.shape:
        bottom = cv2.resize(bottom, (top.shape[1], top.shape[0]))
    if scale != 1.0:
        size = (round(top.shape[1] * scale), round(top.shape[0] * scale))
        top = cv2.resize(top, size, interpolation=cv2.INTER_AREA)
        bottom = cv2.resize(bottom, size, interpolation=cv2.INTER_AREA)
    if LABELS:
        put_label(top, LABELS[0], (12, top.shape[0] - 14))
        put_label(bottom, LABELS[1], (12, bottom.shape[0] - 14))
    if SHOW_FRAME_NAME:
        put_label(top, top_path.stem, (12, 30))
    divider = np.full((DIVIDER_PX, top.shape[1], 3), DIVIDER_COLOUR, dtype=np.uint8)
    frame = np.vstack([top, divider, bottom])
    h, w = frame.shape[:2]          # most codecs need even dimensions
    return frame[: h - h % 2, : w - w % 2]


def make_stacked_video(top_dir, bottom_dir, output_path, fps=5.0, slowdown=1, scale=0.5,
                       interp_steps=0):
    """fps: rate the frames were extracted at. slowdown: play this many times slower
    than real time (each frame held longer). interp_steps: extra cross-faded frames
    between each pair for smoother motion."""
    top_dir, bottom_dir = Path(top_dir), Path(bottom_dir)
    names = sorted(p.name for p in top_dir.glob("*.jpg") if (bottom_dir / p.name).exists())
    if not names:
        print(f"No matching frames in {top_dir} and {bottom_dir}; no video written")
        return None

    slow = max(1, int(slowdown))
    per_frame = (interp_steps + 1) * slow        # video frames per source frame
    video_fps = fps * (interp_steps + 1)

    writer, prev = None, None
    for name in tqdm(names, desc="Writing video", unit="frame"):
        frame = compose(top_dir / name, bottom_dir / name, scale)
        if frame is None:
            tqdm.write(f"Could not read {name}, skipping")
            continue
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     video_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {output_path}")
        if prev is not None and prev.shape == frame.shape and interp_steps > 0:
            for k in range(1, per_frame):        # cross-fade from the previous frame
                a = k / per_frame
                writer.write(cv2.addWeighted(prev, 1 - a, frame, a, 0))
            writer.write(frame)
        else:
            for _ in range(per_frame):           # hold the frame
                writer.write(frame)
        prev = frame

    if writer is not None:
        writer.release()
    n = len(names)
    print(f"Saved {output_path}: {n} frames, {n / fps:.1f} s of recording "
          f"-> {n * slow / fps:.1f} s of video ({slow}x slower)")
    return output_path


def main():
    import settings as S
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("visus_dir", type=Path, help="folder containing tracking/ and abstract/")
    ap.add_argument("--fps", type=float, default=S.FRAME_RATE_HZ)
    ap.add_argument("--slowdown", type=int, default=S.VIDEO_SLOWDOWN)
    ap.add_argument("--scale", type=float, default=S.VIDEO_SCALE)
    ap.add_argument("--interp", type=int, default=S.VIDEO_INTERP_STEPS)
    args = ap.parse_args()
    make_stacked_video(args.visus_dir / "tracking", args.visus_dir / "abstract",
                       args.visus_dir / "tracking_stacked.mp4", args.fps, args.slowdown,
                       args.scale, args.interp)


if __name__ == "__main__":
    main()