"""
Run the pipeline on one Aria VRS recording.

  python main.py                                   # settings.VRS_PATH, whole recording
  python main.py --vrs ../streams/xyz.vrs --start 3:20 --end 4:30
  python main.py --no-vlm                          # tracking + plate state only

Outputs (in <OUTPUT_ROOT>/<recording name>/):
  report.md / report.json   running experiment report (rewritten every VLM update)
  frames.jsonl              one line per frame: objects, tracking, press, volume, dispense
  frames/                   raw 5 Hz frames (if SAVE_FRAMES)
  visus/tracking/           annotated camera frames, 5 Hz
  visus/abstract/           white-on-black plate/ray view + well map, 5 Hz
  visus/objects/            detector boxes + SAM masks, 1 Hz (the VLM frames)
  visus/tracking_stacked.mp4  tracking above abstract
"""
import argparse
from pathlib import Path

from tqdm import tqdm

import settings as S
from experiment_pipeline import ExperimentPipeline
from vrs_reader import VRSReader


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vrs", type=Path, default=S.VRS_PATH, help="Aria VRS recording")
    ap.add_argument("--start", default=S.WINDOW_START, help='"m:ss" from the start (default: beginning)')
    ap.add_argument("--end", default=S.WINDOW_END, help='"m:ss" from the start (default: end)')
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: OUTPUT_ROOT/<recording name>)")
    ap.add_argument("--no-vlm", action="store_true", help="skip the VLM report updates")
    return ap.parse_args()


def main():
    args = parse_args()
    reader = VRSReader(args.vrs, S.CAMERA_LABEL)
    out_dir = args.out or S.OUTPUT_ROOT / args.vrs.stem
    w, h = (int(v) for v in reader.camera_calibration.get_image_size())
    print(f"{args.vrs.name}: '{S.CAMERA_LABEL}' {w}x{h}, frames at {S.FRAME_RATE_HZ:g} Hz, "
          f"VLM every {S.VLM_EVERY_N_FRAMES} frames -> {out_dir}")

    pipeline = ExperimentPipeline(reader.camera_calibration, out_dir, args.vrs.name,
                                  use_vlm=S.RUN_VLM and not args.no_vlm)
    frames = reader.frames(args.start, args.end, S.FRAME_RATE_HZ,
                           save_dir=out_dir / "frames" if S.SAVE_FRAMES else None)
    total = len(reader.query_times(args.start, args.end, S.FRAME_RATE_HZ))
    try:
        for frame in tqdm(frames, total=total, desc="Processing", unit="frame"):
            pipeline.process(frame)
    finally:   # keep the report and calibration even if the run is interrupted
        pipeline.finish()


if __name__ == "__main__":
    main()