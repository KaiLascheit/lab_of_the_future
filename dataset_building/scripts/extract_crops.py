"""
Extract upright plunger crops for every RGB frame of each session, straight from the VRS.

Usage:  python extract_crops.py                  # all sessions in config.SESSIONS
        python extract_crops.py multi_p300_154736 # just these sessions

Output: data/crops/<session>/<frame_idx:06d>.png   (only frames where the tag was found)
        data/crops/<session>/index.csv             (every frame in range, with tag quality)
Re-running skips crops that already exist unless --overwrite is given.
"""
import argparse
import csv

import cv2
from tqdm import tqdm

import common
import config as C


def extract_session(name, overwrite=False):
    cfg = common.session_cfg(name)
    vrs = common.resolve(cfg["vrs"])
    provider, sid, cam = common.open_vrs(vrs)
    calib_w = int(cam.get_image_size()[0])
    mid = common.marker_id(name)
    detector = common.make_detector()

    n = provider.get_num_data(sid)
    start = cfg.get("start", 0) or 0
    end = n if cfg.get("end") is None else min(cfg["end"], n)
    out_dir = C.CROPS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] {vrs.name}: frames {start}..{end - 1} of {n}, pipette marker {mid}")

    found = 0
    with open(out_dir / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(common.INDEX_FIELDS)
        for i in tqdm(range(start, end), desc=name, unit="frame"):
            img, ts = common.read_rgb_frame(provider, sid, i)
            scale = img.shape[1] / calib_w
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            pose = None
            if ids is not None:
                for c, k in zip(corners, ids.ravel()):
                    if int(k) == mid:
                        pose = common.estimate_pose(c, C.TAG_SIZE_M, cam, scale)
                        break
            crop_name = f"{i:06d}.png"
            crop = None
            if pose is not None:
                if overwrite or not (out_dir / crop_name).exists():
                    crop = common.rectified_crop(img, pose["T"], cam, scale, session=name)
                    if crop is not None:
                        cv2.imwrite(str(out_dir / crop_name), crop)
                else:
                    crop = True   # already extracted
            if pose is None or crop is None:
                w.writerow([i, ts, "", 0, "", "", "", ""])
                continue
            found += 1
            w.writerow([i, ts, crop_name, 1, f"{pose['reproj_px']:.3f}", f"{pose['ambiguity']:.3f}",
                        f"{pose['dist_m']:.4f}", f"{pose['view_deg']:.1f}"])
    print(f"[{name}] marker {mid} found in {found}/{end - start} frames -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="session names (default: all)")
    ap.add_argument("--overwrite", action="store_true", help="re-render existing crops")
    args = ap.parse_args()
    for name in common.session_names(args.sessions):
        extract_session(name, args.overwrite)


if __name__ == "__main__":
    main()