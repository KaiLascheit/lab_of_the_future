"""
Detect AprilTags in an image and save a visualisation with each tag's ID.
Uses relaxed detector settings to find more (small, blurry, angled, low-contrast) tags.

Install:  pip install opencv-contrib-python
Usage:    python detect_apriltags_image.py image.png [output.jpg] [--family 36h11]
              [--ids 26 36 43 45]     keep only these IDs (recommended: relaxed = more false hits)
              [--upscale 2]           enlarge the image first (helps very small tags)
              [--show-rejected]       draw candidates that were found but not decoded (red)
              [--default-params]      use OpenCV's default (stricter) settings, for comparison
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

FAMILIES = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

# Relaxed detector settings (OpenCV defaults in comments)
RELAXED = {
    # Decoding: tolerate more misread bits
    "errorCorrectionRate": 1.0,                     # 0.6  -> correct as many bits as the family allows
    "maxErroneousBitsInBorderRate": 0.5,            # 0.35 -> accept a less clean black border
    "minOtsuStdDev": 2.0,                           # 5.0  -> decode low-contrast tags
    "perspectiveRemovePixelPerCell": 8,             # 4    -> sample each bit more finely
    "perspectiveRemoveIgnoredMarginPerCell": 0.15,  # 0.13 -> ignore a bit more of each cell's edge
    # Candidate search: smaller, more distorted, nearer the image border
    "minMarkerPerimeterRate": 0.01,                 # 0.03 -> allow smaller tags
    "polygonalApproxAccuracyRate": 0.06,            # 0.03 -> allow slightly curved edges (fisheye)
    "minCornerDistanceRate": 0.02,                  # 0.05
    "minDistanceToBorder": 1,                       # 3
    # Thresholding: try more window sizes (uneven lighting)
    "adaptiveThreshWinSizeMin": 3,                  # 3
    "adaptiveThreshWinSizeMax": 53,                 # 23
    "adaptiveThreshWinSizeStep": 4,                 # 10
}


def make_detector(family, relaxed):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if relaxed:
        for name, value in RELAXED.items():
            setattr(params, name, value)
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(FAMILIES[family]), params)


def draw_label(image, text, centre_x, top_y, scale):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    org = (int(centre_x - tw / 2), int(top_y - 12))
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Detect AprilTags and save a visualisation.")
    ap.add_argument("image")
    ap.add_argument("output", nargs="?", help="default: <image>_tags.jpg")
    ap.add_argument("--family", default="36h11", choices=FAMILIES)
    ap.add_argument("--ids", type=int, nargs="+", help="only keep these tag IDs")
    ap.add_argument("--upscale", type=float, default=1.0, help="enlarge the image before detection")
    ap.add_argument("--show-rejected", action="store_true", help="draw undecoded candidates in red")
    ap.add_argument("--default-params", action="store_true", help="use OpenCV's default settings")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    out_path = args.output or str(Path(args.image).with_name(Path(args.image).stem + "_tags.jpg"))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if args.upscale != 1.0:
        gray = cv2.resize(gray, None, fx=args.upscale, fy=args.upscale, interpolation=cv2.INTER_CUBIC)
    detector = make_detector(args.family, relaxed=not args.default_params)
    corners, ids, rejected = detector.detectMarkers(gray)
    # Map corners back to the original image if it was upscaled
    corners = [c / args.upscale for c in corners]
    rejected = [r / args.upscale for r in rejected]

    found = []
    if ids is not None:
        found = [(c.reshape(4, 2), int(i)) for c, i in zip(corners, ids.ravel())]
        if args.ids:
            dropped = sorted({i for _, i in found if i not in args.ids})
            found = [(c, i) for c, i in found if i in args.ids]
            if dropped:
                print(f"Ignored IDs not in --ids: {dropped}")

    mode = "default" if args.default_params else "relaxed"
    if not found:
        print(f"No AprilTags found ({mode} settings).")
    else:
        print(f"Found {len(found)} tag(s) ({mode} settings):")
    for pts, tag_id in sorted(found, key=lambda t: t[1]):
        centre = pts.mean(axis=0)
        side = np.mean([np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)])
        print(f"  ID {tag_id}: centre ({centre[0]:.1f}, {centre[1]:.1f}), side {side:.1f} px")
        # Outline, first corner (top-left in the tag's own frame), and a readable ID label
        cv2.polylines(image, [pts.round().astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(image, tuple(pts[0].round().astype(int)), 6, (0, 0, 255), -1, cv2.LINE_AA)
        draw_label(image, f"id {tag_id}", centre[0], pts[:, 1].min(), max(0.8, side / 60))

    if args.show_rejected:
        for r in rejected:
            cv2.polylines(image, [r.reshape(4, 2).round().astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
        print(f"{len(rejected)} rejected candidate(s) drawn in red")

    cv2.imwrite(out_path, image)
    print(f"Saved visualisation to {out_path}")


if __name__ == "__main__":
    main()