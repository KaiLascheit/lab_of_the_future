"""
Verify the pipette marker (ID, axes, size) and the plunger crop box on a single frame,
using the real Aria camera calibration from the VRS file.

Install:  pip install opencv-contrib-python numpy projectaria-tools
Usage:    python check_pipette_marker.py
          (edit the CONFIG block below)

Outputs (in OUTPUT_DIR):
  check_annotated.jpg   full frame: all tags with IDs and labelled X/Y/Z axes, the 3D
                        crop box, and cm ticks along the pipette's "up" direction
  check_zoom.jpg        the same, zoomed to the pipette marker and box
  crop_preview.png      the crop region rectified onto the marker plane, pipette upright
                        (exactly what a crop extractor would produce)
  crop_preview_grid.png the same with a 1 cm grid, labelled in marker X / Y (cm)
  + a printed report: per-tag pixel size, distance, viewing angle, reprojection error,
    pose ambiguity, and an optional tag-size check from a measured tag-to-tag distance.
"""
from pathlib import Path

import cv2
import numpy as np
from projectaria_tools.core import data_provider

# ----------------------------------------------------------------------------- CONFIG
FRAME_PATH = Path("frames/camera-rgb/000276_16284244811943.png")   # any frame showing the pipette
VRS_PATH = Path("streams/96-well_plate_pipetting_multi_p300_QR_20261105_154736.vrs")
OUTPUT_DIR = Path("marker_check")
CAMERA_LABEL = "camera-rgb"
FRAMES_ROTATED_CW90 = False   # True if frames were exported upright (rotated 90° CW)

PIPETTE_MARKER_ID = 43
DEFAULT_TAG_SIZE_M = 0.020    # black-border side length, used for every tag ...
TAG_SIZE_M = {}               # ... unless overridden here, e.g. {43: 0.020, 45: 0.020}

# Crop box in the PIPETTE marker's frame (metres). The tag is mounted upside down,
# so "up the pipette" is -Y and the thumb side (image right) is -X.
BOX_X = (-0.05, 0.04)
BOX_Y = (-0.14, 0.01)         # from 1 cm below the tag centre to 14 cm up the pipette
BOX_Z = (-0.02, 0.02)         # depth around the tag plane (only used for the 3D wireframe)
UP_SIGN, UP_AXIS = -1, 1      # "up the pipette" = -Y; used for the cm ticks
TICK_EVERY_CM = 2

# Crop preview: the marker plane (z = CROP_PLANE_Z) over BOX_X x BOX_Y, resampled
CROP_PLANE_Z = 0.0
CROP_PX_PER_MM = 5

# Optional tag-size check: measure the centre-to-centre distance between the pipette
# tag and a second rigid tag (same pipette) with a ruler, in metres. None = skip.
SIZE_CHECK_MARKER_ID = 45
MEASURED_CENTRE_DISTANCE_M = None
# ----------------------------------------------------------------------------- /CONFIG


def tag_size(mid):
    return TAG_SIZE_M.get(mid, DEFAULT_TAG_SIZE_M)


def load_camera_calibration(vrs_path, label):
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise SystemExit(f"Could not open VRS file: {vrs_path}")
    device_calib = provider.get_device_calibration()
    cam = device_calib.get_camera_calib(label)
    if cam is None:
        raise SystemExit(f"No calibration for '{label}'. Available: {device_calib.get_camera_labels()}")
    return cam


class FrameToCalibMapping:
    """Pixel mapping between the exported frame and the calibration image
    (optional 90° CW rotation + uniform resize)."""

    def __init__(self, frame_wh, calib_wh, rotated_cw90):
        self.rot = rotated_cw90
        fw, fh = frame_wh
        self.uw, self.uh = (fh, fw) if rotated_cw90 else (fw, fh)
        self.sx, self.sy = calib_wh[0] / self.uw, calib_wh[1] / self.uh
        if abs(self.sx - self.sy) > 1e-3:
            print(f"WARNING: non-uniform scale frame->calib ({self.sx:.4f}, {self.sy:.4f})")

    def to_calib(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.rot:
            pts = np.stack([pts[:, 1], self.uh - 1 - pts[:, 0]], axis=1)
        return pts * [self.sx, self.sy]

    def to_frame(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2) / [self.sx, self.sy]
        if self.rot:
            pts = np.stack([self.uh - 1 - pts[:, 1], pts[:, 0]], axis=1)
        return pts


def project(cam, pts_cam):
    """3D camera-frame points -> calibration pixels (NaN where behind the camera)."""
    out = np.full((len(pts_cam), 2), np.nan)
    for k, p in enumerate(np.asarray(pts_cam, dtype=np.float64)):
        if p[2] > 1e-6:
            out[k] = np.asarray(cam.project_no_checks(p)).reshape(2)
    return out


def estimate_pose(corners_calib, size, cam):
    """Pose from 4 corners via the Aria fisheye model. Returns a dict with the best
    pose, reprojection error (px) and the error ratio of the two IPPE solutions."""
    norm = []
    for p in corners_calib:
        ray = np.asarray(cam.unproject_no_checks(np.asarray(p, dtype=np.float64))).reshape(3)
        if ray[2] <= 1e-6:
            return None
        norm.append(ray[:2] / ray[2])
    norm = np.array(norm)
    h = size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, norm, np.eye(3), None,
                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if n == 0:
        return None
    errs = np.asarray(errs).ravel()
    best = int(np.argmin(errs))
    R, _ = cv2.Rodrigues(rvecs[best])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(tvecs[best]).ravel()

    proj = project(cam, (R @ obj.T).T + T[:3, 3])
    reproj = float(np.linalg.norm(proj - corners_calib, axis=1).mean())
    # Ratio close to 1 = the flipped solution fits almost as well -> orientation unreliable
    ambiguity = float(errs.min() / errs.max()) if n > 1 and errs.max() > 0 else 0.0
    return {"T": T, "reproj_px": reproj, "ambiguity": ambiguity}


def to_cam(T, pts):
    return (T[:3, :3] @ np.asarray(pts, dtype=np.float64).T).T + T[:3, 3]


def draw_poly3d(img, T, pts3d, cam, mapping, colour, thickness, closed=False, samples=20):
    """Draw a 3D polyline (marker frame), finely sampled so it bends with the fisheye."""
    pts3d = np.asarray(pts3d, dtype=np.float64)
    if closed:
        pts3d = np.vstack([pts3d, pts3d[:1]])
    for a, b in zip(pts3d[:-1], pts3d[1:]):
        uv = project(cam, to_cam(T, np.linspace(a, b, samples)))
        if np.isnan(uv).any():
            continue
        uv = mapping.to_frame(uv).round().astype(np.int32)
        cv2.polylines(img, [uv.reshape(-1, 1, 2)], False, colour, thickness, cv2.LINE_AA)


def text(img, s, org, colour, scale=0.8):
    org = tuple(int(v) for v in org)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def point_px(T, p, cam, mapping):
    uv = project(cam, to_cam(T, [p]))
    return None if np.isnan(uv).any() else mapping.to_frame(uv)[0]


def draw_axes(img, T, length, cam, mapping):
    o = point_px(T, (0, 0, 0), cam, mapping)
    for axis, name, colour in ((0, "X", (0, 0, 255)), (1, "Y", (0, 255, 0)), (2, "Z", (255, 0, 0))):
        e = np.zeros(3)
        e[axis] = length
        draw_poly3d(img, T, [(0, 0, 0), e], cam, mapping, colour, 3)
        tip = point_px(T, e, cam, mapping)
        if o is not None and tip is not None:
            cv2.circle(img, tuple(tip.round().astype(int)), 5, colour, -1, cv2.LINE_AA)
            text(img, f"+{name}", tip + (tip - o) / max(np.linalg.norm(tip - o), 1) * 18, colour, 0.7)


def draw_box(img, T, cam, mapping):
    (x0, x1), (y0, y1), (z0, z1) = BOX_X, BOX_Y, BOX_Z
    yellow = (0, 255, 255)
    front = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    back = [(x, y, z0) for x, y, _ in front]
    draw_poly3d(img, T, front, cam, mapping, yellow, 3, closed=True)
    draw_poly3d(img, T, back, cam, mapping, yellow, 1, closed=True)
    for a, b in zip(front, back):
        draw_poly3d(img, T, [a, b], cam, mapping, yellow, 1)
    # Crop plane outline
    plane = [(x0, y0, CROP_PLANE_Z), (x1, y0, CROP_PLANE_Z), (x1, y1, CROP_PLANE_Z), (x0, y1, CROP_PLANE_Z)]
    draw_poly3d(img, T, plane, cam, mapping, (255, 255, 0), 1, closed=True)
    # cm ticks along "up the pipette"
    lo, hi = sorted(UP_SIGN * np.array(BOX_Y if UP_AXIS == 1 else BOX_X))
    for cm in range(0, int(hi * 100) + 1, TICK_EVERY_CM):
        if cm / 100 < lo:
            continue
        v = UP_SIGN * cm / 100
        a, b = [x0, 0, CROP_PLANE_Z], [x1, 0, CROP_PLANE_Z]
        a[UP_AXIS] = b[UP_AXIS] = v
        draw_poly3d(img, T, [a, b], cam, mapping, yellow, 1)
        p = point_px(T, b, cam, mapping)
        if p is not None and cm > 0:
            text(img, f"{cm} cm", p + (8, 6), yellow, 0.6)
    # cm ticks across the width (marker X), labelled with the signed X position,
    # drawn as short marks along the box's lower edge (the end nearest the tag)
    near_y = BOX_Y[1] if UP_SIGN < 0 else BOX_Y[0]
    tick_len = 0.01 * (-UP_SIGN)          # 1 cm long, pointing up the pipette
    for cm in range(int(np.ceil(x0 * 100)), int(np.floor(x1 * 100)) + 1):
        x = cm / 100
        draw_poly3d(img, T, [(x, near_y, CROP_PLANE_Z), (x, near_y - tick_len, CROP_PLANE_Z)],
                    cam, mapping, yellow, 2 if cm % TICK_EVERY_CM == 0 else 1, samples=4)
        if cm % TICK_EVERY_CM == 0:
            p = point_px(T, (x, near_y, CROP_PLANE_Z), cam, mapping)
            if p is not None:
                text(img, f"{cm:+d}" if cm else "0", p + (-12, 26), yellow, 0.55)
    w_cm, h_cm = (x1 - x0) * 100, (y1 - y0) * 100
    p = point_px(T, ((x0 + x1) / 2, near_y, CROP_PLANE_Z), cam, mapping)
    if p is not None:
        text(img, f"X (cm): box {w_cm:.0f} cm wide x {h_cm:.0f} cm tall", p + (-150, 58), yellow, 0.6)


def rectified_crop(img, T, cam, mapping):
    """Resample the marker plane over BOX_X x BOX_Y into an upright image:
    rows go up the pipette (top row = furthest up), columns go from +X to -X,
    which for an upside-down tag facing the camera matches the camera's view."""
    (x0, x1), (y0, y1) = BOX_X, BOX_Y
    W = int(round((x1 - x0) * 1000 * CROP_PX_PER_MM))
    H = int(round((y1 - y0) * 1000 * CROP_PX_PER_MM))
    xs = np.linspace(x1, x0, W) if UP_SIGN < 0 else np.linspace(x0, x1, W)
    ys = np.linspace(y0, y1, H) if UP_SIGN < 0 else np.linspace(y1, y0, H)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, CROP_PLANE_Z)], axis=1)
    uv = mapping.to_frame(project(cam, to_cam(T, pts))).astype(np.float32)
    map_x = uv[:, 0].reshape(H, W)
    map_y = uv[:, 1].reshape(H, W)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), xs, ys


def draw_crop_grid(crop, xs, ys):
    """1 cm grid on the rectified crop, labelled in marker coordinates (cm)."""
    out = crop.copy()
    H, W = out.shape[:2]
    x_cm, y_cm = xs * 100, ys * 100
    for cm in range(int(np.ceil(min(x_cm))), int(np.floor(max(x_cm))) + 1):
        c = int(round(np.interp(cm, x_cm[::-1], np.arange(W)[::-1]) if x_cm[0] > x_cm[-1]
                      else np.interp(cm, x_cm, np.arange(W))))
        cv2.line(out, (c, 0), (c, H - 1), (0, 255, 255), 2 if cm == 0 else 1, cv2.LINE_AA)
        text(out, f"{cm:+d}" if cm else "0", (c - 12, H - 8), (0, 255, 255), 0.45)
    for cm in range(int(np.ceil(min(y_cm))), int(np.floor(max(y_cm))) + 1):
        r = int(round(np.interp(cm, y_cm, np.arange(H)) if y_cm[0] < y_cm[-1]
                      else np.interp(cm, y_cm[::-1], np.arange(H)[::-1])))
        cv2.line(out, (0, r), (W - 1, r), (0, 255, 255), 2 if cm == 0 else 1, cv2.LINE_AA)
        if cm % 2 == 0:
            text(out, f"{cm:+d}" if cm else "0", (4, r - 4), (0, 255, 255), 0.45)
    text(out, f"{W / (CROP_PX_PER_MM * 10):.0f} x {H / (CROP_PX_PER_MM * 10):.0f} cm, "
              f"labels = marker X / Y (cm)", (4, 16), (255, 255, 255), 0.45)
    return out


def main():
    img = cv2.imread(str(FRAME_PATH))
    if img is None:
        raise SystemExit(f"Could not read {FRAME_PATH}")
    cam = load_camera_calibration(VRS_PATH, CAMERA_LABEL)
    calib_wh = tuple(int(v) for v in cam.get_image_size())
    mapping = FrameToCalibMapping((img.shape[1], img.shape[0]), calib_wh, FRAMES_ROTATED_CW90)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    det_params = cv2.aruco.DetectorParameters()
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), det_params)
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None:
        raise SystemExit("No AprilTags detected in this frame")

    print(f"Frame {FRAME_PATH.name}: {img.shape[1]}x{img.shape[0]}, calibration {calib_wh[0]}x{calib_wh[1]}")
    print(f"{'id':>4} {'side_px':>8} {'size_cm':>8} {'dist_cm':>8} {'view_deg':>9} "
          f"{'reproj_px':>10} {'ambiguity':>10}")
    vis = img.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    poses = {}
    for c, mid in zip(corners, ids.ravel()):
        mid = int(mid)
        c = c.reshape(4, 2)
        side_px = np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)])
        res = estimate_pose(mapping.to_calib(c), tag_size(mid), cam)
        if res is None:
            print(f"{mid:>4} {side_px:8.1f}   pose failed")
            continue
        poses[mid] = res
        T = res["T"]
        dist = np.linalg.norm(T[:3, 3])
        # Angle between the tag normal (+Z, towards camera) and the line of sight
        view = np.degrees(np.arccos(np.clip(np.dot(T[:3, 2], -T[:3, 3] / dist), -1, 1)))
        flag = "  <-- orientation may flip" if res["ambiguity"] > 0.5 else ""
        print(f"{mid:>4} {side_px:8.1f} {tag_size(mid) * 100:8.1f} {dist * 100:8.1f} {view:9.1f} "
              f"{res['reproj_px']:10.2f} {res['ambiguity']:10.2f}{flag}")
        draw_axes(vis, T, tag_size(mid) * 0.9, cam, mapping)
        o = point_px(T, (0, 0, 0), cam, mapping)
        if o is not None:
            text(vis, f"id {mid}", o + (40, -30), (255, 255, 255), 1.0)

    if PIPETTE_MARKER_ID not in poses:
        print(f"\nPipette marker {PIPETTE_MARKER_ID} not found. Detected: {sorted(poses)}")
        cv2.imwrite(str(OUTPUT_DIR / "check_annotated.jpg"), vis)
        return

    T = poses[PIPETTE_MARKER_ID]["T"]
    draw_box(vis, T, cam, mapping)

    # Which way do the marker axes point in the image? (helps confirm "up" = -Y)
    o = point_px(T, (0, 0, 0), cam, mapping)
    print(f"\nPipette marker {PIPETTE_MARKER_ID} axes in the image:")
    for axis, name in ((0, "X"), (1, "Y")):
        e = np.zeros(3)
        e[axis] = 0.01
        d = point_px(T, e, cam, mapping) - o
        horiz = "right" if d[0] > 0 else "left"
        vert = "down" if d[1] > 0 else "up"
        main = vert if abs(d[1]) > abs(d[0]) else horiz
        print(f"  +{name} points mostly {main} (dx={d[0]:+.0f}px, dy={d[1]:+.0f}px per cm)")

    # Optional tag-size check from a measured tag-to-tag distance
    if SIZE_CHECK_MARKER_ID in poses:
        est = np.linalg.norm(poses[SIZE_CHECK_MARKER_ID]["T"][:3, 3] - T[:3, 3])
        print(f"\nEstimated centre distance {PIPETTE_MARKER_ID}<->{SIZE_CHECK_MARKER_ID}: {est * 100:.2f} cm")
        if MEASURED_CENTRE_DISTANCE_M:
            ratio = MEASURED_CENTRE_DISTANCE_M / est
            print(f"Measured: {MEASURED_CENTRE_DISTANCE_M * 100:.2f} cm -> ratio {ratio:.3f}. "
                  f"If both tags share the same size, it is probably "
                  f"{tag_size(PIPETTE_MARKER_ID) * ratio * 100:.2f} cm, not "
                  f"{tag_size(PIPETTE_MARKER_ID) * 100:.2f} cm.")
        else:
            print("Set MEASURED_CENTRE_DISTANCE_M to check the tag size against a ruler.")

    cv2.imwrite(str(OUTPUT_DIR / "check_annotated.jpg"), vis)

    # Zoomed view around the box
    (x0, x1), (y0, y1), (z0, z1) = BOX_X, BOX_Y, BOX_Z
    corners3d = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    uv = mapping.to_frame(project(cam, to_cam(T, corners3d)))
    uv = uv[~np.isnan(uv).any(axis=1)]
    if len(uv):
        pad = 80
        xa, ya = np.maximum(uv.min(0).astype(int) - pad, 0)
        xb, yb = np.minimum(uv.max(0).astype(int) + pad, [img.shape[1], img.shape[0]])
        cv2.imwrite(str(OUTPUT_DIR / "check_zoom.jpg"), vis[ya:yb, xa:xb])

    crop, xs, ys = rectified_crop(img, T, cam, mapping)
    cv2.imwrite(str(OUTPUT_DIR / "crop_preview.png"), crop)
    cv2.imwrite(str(OUTPUT_DIR / "crop_preview_grid.png"), draw_crop_grid(crop, xs, ys))
    print(f"\nCrop box (marker {PIPETTE_MARKER_ID} frame): "
          f"X {BOX_X[0] * 100:+.1f}..{BOX_X[1] * 100:+.1f} cm ({(BOX_X[1] - BOX_X[0]) * 100:.1f} cm wide), "
          f"Y {BOX_Y[0] * 100:+.1f}..{BOX_Y[1] * 100:+.1f} cm ({(BOX_Y[1] - BOX_Y[0]) * 100:.1f} cm tall), "
          f"Z {BOX_Z[0] * 100:+.1f}..{BOX_Z[1] * 100:+.1f} cm ({(BOX_Z[1] - BOX_Z[0]) * 100:.1f} cm deep)")
    print(f"Saved check_annotated.jpg, check_zoom.jpg, crop_preview.png and crop_preview_grid.png "
          f"({crop.shape[1]}x{crop.shape[0]}) to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()