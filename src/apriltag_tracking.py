"""
Detect AprilTag markers on the pipette (IDs 30) and 96-well plate (IDs 26, 36)
in every frame of a directory, and estimate each marker's 6-DoF pose using the
RGB camera calibration stored in an Aria Gen 2 VRS recording.

Install:  pip install opencv-contrib-python tqdm projectaria-tools
Usage:    python detect_apriltags.py
          (edit the CONFIG block below: paths, frame range, tag size)

Outputs:
  OUTPUT_DIR/annotated/<frame>.jpg   frames with outlines, IDs and pose axes
  OUTPUT_DIR/abstract/<frame>.jpg    same-size white-on-black view of plate box,
                                     grid, hit cell, marker 30 and its ray
  OUTPUT_DIR/poses.csv               one row per detected marker per frame

Pose conventions:
  * Marker frame: origin at tag centre, X right, Y up, Z out of the tag
    towards the camera (OpenCV ArUco convention).
  * T_camera_marker: marker pose in the RGB camera frame (metres).
  * T_device_marker: the same pose in the Aria device frame, via the factory
    extrinsics T_device_camera. To get world poses, chain with
    T_world_device from MPS closed-loop trajectory.
  * T_global_marker: pose in the "global" frame, centred on ORIGIN_MARKER_ID (36)
    as seen in the current frame. If 36 isn't visible, the most recent 36 pose
    is used instead:
        T_global_cam(t)    = inv(T_cam_36(t))       if 36 is visible in frame t
                           = inv(T_cam_36(t_last))  otherwise
        T_global_marker(t) = T_global_cam(t) @ T_cam_marker(t)
    The "anchor" column records "current" or "most_recent" for each row.
"""
from pathlib import Path
import csv

import cv2
import numpy as np
from tqdm import tqdm
from projectaria_tools.core import data_provider

# ----------------------------------------------------------------------------- CONFIG
FRAMES_DIR = Path("frames/camera-rgb")
VRS_PATH = Path("streams/96-well_plate_pipetting_p1000_QR_20261105_154354.vrs")
OUTPUT_DIR = Path("tracking_results")

START_IDX = 330      # first frame index (into the sorted frame list)
END_IDX = 380     # one past the last frame index; None = run to the end
SAVE_ANNOTATED = True

CAMERA_LABEL = "camera-rgb"   # Aria RGB camera stream label

# Black-border side length of each tag in METRES (measure your printed tags!).
TAG_SIZE_M = {
    30: 0.020,     # pipette
    26: 0.020, 36: 0.020,     # 96-well plate
}
MARKER_TO_OBJECT = {30: "pipette", 26: "plate", 36: "plate"}

# Marker that defines the global coordinate system in each frame
ORIGIN_MARKER_ID = 36

# 96-well plate bounding box, estimated from the two plate markers.
# Seen from above, PLATE_TOP_ID and PLATE_BOTTOM_ID sit at the upper and lower
# right corners of the plate, and the plate body extends to the LEFT of the
# line from PLATE_TOP_ID to PLATE_BOTTOM_ID.
DRAW_PLATE_BOX = True
PLATE_TOP_ID = 36
PLATE_BOTTOM_ID = 26
PLATE_EDGE_EXTENSION_M = 0.015   # plate extends this far beyond each marker along the edge
PLATE_LENGTH_M = 0.12776         # extent away from the markers (SBS standard long side)
PLATE_RIGHT_MARGIN_M = 0.010     # marker centre line -> plate's right edge (measure this)
PLATE_HEIGHT_M = 0.0144          # SBS plate height; set 0 to draw only the top face
PLATE_BOX_COLOUR = (0, 215, 255)  # BGR (yellow-orange)

# Well grid covering the box's top face exactly: GRID_ROWS equal cells along the
# marker edge, GRID_COLS equal cells across the plate.
DRAW_WELL_GRID = True
GRID_ROWS, GRID_COLS = 8, 12
GRID_COLOUR = (0, 255, 255)       # BGR (yellow)
GRID_ALPHA = 0.45                 # opacity of the grid lines (0..1)
GRID_DASH_SAMPLES = 60            # samples per grid line; every other segment is drawn (dashes)

# Ray cast from this marker's centre along one of its axes. If the ray hits the
# plate box, the hit cell is filled in and the distance along the ray is reported.
DISTANCE_MARKER_ID = 30
RAY_AXIS = "x"                    # marker axis to cast along: "x" (red), "y" (green), "z" (blue)
RAY_AXIS_SIGN = -1                # +1: positive direction, -1: negative direction
DISTANCE_COLOUR = (255, 0, 255)   # BGR (magenta): ray from marker to hit point
# Grid cell hit by the ray is filled in when the ray distance is below this threshold
HIT_DIST_THRESHOLD_M = 0.30
HIT_CELL_COLOUR = (0, 255, 0)     # BGR (green)
HIT_CELL_ALPHA = 0.5              # opacity of the filled cell (0..1)

# Abstract view: same size as the frames, black background, white lines
# (hit cell filled in colour). Shows
# only the plate box, well grid, hit cell, marker DISTANCE_MARKER_ID's position
# and (when it hits the box) its ray with the distance.
SAVE_ABSTRACT = True
ABSTRACT_COLOUR = (255, 255, 255)  # BGR (white)
ABSTRACT_GRID_ALPHA = 0.45         # opacity of the grid lines
ABSTRACT_CELL_COLOUR = HIT_CELL_COLOUR  # hit cell is filled in colour (green)
ABSTRACT_CELL_ALPHA = 0.6          # opacity of the filled hit cell

# Markers detected but not in TAG_SIZE_M: outlined in this colour, not logged
OTHER_MARKER_COLOUR = (0, 0, 255)  # BGR (red)

# How the frames were exported relative to the raw sensor image:
# Aria images are often rotated 90° clockwise for viewing ("upright"). Set this
# to True if your frames are upright; leave False if they're the raw sensor orientation.
FRAMES_ROTATED_CW90 = False

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}   # column of the rotation matrix
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ----------------------------------------------------------------------------- /CONFIG


def load_camera_calibration(vrs_path: Path, label: str):
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise SystemExit(f"Could not open VRS file: {vrs_path}")
    device_calib = provider.get_device_calibration()
    cam_calib = device_calib.get_camera_calib(label)
    if cam_calib is None:
        raise SystemExit(
            f"No calibration for '{label}'. Available: {device_calib.get_camera_labels()}"
        )
    return cam_calib


class FrameToCalibMapping:
    """Maps pixels between the exported frame and the calibration's raw image
    (handles optional 90° CW rotation and uniform resizing)."""

    def __init__(self, frame_wh, calib_wh, rotated_cw90):
        self.rot = rotated_cw90
        fw, fh = frame_wh
        # Size of the frame once un-rotated back into sensor orientation
        self.uw, self.uh = (fh, fw) if rotated_cw90 else (fw, fh)
        cw, ch = calib_wh
        self.sx, self.sy = cw / self.uw, ch / self.uh
        if abs(self.sx - self.sy) > 1e-3:
            tqdm.write(
                f"WARNING: non-uniform scale frame->calib ({self.sx:.4f}, {self.sy:.4f}). "
                "Check FRAMES_ROTATED_CW90 or whether frames were cropped."
            )

    def to_calib(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.rot:  # inverse of CW rotation: (x', y') -> (y', uh-1-x')
            pts = np.stack([pts[:, 1], self.uh - 1 - pts[:, 0]], axis=1)
        return pts * [self.sx, self.sy]

    def to_frame(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2) / [self.sx, self.sy]
        if self.rot:  # CW rotation: (x, y) -> (uh-1-y, x)
            pts = np.stack([self.uh - 1 - pts[:, 1], pts[:, 0]], axis=1)
        return pts


def marker_object_points(size):
    """3D corners in the marker frame, in ArUco detection order (TL, TR, BR, BL)."""
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def estimate_marker_pose(corners_calib_px, size, cam_calib):
    """Pose of one square marker from its 4 corners (calibration pixel coords).

    The Aria RGB camera uses a fisheye (FisheyeRadTanThinPrism / Fisheye624)
    model that OpenCV doesn't support, so we unproject each corner into a ray
    with the Aria model, normalise to z=1, and solve PnP with an identity
    intrinsic matrix and no distortion.
    """
    norm = []
    for p in corners_calib_px:
        ray = cam_calib.unproject_no_checks(np.asarray(p, dtype=np.float64))
        ray = np.asarray(ray, dtype=np.float64).reshape(3)
        if ray[2] <= 1e-6:
            return None
        norm.append(ray[:2] / ray[2])
    norm = np.array(norm, dtype=np.float64)

    obj = marker_object_points(size)
    ok, rvec, tvec = cv2.solvePnP(
        obj, norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, tvec.ravel()

    # Reprojection error through the full Aria camera model
    proj = project_points(cam_calib, (R @ obj.T).T + tvec.ravel())
    err = (np.linalg.norm(proj - corners_calib_px, axis=1).mean()
           if not np.isnan(proj).any() else float("nan"))
    return T, err


def project_points(cam_calib, pts_cam):
    out = []
    for p in pts_cam:
        if p[2] <= 1e-6:
            out.append([np.nan, np.nan])
            continue
        uv = cam_calib.project_no_checks(np.asarray(p, dtype=np.float64))
        out.append(np.asarray(uv, dtype=np.float64).reshape(2))
    return np.array(out)


def draw_axes(image, T_cam_marker, length, cam_calib, mapping, thickness=2):
    pts = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
    pts_cam = (T_cam_marker[:3, :3] @ pts.T).T + T_cam_marker[:3, 3]
    uv = project_points(cam_calib, pts_cam)
    if np.isnan(uv).any():
        return
    uv = mapping.to_frame(uv).round().astype(int)
    o = tuple(uv[0])
    for end, colour in zip(uv[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # X red, Y green, Z blue
        cv2.line(image, o, tuple(end), colour, thickness, cv2.LINE_AA)


def plate_corners_cam(T_cam_top, T_cam_bottom):
    """8 plate box corners in the camera frame, from the camera-frame poses of the
    top-right and bottom-right plate markers:
      [0:4]  top face    (top-right, bottom-right, bottom-left, top-left)
      [4:8]  bottom face (same order)"""
    p_t, p_b = T_cam_top[:3, 3], T_cam_bottom[:3, 3]
    v = p_b - p_t                       # along the right edge, top -> bottom
    v /= np.linalg.norm(v)
    n = T_cam_top[:3, 2] + T_cam_bottom[:3, 2]   # average marker normal (plate "up")
    if np.dot(n, (p_t + p_b) / 2) > 0:  # make it point towards the camera
        n = -n
    n -= np.dot(n, v) * v               # orthogonalise against the edge direction
    n /= np.linalg.norm(n)
    u = np.cross(v, n)                  # in the plate plane, pointing into the plate body

    a = p_t - PLATE_EDGE_EXTENSION_M * v
    b = p_b + PLATE_EDGE_EXTENSION_M * v
    right = -PLATE_RIGHT_MARGIN_M * u
    left = (PLATE_LENGTH_M - PLATE_RIGHT_MARGIN_M) * u
    top = [a + right, b + right, b + left, a + left]
    bottom = [c - PLATE_HEIGHT_M * n for c in top]
    return np.array(top + bottom)


def transform_points(T, pts):
    return (T[:3, :3] @ np.asarray(pts).T).T + T[:3, 3]


def draw_line_3d(image, p0, p1, cam_calib, mapping, colour, thickness, samples=20):
    """Draw a straight 3D segment, sampled so it curves correctly under fisheye distortion."""
    pts = np.linspace(p0, p1, samples)
    uv = project_points(cam_calib, pts)
    if np.isnan(uv).any():
        return
    uv = mapping.to_frame(uv).round().astype(np.int32)
    cv2.polylines(image, [uv.reshape(-1, 1, 2)], False, colour, thickness, cv2.LINE_AA)


def draw_dashed_line_3d(image, p0, p1, cam_calib, mapping, colour, thickness,
                        samples=GRID_DASH_SAMPLES):
    """Dashed 3D segment: sampled finely (fisheye-correct), every other piece drawn."""
    uv = project_points(cam_calib, np.linspace(p0, p1, samples + 1))
    if np.isnan(uv).any():
        return
    uv = mapping.to_frame(uv).round().astype(int)
    for k in range(0, samples, 2):
        cv2.line(image, tuple(uv[k]), tuple(uv[k + 1]), colour, thickness, cv2.LINE_AA)


def draw_well_grid(image, plate_points, cam_calib, mapping,
                   colour=GRID_COLOUR, alpha=GRID_ALPHA):
    """Semi-transparent dashed GRID_ROWS x GRID_COLS grid covering the box's top face."""
    g0, g1, _, g3 = plate_points[0:4]
    ev, eu = g1 - g0, g3 - g0            # along the marker edge (rows), across the plate (cols)
    overlay = image.copy()
    for i in range(GRID_ROWS + 1):        # lines separating rows
        a = g0 + (i / GRID_ROWS) * ev
        draw_dashed_line_3d(overlay, a, a + eu, cam_calib, mapping, colour, 1)
    for j in range(GRID_COLS + 1):        # lines separating columns
        a = g0 + (j / GRID_COLS) * eu
        draw_dashed_line_3d(overlay, a, a + ev, cam_calib, mapping, colour, 1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def draw_plate_box(image, corners_cam, cam_calib, mapping, label, colour=PLATE_BOX_COLOUR):
    top, bottom = corners_cam[:4], corners_cam[4:8]
    for i in range(4):
        j = (i + 1) % 4
        draw_line_3d(image, top[i], top[j], cam_calib, mapping, colour, 3)
        if PLATE_HEIGHT_M > 0:
            draw_line_3d(image, bottom[i], bottom[j], cam_calib, mapping, colour, 1)
            draw_line_3d(image, top[i], bottom[i], cam_calib, mapping, colour, 1)
    uv = project_points(cam_calib, top[3:4])
    if not np.isnan(uv).any():
        x, y = mapping.to_frame(uv)[0].round().astype(int)
        cv2.putText(image, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    colour, 2, cv2.LINE_AA)


def ray_box_intersection(origin, direction, corners, eps=1e-9):
    """Intersect a ray with the solid box given by plate_corners_cam()
    (corner 0 is the box origin; edges 0->1, 0->3 and 0->4 span the box).
    Slab method. Returns (t_enter, t_exit) along the unit `direction`, with
    t_enter clamped to 0 if the ray starts inside the box, or None if it misses."""
    c0 = corners[0]
    t_min, t_max = 0.0, np.inf
    for e in (corners[1] - c0, corners[3] - c0, corners[4] - c0):
        L = np.linalg.norm(e)
        if L < 1e-9:              # degenerate axis (PLATE_HEIGHT_M = 0): flat slab
            e_hat = np.cross(corners[1] - c0, corners[3] - c0)
            e_hat /= np.linalg.norm(e_hat)
        else:
            e_hat = e / L
        s0 = np.dot(origin - c0, e_hat)   # ray origin coordinate along this axis
        ds = np.dot(direction, e_hat)     # ray speed along this axis
        if abs(ds) < eps:                 # parallel to this slab
            if s0 < -eps or s0 > L + eps:
                return None
            continue
        t1, t2 = (0.0 - s0) / ds, (L - s0) / ds
        if t1 > t2:
            t1, t2 = t2, t1
        t_min, t_max = max(t_min, t1), min(t_max, t2)
        if t_min > t_max:
            return None
    return t_min, t_max


def grid_cell_at(p, plate_points):
    """(row, col) of the grid cell on the top face containing point p
    (p is projected onto the top face; points on the box sides map to the edge cells)."""
    g0, g1, _, g3 = plate_points[0:4]
    ev, eu = g1 - g0, g3 - g0
    a = np.clip(np.dot(p - g0, ev) / np.dot(ev, ev), 0.0, 1.0 - 1e-9)
    b = np.clip(np.dot(p - g0, eu) / np.dot(eu, eu), 0.0, 1.0 - 1e-9)
    return int(a * GRID_ROWS), int(b * GRID_COLS)


def fill_grid_cell(image, plate_points, row, col, cam_calib, mapping, samples=8,
                   colour=HIT_CELL_COLOUR, alpha=HIT_CELL_ALPHA):
    """Semi-transparent fill of one grid cell (edges sampled for fisheye correctness)."""
    g0, g1, _, g3 = plate_points[0:4]
    ev, eu = (g1 - g0) / GRID_ROWS, (g3 - g0) / GRID_COLS
    c00 = g0 + row * ev + col * eu
    quad = [c00, c00 + ev, c00 + ev + eu, c00 + eu]
    pts = np.concatenate([np.linspace(quad[k], quad[(k + 1) % 4], samples, endpoint=False)
                          for k in range(4)])
    uv = project_points(cam_calib, pts)
    if np.isnan(uv).any():
        return
    uv = mapping.to_frame(uv).round().astype(np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [uv.reshape(-1, 1, 2)], colour, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def draw_ray(image, p_marker, p_enter, dist, cam_calib, mapping, colour=DISTANCE_COLOUR):
    # Ray from the marker to where it enters the box
    draw_line_3d(image, p_marker, p_enter, cam_calib, mapping, colour, 2)
    uv = project_points(cam_calib, np.array([p_marker, p_enter, (p_marker + p_enter) / 2]))
    if np.isnan(uv).any():
        return
    uv = mapping.to_frame(uv).round().astype(int)
    cv2.circle(image, tuple(uv[0]), 6, colour, -1, cv2.LINE_AA)
    cv2.circle(image, tuple(uv[1]), 4, colour, -1, cv2.LINE_AA)
    text = f"{dist * 100:.1f} cm"
    org = (int(uv[2][0]) + 10, int(uv[2][1]))
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)


def draw_marker_position(image, p, label, cam_calib, mapping, colour):
    uv = project_points(cam_calib, np.array([p]))
    if np.isnan(uv).any():
        return
    x, y = mapping.to_frame(uv)[0].round().astype(int)
    cv2.circle(image, (x, y), 8, colour, -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), 14, colour, 1, cv2.LINE_AA)
    cv2.putText(image, label, (x + 18, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                colour, 2, cv2.LINE_AA)


def rot_to_quat_wxyz(R):
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def main():
    frame_paths = sorted(p for p in FRAMES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not frame_paths:
        raise SystemExit(f"No images found in {FRAMES_DIR}")
    selected = frame_paths[START_IDX:END_IDX]

    cam_calib = load_camera_calibration(VRS_PATH, CAMERA_LABEL)
    calib_wh = tuple(int(v) for v in cam_calib.get_image_size())
    T_device_camera = cam_calib.get_transform_device_camera().to_matrix()
    print(f"Calibration '{CAMERA_LABEL}': {calib_wh[0]}x{calib_wh[1]}, "
          f"{len(selected)} frames [{START_IDX}:{END_IDX}] of {len(frame_paths)}")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # better corners → better poses
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    wanted = set(TAG_SIZE_M)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated_dir = OUTPUT_DIR / "annotated"
    if SAVE_ANNOTATED:
        annotated_dir.mkdir(exist_ok=True)
    abstract_dir = OUTPUT_DIR / "abstract"
    if SAVE_ABSTRACT:
        abstract_dir.mkdir(exist_ok=True)

    header = ["frame_idx", "frame", "marker_id", "object",
              "global_x", "global_y", "global_z",
              "global_qw", "global_qx", "global_qy", "global_qz",
              "anchor", "reproj_err_px",
              "cam_tx", "cam_ty", "cam_tz", "cam_qw", "cam_qx", "cam_qy", "cam_qz",
              "dev_tx", "dev_ty", "dev_tz", "dev_qw", "dev_qx", "dev_qy", "dev_qz",
              "ray_dist_to_plate_m"]   # DISTANCE_MARKER_ID rows only; blank if the ray misses
    mapping, n_det = None, 0
    T_global_cam = None   # inv(pose of marker 36), from the current or most recent frame
    anchor_counts = {"current": 0, "most_recent": 0}
    # Plate corners cached in each plate marker's own frame, so the box can still
    # be drawn when only one of the two markers is visible.
    plate_in_marker = {}   # marker_id -> (8, 3) box corners in that marker's frame

    with open(OUTPUT_DIR / "poses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, path in enumerate(tqdm(selected, desc="Frames", unit="frame"), start=START_IDX):
            image = cv2.imread(str(path))
            if image is None:
                tqdm.write(f"Could not read {path}, skipping")
                continue
            if mapping is None or (mapping.uw, mapping.uh) != (
                (image.shape[0], image.shape[1]) if FRAMES_ROTATED_CW90
                else (image.shape[1], image.shape[0])
            ):
                mapping = FrameToCalibMapping(
                    (image.shape[1], image.shape[0]), calib_wh, FRAMES_ROTATED_CW90
                )

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            # 1) Detect + estimate camera-frame poses for all wanted markers in this frame
            keep_c, keep_i, poses = [], [], []
            other_c, other_i = [], []   # detected but not tracked: drawn only, not logged
            if ids is not None:
                for c, mid in zip(corners, ids.flatten()):
                    mid = int(mid)
                    if mid not in wanted:
                        other_c.append(c)
                        other_i.append(mid)
                        continue
                    keep_c.append(c)
                    keep_i.append(mid)

                    pts_calib = mapping.to_calib(c.reshape(4, 2))
                    res = estimate_marker_pose(pts_calib, TAG_SIZE_M[mid], cam_calib)
                    if res is not None:
                        poses.append((mid, *res))

            # 2) Centre the global frame on marker 36 in this frame, or the most recent one
            origin = [T for mid, T, _ in poses if mid == ORIGIN_MARKER_ID]
            if origin:
                T_global_cam, anchor = np.linalg.inv(origin[0]), "current"
            elif T_global_cam is not None:
                anchor = "most_recent"
            else:
                raise SystemExit(
                    f"Origin marker {ORIGIN_MARKER_ID} not found in first frame "
                    f"{path.name} (index {idx}). Choose a different START_IDX."
                )
            anchor_counts[anchor] += 1

            # 3) Plate bounding box from markers 36 (top-right) and 26 (bottom-right)
            T_cam = {mid: T for mid, T, _ in poses}
            plate_corners, plate_label = None, None
            if PLATE_TOP_ID in T_cam and PLATE_BOTTOM_ID in T_cam:
                plate_corners = plate_corners_cam(T_cam[PLATE_TOP_ID], T_cam[PLATE_BOTTOM_ID])
                for mid in (PLATE_TOP_ID, PLATE_BOTTOM_ID):
                    plate_in_marker[mid] = transform_points(np.linalg.inv(T_cam[mid]), plate_corners)
                plate_label = "plate"
            else:
                for mid in (PLATE_TOP_ID, PLATE_BOTTOM_ID):
                    if mid in T_cam and mid in plate_in_marker:
                        plate_corners = transform_points(T_cam[mid], plate_in_marker[mid])
                        plate_label = f"plate (from {mid})"
                        break

            # 4) Cast a ray along the DISTANCE_MARKER_ID's RAY_AXIS and intersect it with the plate box
            dist, p_marker, p_enter, hit_cell = None, None, None, None
            if plate_corners is not None and DISTANCE_MARKER_ID in T_cam:
                p_marker = T_cam[DISTANCE_MARKER_ID][:3, 3]
                ray_dir = RAY_AXIS_SIGN * T_cam[DISTANCE_MARKER_ID][:3, AXIS_INDEX[RAY_AXIS]]
                ray_dir /= np.linalg.norm(ray_dir)
                hit = ray_box_intersection(p_marker, ray_dir, plate_corners)
                if hit is not None:
                    dist = hit[0]
                    p_enter = p_marker + dist * ray_dir
                    if dist < HIT_DIST_THRESHOLD_M:
                        hit_cell = grid_cell_at(p_enter, plate_corners)

            # 5) Write rows: global, camera and device-frame poses
            for mid, T_cam_marker, err in poses:
                T_global_marker = T_global_cam @ T_cam_marker
                T_dev_marker = T_device_camera @ T_cam_marker
                n_det += 1

                writer.writerow([
                    idx, path.name, mid, MARKER_TO_OBJECT[mid],
                    *(f"{v:.6f}" for v in T_global_marker[:3, 3]),
                    *(f"{v:.6f}" for v in rot_to_quat_wxyz(T_global_marker[:3, :3])),
                    anchor,
                    f"{err:.3f}",
                    *(f"{v:.6f}" for v in T_cam_marker[:3, 3]),
                    *(f"{v:.6f}" for v in rot_to_quat_wxyz(T_cam_marker[:3, :3])),
                    *(f"{v:.6f}" for v in T_dev_marker[:3, 3]),
                    *(f"{v:.6f}" for v in rot_to_quat_wxyz(T_dev_marker[:3, :3])),
                    f"{dist:.6f}" if (mid == DISTANCE_MARKER_ID and dist is not None) else "",
                ])
                if SAVE_ANNOTATED:
                    draw_axes(image, T_cam_marker, TAG_SIZE_M[mid] * 0.75, cam_calib, mapping)

            # 6) Draw grid, hit cell, plate box and ray on the annotated frame
            if SAVE_ANNOTATED and DRAW_WELL_GRID and plate_corners is not None:
                draw_well_grid(image, plate_corners, cam_calib, mapping)
            if SAVE_ANNOTATED and hit_cell is not None:
                fill_grid_cell(image, plate_corners, *hit_cell, cam_calib, mapping)
            if SAVE_ANNOTATED and DRAW_PLATE_BOX and plate_corners is not None:
                draw_plate_box(image, plate_corners, cam_calib, mapping, plate_label)
            if SAVE_ANNOTATED and dist is not None:
                draw_ray(image, p_marker, p_enter, dist, cam_calib, mapping)

            if SAVE_ANNOTATED:
                if keep_c:
                    cv2.aruco.drawDetectedMarkers(image, keep_c, np.array(keep_i).reshape(-1, 1))
                if other_c:
                    cv2.aruco.drawDetectedMarkers(image, other_c, np.array(other_i).reshape(-1, 1),
                                                  borderColor=OTHER_MARKER_COLOUR)
                cv2.imwrite(str(annotated_dir / f"{path.stem}.jpg"), image)

            # 7) Abstract view: white on black, same dimensions as the frame
            if SAVE_ABSTRACT:
                abstract = np.zeros_like(image)
                W = ABSTRACT_COLOUR
                if plate_corners is not None:
                    draw_well_grid(abstract, plate_corners, cam_calib, mapping,
                                   colour=W, alpha=ABSTRACT_GRID_ALPHA)
                    if hit_cell is not None:
                        fill_grid_cell(abstract, plate_corners, *hit_cell, cam_calib, mapping,
                                       colour=ABSTRACT_CELL_COLOUR, alpha=ABSTRACT_CELL_ALPHA)
                    draw_plate_box(abstract, plate_corners, cam_calib, mapping, plate_label,
                                   colour=W)
                if dist is not None:
                    draw_ray(abstract, p_marker, p_enter, dist, cam_calib, mapping, colour=W)
                if DISTANCE_MARKER_ID in T_cam:
                    draw_marker_position(abstract, T_cam[DISTANCE_MARKER_ID][:3, 3],
                                         str(DISTANCE_MARKER_ID), cam_calib, mapping, W)
                cv2.imwrite(str(abstract_dir / f"{path.stem}.jpg"), abstract)

    print(f"Done: {n_det} marker poses written to {OUTPUT_DIR / 'poses.csv'}")
    print("Frames per anchor type: " + ", ".join(f"{k}={v}" for k, v in anchor_counts.items()))


if __name__ == "__main__":
    main()