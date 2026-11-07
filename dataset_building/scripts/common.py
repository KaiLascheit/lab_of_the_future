"""Shared helpers: VRS/calibration access, marker pose, rectified crops, label files."""
import csv
from pathlib import Path

import cv2
import numpy as np

import config as C


# ----------------------------------------------------------------------------- sessions
def session_cfg(name):
    if name not in C.SESSIONS:
        raise SystemExit(f"Unknown session '{name}'. Known: {list(C.SESSIONS)}")
    return C.SESSIONS[name]


def session_names(args):
    """Session names from the command line, or all configured sessions."""
    names = args or list(C.SESSIONS)
    for n in names:
        session_cfg(n)
    return names


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else C.PROJECT_ROOT / p


def marker_id(name):
    return session_cfg(name).get("marker_id", C.PIPETTE_MARKER_ID)


# ----------------------------------------------------------------------------- VRS
def open_vrs(vrs_path):
    """Returns (provider, rgb_stream_id, camera_calibration)."""
    from projectaria_tools.core import data_provider   # imported lazily (only extract needs it)

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise SystemExit(f"Could not open VRS file: {vrs_path}")
    sid = provider.get_stream_id_from_label(C.CAMERA_LABEL)
    if sid is None:
        raise SystemExit(f"No '{C.CAMERA_LABEL}' stream in {vrs_path}")
    device_calib = provider.get_device_calibration()
    cam = device_calib.get_camera_calib(C.CAMERA_LABEL)
    if cam is None:
        raise SystemExit(f"No calibration for '{C.CAMERA_LABEL}' in {vrs_path}")
    return provider, sid, cam


def read_rgb_frame(provider, sid, i):
    """(BGR image, capture timestamp ns) for RGB frame i, in raw sensor orientation."""
    image_data, record = provider.get_image_data_by_index(sid, i)
    img = image_data.to_numpy_array()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, int(record.capture_timestamp_ns)


# ----------------------------------------------------------------------------- markers
def make_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)


def project(cam, pts_cam, scale=1.0):
    """3D camera-frame points -> frame pixels (NaN where behind the camera).
    `scale` maps calibration pixels to frame pixels if the frame was resized."""
    pts_cam = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
    out = np.full((len(pts_cam), 2), np.nan)
    for k, p in enumerate(pts_cam):
        if p[2] > 1e-6:
            out[k] = np.asarray(cam.project_no_checks(p)).reshape(2)
    return out * scale


def estimate_pose(corners_px, size, cam, scale=1.0):
    """Marker pose from its 4 detected corners using the Aria fisheye model.
    Returns dict(T=4x4 camera-from-marker, reproj_px, ambiguity, dist_m, view_deg) or None.
    ambiguity = error ratio of the two IPPE solutions (close to 1 = orientation unreliable)."""
    corners_calib = np.asarray(corners_px, dtype=np.float64).reshape(4, 2) / scale
    norm = []
    for p in corners_calib:
        ray = np.asarray(cam.unproject_no_checks(p)).reshape(3)
        if ray[2] <= 1e-6:
            return None
        norm.append(ray[:2] / ray[2])
    h = size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, np.array(norm), np.eye(3), None,
                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if n == 0:
        return None
    errs = np.asarray(errs).ravel()
    best = int(np.argmin(errs))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvecs[best])[0]
    T[:3, 3] = np.asarray(tvecs[best]).ravel()
    proj = project(cam, (T[:3, :3] @ obj.T).T + T[:3, 3])
    dist = float(np.linalg.norm(T[:3, 3]))
    view = float(np.degrees(np.arccos(np.clip(np.dot(T[:3, 2], -T[:3, 3] / dist), -1, 1))))
    return {
        "T": T,
        "reproj_px": float(np.linalg.norm(proj - corners_calib, axis=1).mean()),
        "ambiguity": float(errs.min() / errs.max()) if n > 1 and errs.max() > 0 else 0.0,
        "dist_m": dist,
        "view_deg": view,
    }


# ----------------------------------------------------------------------------- crops
def crop_axes():
    """Marker-plane coordinates (metres) of the crop's columns and rows, so that the
    pipette is upright: top row = furthest up the pipette."""
    (x0, x1), (y0, y1) = C.BOX_X, C.BOX_Y
    W = int(round((x1 - x0) * 1000 * C.CROP_PX_PER_MM))
    H = int(round((y1 - y0) * 1000 * C.CROP_PX_PER_MM))
    if C.UP_SIGN < 0:
        xs, ys = np.linspace(x1, x0, W), np.linspace(y0, y1, H)
    else:
        xs, ys = np.linspace(x0, x1, W), np.linspace(y1, y0, H)
    return xs, ys


def rectified_crop(img, T, cam, scale=1.0):
    """Resample the marker plane over BOX_X x BOX_Y into an upright crop.
    Exact fisheye projection on a coarse grid (CROP_GRID_STEP_MM), bilinearly
    interpolated to full resolution - accurate and fast."""
    xs, ys = crop_axes()
    W, H = len(xs), len(ys)
    step = max(1, int(C.CROP_GRID_STEP_MM * C.CROP_PX_PER_MM))
    cols = np.unique(np.r_[np.arange(0, W, step), W - 1])
    rows = np.unique(np.r_[np.arange(0, H, step), H - 1])
    gx, gy = np.meshgrid(xs[cols], ys[rows])
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, C.CROP_PLANE_Z)], axis=1)
    uv = project(cam, (T[:3, :3] @ pts.T).T + T[:3, 3], scale)
    if np.isnan(uv).any():
        return None
    map_x = uv[:, 0].reshape(len(rows), len(cols)).astype(np.float32)
    map_y = uv[:, 1].reshape(len(rows), len(cols)).astype(np.float32)
    # Interpolate the coarse maps onto every output pixel (separable linear interp)
    map_x = _interp_grid(map_x, rows, cols, H, W)
    map_y = _interp_grid(map_y, rows, cols, H, W)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _interp_grid(m, rows, cols, H, W):
    tmp = np.stack([np.interp(np.arange(W), cols, r) for r in m])            # (len(rows), W)
    out = np.stack([np.interp(np.arange(H), rows, tmp[:, c]) for c in range(W)], axis=1)
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- index
INDEX_FIELDS = ["frame_idx", "timestamp_ns", "crop", "marker_found", "reproj_px",
                "ambiguity", "dist_m", "view_deg"]


def read_index(session):
    path = C.CROPS_DIR / session / "index.csv"
    if not path.exists():
        raise SystemExit(f"No crop index for '{session}' ({path}). Run extract_crops.py first.")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["frame_idx"] = int(r["frame_idx"])
        r["marker_found"] = r["marker_found"] == "1"
    return rows


# ----------------------------------------------------------------------------- labels
def labels_path(session):
    return C.LABELS_DIR / f"{session}.csv"


def load_change_points(session):
    """{frame_idx: label}: from that frame on, frames carry that label (until the next point)."""
    path = labels_path(session)
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {int(r["start_frame"]): r["label"] for r in csv.DictReader(f)}


def save_change_points(session, points):
    C.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(labels_path(session), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_frame", "label"])
        for k in sorted(points):
            w.writerow([k, points[k]])


def expand_labels(points, frame_indices):
    """Per-frame labels for the given (sorted) frame indices."""
    keys = sorted(points)
    out, j, current = [], -1, C.UNLABELLED
    for fi in frame_indices:
        while j + 1 < len(keys) and keys[j + 1] <= fi:
            j += 1
            current = points[keys[j]]
        out.append(current)
    return out
