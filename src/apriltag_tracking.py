"""
Detect AprilTag markers on the pipette (IDs 30, 23) and 96-well plate (IDs 26, 36)
in every frame of a directory, estimate each marker's 6-DoF pose using the
RGB camera calibration stored in an Aria Gen 2 VRS recording, and fuse the
pipette's two markers into one rigid-body pose.

Install:  pip install opencv-contrib-python tqdm projectaria-tools scipy
          (+ for the press classifier: torch torchvision scikit-learn joblib pandas)
Usage:    python detect_apriltags.py
          (edit the CONFIG block below: paths, frame range, tag size, classifier)

Outputs:
  OUTPUT_DIR/annotated/<frame>.jpg   frames with outlines, IDs, marker axes, the
                                     (thicker, longer) fused pipette axes and, when
                                     the ray hits the plate, the press label
                                     (banner, hit cell in red/green, crop inset)
  OUTPUT_DIR/abstract/<frame>.jpg    same-size white-on-black view of plate box,
                                     grid, hit cell, pipette origin, ray and label
  OUTPUT_DIR/press_crops/<frame>.png the plunger crops fed to the classifier
  OUTPUT_DIR/poses.csv               one row per detected marker per frame, plus one
                                     row per rigid body (source = fused / single);
                                     the pipette row also holds ray distance, hit
                                     cell and press_p_raw / press_p_smoothed /
                                     press_label (pressed, not_pressed, no_crop)
  BODY_CALIB_FILE                    fixed marker-to-marker offsets of each rigid body
                                     (estimated automatically on the first run)

Multi-marker rigid bodies (pipette = markers 30 + 23):
  1. Calibration (once). Both tags are stuck to the same rigid object, so the
     transform T_ref_m from the reference tag (30) to every other tag (23) is
     constant. It is estimated from all frames where both tags are seen with a
     low reprojection error, using a robust average (medoid + inlier mean, which
     rejects the occasional flipped single-tag pose), and cached in
     BODY_CALIB_FILE. Delete that file (or set RECALIBRATE_BODIES = True) after
     moving or re-sticking a tag.
  2. Fusion (every frame). The corners of all visible body tags are expressed in
     the body frame (= the reference tag's frame) and a single PnP is solved over
     all of them (8 corners for two tags). The long baseline between the tags
     constrains orientation much better than one 2 cm tag and removes the
     planar-tag flip ambiguity. If only one tag is visible, the body pose comes
     from that tag plus the calibrated offset, so tracking survives occlusion.

Pose conventions:
  * Marker frame: origin at tag centre, X right, Y up, Z out of the tag
    towards the camera (OpenCV ArUco convention).
  * Body frame: identical to its reference marker's frame (pipette: marker 30),
    so axis-based settings like RAY_AXIS keep their meaning.
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
import json
import sys

import cv2
import numpy as np
from tqdm import tqdm
from projectaria_tools.core import data_provider

# ----------------------------------------------------------------------------- CONFIG
stream = "Well_plate_p1000_2QR_2_20261107_162219"
FRAMES_DIR = Path(f"frames/{stream}")
VRS_PATH = Path(f"streams/{stream}.vrs")
OUTPUT_DIR = Path(f"tracking_results/{stream}")

START_IDX = 700      # first frame index (into the sorted frame list)
END_IDX = 800     # one past the last frame index; None = run to the end
SAVE_ANNOTATED = True

CAMERA_LABEL = "camera-rgb"   # Aria RGB camera stream label

# Black-border side length of each tag in METRES (measure your printed tags!).
TAG_SIZE_M = {
    30: 0.020, 23: 0.020,     # pipette
    26: 0.020, 36: 0.020,     # 96-well plate
}
MARKER_TO_OBJECT = {30: "pipette", 23: "pipette", 26: "plate", 36: "plate"}

# Rigid bodies tracked with several markers. The body frame is the frame of
# "ref"; the offsets of the other markers are calibrated automatically.
RIGID_BODIES = {
    "pipette": {"markers": [30, 23], "ref": 30},
}
BODY_CALIB_FILE = Path("rigid_body_calibration.json")  # kept outside OUTPUT_DIR on purpose
RECALIBRATE_BODIES = False     # True: ignore BODY_CALIB_FILE and re-estimate the offsets
CALIB_MAX_REPROJ_PX = 1.5      # preferred: single-tag poses below this error (px) ...
CALIB_MIN_FRAMES = 10          # ... but if fewer frames pass, use the best co-visible frames
CALIB_MIN_FRAMES_HARD = 3      # below this many co-visible frames the body can't be calibrated
CALIB_REFINE_BA = True         # refine the offset on raw corners of all frames (needs scipy)
CALIB_INLIER_DEG = 10.0        # inlier gate around the medoid offset (rotation)
CALIB_INLIER_M = 0.010         # inlier gate around the medoid offset (translation)

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

# Ray cast from the body origin (centre of its ref marker). If the ray hits the
# plate box, the hit cell is filled in and the distance along the ray is reported.
DISTANCE_BODY = "pipette"
# "marker_line": from the ref marker's centre through RAY_LINE_TO_ID's centre
#                (with 30 above 23 on the shaft, this runs down the pipette towards
#                the tip, independent of how the stickers are rotated)
# "axis":        along the body-frame axis RAY_AXIS (sign RAY_AXIS_SIGN); depends
#                on the sticker's rotation. Also used as the fallback for
#                "marker_line" when RAY_LINE_TO_ID's offset isn't calibrated.
RAY_DIRECTION = "marker_line"
RAY_LINE_TO_ID = 23
RAY_AXIS = "y"                    # body axis to cast along: "x" (red), "y" (green), "z" (blue)
RAY_AXIS_SIGN = +1                # +1: positive direction, -1: negative direction
DISTANCE_COLOUR = (255, 0, 255)   # BGR (magenta): ray from marker to hit point
# Grid cell hit by the ray is filled in when the ray distance is below this threshold
HIT_DIST_THRESHOLD_M = 0.30
HIT_CELL_COLOUR = (0, 255, 0)     # BGR (green)
HIT_CELL_ALPHA = 0.5              # opacity of the filled cell (0..1)

# Button-press classifier (model.joblib from train_eval.py). It runs only on frames
# where the ray hits the plate box closer than HIT_DIST_THRESHOLD_M. The plunger
# crop is made exactly as in extract_crops.py (common.estimate_pose +
# common.rectified_crop on CROP_MARKER_ID), so PRESS_PROJECT_DIR must contain
# common.py, config.py and train_eval.py.
CLASSIFY_PRESS = True
PRESS_PROJECT_DIR = Path("dataset_building/scripts")   # common.py / config.py / train_eval.py
CLASSIFIER_RUN = "dinov2_vits14"   # results folder of train_eval.py (--run-name, default backbone)
CLASSIFIER_PATH = None        # None: <config.RESULTS_DIR>/<CLASSIFIER_RUN>/model.joblib
CROP_MARKER_ID = 30           # pipette marker whose frame the crop box is in
# Session in config.SESSIONS whose crop box ("box_x"/"box_y"/"up_sign", else the global
# BOX_X/BOX_Y/UP_SIGN) is used, so crops match that pipette's training crops. None = global box.
PRESS_CROP_SESSION = "p1000_162006"
# Pose used to place the crop:
#   "fused":  the fused pipette pose (needs CROP_MARKER_ID to be the body's "ref").
#             Steadier, and still works when only the other pipette tag is visible.
#   "single": CROP_MARKER_ID's own pose via common.estimate_pose, exactly as in
#             extract_crops.py.
CROP_POSE = "fused"
PRESS_THRESHOLD = None        # None: use the threshold saved with the model
PRESS_SMOOTH_WINDOW = 5       # centred median over classified frames, as in train_eval.py (1 = off)
PRESS_BATCH_SIZE = 32
SAVE_PRESS_CROPS = True       # write the crops fed to the model to OUTPUT_DIR/press_crops/
DRAW_CROP_INSET = True        # show the crop in the annotated frame's top-right corner
PRESSED_COLOUR = (0, 0, 255)          # BGR (red): hit cell + label when pressed
NOT_PRESSED_COLOUR = HIT_CELL_COLOUR  # green: hit cell + label when not pressed
UNKNOWN_COLOUR = (0, 200, 255)        # orange: ray hits but no crop (crop marker not seen)

# Abstract view: same size as the frames, black background, white lines
# (hit cell filled in colour). Shows only the plate box, well grid, hit cell,
# the DISTANCE_BODY origin and (when it hits the box) its ray with the distance.
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


# ----------------------------------------------------------------------------- geometry helpers
def marker_object_points(size):
    """3D corners in the marker frame, in ArUco detection order (TL, TR, BR, BL)."""
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def transform_points(T, pts):
    return (T[:3, :3] @ np.asarray(pts).T).T + T[:3, 3]


def rt_to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return T


def project_points(cam_calib, pts_cam):
    out = []
    for p in pts_cam:
        if p[2] <= 1e-6:
            out.append([np.nan, np.nan])
            continue
        uv = cam_calib.project_no_checks(np.asarray(p, dtype=np.float64))
        out.append(np.asarray(uv, dtype=np.float64).reshape(2))
    return np.array(out)


def unproject_normalized(cam_calib, pts_px):
    """Calibration pixels -> normalised z=1 coordinates via the Aria fisheye model."""
    out = []
    for p in pts_px:
        ray = cam_calib.unproject_no_checks(np.asarray(p, dtype=np.float64))
        ray = np.asarray(ray, dtype=np.float64).reshape(3)
        if ray[2] <= 1e-6:
            return None
        out.append(ray[:2] / ray[2])
    return np.array(out, dtype=np.float64)


def reprojection_error(cam_calib, T_cam_obj, obj_pts, px):
    """Mean corner reprojection error (calibration pixels) through the full Aria model."""
    proj = project_points(cam_calib, transform_points(T_cam_obj, obj_pts))
    if np.isnan(proj).any():
        return float("inf")
    return float(np.linalg.norm(proj - px, axis=1).mean())


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def project_to_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


# ----------------------------------------------------------------------------- single-marker pose
def estimate_marker_pose(corners_calib_px, size, cam_calib):
    """Pose of one square marker from its 4 corners (calibration pixel coords).

    The Aria RGB camera uses a fisheye (FisheyeRadTanThinPrism / Fisheye624)
    model that OpenCV doesn't support, so we unproject each corner into a ray
    with the Aria model, normalise to z=1, and solve PnP with an identity
    intrinsic matrix and no distortion.

    A small planar tag has two plausible poses (the IPPE "flip" ambiguity). Both
    are returned as candidates so the multi-marker fusion can pick the one that
    agrees with the other tags. Returns (T_best, err_best, [T_candidates]) or None.
    """
    norm = unproject_normalized(cam_calib, corners_calib_px)
    if norm is None:
        return None
    obj = marker_object_points(size)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj, norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not n:
        return None
    cands = [rt_to_T(r, t) for r, t in zip(rvecs, tvecs)]
    errs = [reprojection_error(cam_calib, T, obj, corners_calib_px) for T in cands]
    best = int(np.argmin(errs))
    if not np.isfinite(errs[best]):
        return None
    return cands[best], errs[best], cands


# ----------------------------------------------------------------------------- rigid bodies
def refine_offset_ba(T_init, ref, mid, frame_dets, cam_calib):
    """Bundle adjustment of T_ref_mid: jointly optimise the offset and one body
    pose per frame so that all 8 corners of both tags reproject onto their
    detections (robust Huber loss, residuals in ~pixels). Uses every co-visible
    frame, including blurry ones, instead of trusting noisy single-tag poses."""
    try:
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("    (scipy not installed: skipping bundle-adjustment refinement)")
        return T_init, None

    body = {"ref": ref, "T_ref_marker": {ref: np.eye(4), mid: T_init}}
    obs, poses, errs = [], [], []
    for det in frame_dets:
        if ref not in det["poses"] or mid not in det["poses"]:
            continue
        nr = unproject_normalized(cam_calib, det["px"][ref])
        nm = unproject_normalized(cam_calib, det["px"][mid])
        res = estimate_body_pose(body, det, cam_calib)
        if nr is None or nm is None or res is None:
            continue
        obs.append(np.concatenate([nr, nm]))
        poses.append(res[0])
        errs.append(res[1])
    if len(obs) < CALIB_MIN_FRAMES_HARD:
        return T_init, None
    keep = np.array(errs) < 3 * np.median(errs) + 1.0   # drop gross failures of the init
    obs = np.array(obs)[keep]
    poses = [p for p, k in zip(poses, keep) if k]
    n = len(poses)

    # Approximate focal length, so residuals (normalised coords) read as pixels
    f_px = np.linalg.norm(
        np.asarray(cam_calib.project_no_checks(np.array([0.01, 0.0, 1.0]))).ravel()
        - np.asarray(cam_calib.project_no_checks(np.array([0.0, 0.0, 1.0]))).ravel()) / 0.01
    obj_ref = marker_object_points(TAG_SIZE_M[ref])
    obj_mid = marker_object_points(TAG_SIZE_M[mid])

    def pack(T):
        return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]

    def residuals(x):
        Ro = Rotation.from_rotvec(x[:3]).as_matrix()
        P = np.concatenate([obj_ref, obj_mid @ Ro.T + x[3:6]])          # (8, 3) body frame
        F = x[6:].reshape(n, 6)
        Rf = Rotation.from_rotvec(F[:, :3]).as_matrix()                 # (n, 3, 3)
        X = np.einsum("nij,kj->nki", Rf, P) + F[:, None, 3:]            # (n, 8, 3)
        z = np.where(np.abs(X[..., 2:3]) < 1e-6, 1e-6, X[..., 2:3])
        return ((X[..., :2] / z - obs) * f_px).ravel()

    x0 = np.concatenate([pack(T_init)] + [pack(T) for T in poses])
    A = lil_matrix((n * 16, 6 + 6 * n), dtype=int)
    for i in range(n):
        A[i * 16:(i + 1) * 16, :6] = 1
        A[i * 16:(i + 1) * 16, 6 + 6 * i:12 + 6 * i] = 1
    # Robust soft-L1 loss (~2 px knee); explicit scales: 0.05 rad, 5 mm
    xs = np.tile([0.05, 0.05, 0.05, 0.005, 0.005, 0.005], n + 1)
    sol = least_squares(residuals, x0, jac_sparsity=A, loss="soft_l1", f_scale=2.0,
                        x_scale=xs, method="trf", max_nfev=300)

    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    T[:3, 3] = sol.x[3:6]
    r = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)
    print(f"    bundle adjustment over {n} frames: median corner error "
          f"{np.median(np.linalg.norm(residuals(x0).reshape(-1, 2), axis=1)):.2f} -> "
          f"{np.median(r):.2f} px; offset moved "
          f"{np.linalg.norm(T[:3, 3] - T_init[:3, 3]) * 1000:.1f} mm / "
          f"{rotation_angle_deg(T_init[:3, :3].T @ T[:3, :3]):.1f} deg")
    return T, float(np.median(r))


def calibrate_body(name, cfg, frame_dets, cam_calib):
    """Estimate T_ref_marker for every marker of a rigid body from frames in which
    it is seen together with the reference marker. Robust: medoid of all
    per-frame relative poses, then the mean of the inliers around it."""
    ref = cfg["ref"]
    T_ref_marker = {ref: np.eye(4)}
    stats = {}
    for mid in cfg["markers"]:
        if mid == ref:
            continue
        # All frames where both tags have a pose, with the worse of the two errors
        covis = []
        for det in frame_dets:
            p = det["poses"]
            if ref in p and mid in p:
                covis.append((max(p[ref][1], p[mid][1]),
                              np.linalg.inv(p[ref][0]) @ p[mid][0]))
        n_ref = sum(ref in d["poses"] for d in frame_dets)
        n_mid = sum(mid in d["poses"] for d in frame_dets)
        if len(covis) < CALIB_MIN_FRAMES_HARD:
            print(f"  WARNING {name}: markers {ref} and {mid} have poses together in only "
                  f"{len(covis)} frames ({ref}: {n_ref}, {mid}: {n_mid} of {len(frame_dets)}). "
                  f"Marker {mid} is ignored for this run; use a frame range where both "
                  "tags are visible to calibrate.")
            continue
        covis.sort(key=lambda s: s[0])
        errs = np.array([e for e, _ in covis])
        good = int((errs < CALIB_MAX_REPROJ_PX).sum())
        # Prefer low-error frames; if there are too few, take the best half
        # (at least CALIB_MIN_FRAMES) - the inlier step below rejects bad ones.
        n_use = max(good, min(len(covis), max(CALIB_MIN_FRAMES, len(covis) // 2)))
        print(f"  {name}: {ref}+{mid} co-visible in {len(covis)} frames, reproj error "
              f"median {np.median(errs):.2f} px, {good} below {CALIB_MAX_REPROJ_PX} px; "
              f"using the best {n_use}")
        S = np.array([T for _, T in covis[:n_use]])
        if len(S) > 1000:   # keep the O(n^2) medoid search cheap
            S = S[np.linspace(0, len(S) - 1, 1000).astype(int)]
        R, t = S[:, :3, :3], S[:, :3, 3]

        # Pairwise rotation angles and translation distances
        cos = np.clip((np.einsum("nij,mij->nm", R, R) - 1) / 2, -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        dist = np.linalg.norm(t[:, None] - t[None], axis=2)
        score = ang / CALIB_INLIER_DEG + dist / CALIB_INLIER_M
        medoid = int(np.argmin(np.median(score, axis=1)))

        inl = (ang[medoid] < CALIB_INLIER_DEG) & (dist[medoid] < CALIB_INLIER_M)
        T = np.eye(4)
        T[:3, :3] = project_to_so3(R[inl].mean(axis=0))   # chordal L2 rotation mean
        T[:3, 3] = t[inl].mean(axis=0)
        ba_err = None
        if CALIB_REFINE_BA:
            T, ba_err = refine_offset_ba(T, ref, mid, frame_dets, cam_calib)
        T_ref_marker[mid] = T

        rot_spread = np.mean([rotation_angle_deg(T[:3, :3].T @ Ri) for Ri in R[inl]])
        trans_spread = np.linalg.norm(t[inl] - T[:3, 3], axis=1).mean()
        stats[mid] = {"frames": int(len(S)), "inliers": int(inl.sum()),
                      "mean_rot_dev_deg": round(float(rot_spread), 3),
                      "mean_trans_dev_mm": round(float(trans_spread) * 1000, 3),
                      "centre_distance_mm": round(float(np.linalg.norm(T[:3, 3])) * 1000, 2),
                      "ba_median_corner_err_px": None if ba_err is None else round(ba_err, 3)}
        print(f"  {name}: T_{ref}_{mid} (robust init: {inl.sum()}/{len(S)} inlier frames), "
              f"spread {rot_spread:.2f} deg / {trans_spread * 1000:.2f} mm, "
              f"centre distance {np.linalg.norm(T[:3, 3]) * 1000:.1f} mm "
              f"(check this with a ruler)")
    return T_ref_marker, stats


def load_or_calibrate_bodies(frame_dets, cam_calib):
    stored = {}
    if BODY_CALIB_FILE.exists() and not RECALIBRATE_BODIES:
        stored = json.loads(BODY_CALIB_FILE.read_text())

    bodies, changed = {}, False
    for name, cfg in RIGID_BODIES.items():
        s = stored.get(name)
        if (s is not None and s["ref"] == cfg["ref"]
                and sorted(int(k) for k in s["T_ref_marker"]) == sorted(cfg["markers"])):
            T_ref_marker = {int(k): np.array(v) for k, v in s["T_ref_marker"].items()}
            print(f"  {name}: offsets loaded from {BODY_CALIB_FILE}")
        else:
            T_ref_marker, stats = calibrate_body(name, cfg, frame_dets, cam_calib)
            if len(T_ref_marker) == len(cfg["markers"]):   # only cache complete calibrations
                stored[name] = {"ref": cfg["ref"],
                                "T_ref_marker": {str(k): v.tolist()
                                                 for k, v in T_ref_marker.items()},
                                "stats": {str(k): v for k, v in stats.items()}}
                changed = True
        bodies[name] = {"ref": cfg["ref"], "T_ref_marker": T_ref_marker}
    if changed:
        BODY_CALIB_FILE.write_text(json.dumps(stored, indent=2))
        print(f"  offsets saved to {BODY_CALIB_FILE}")
    return bodies


def estimate_body_pose(body, det, cam_calib):
    """Fused pose of one rigid body in the camera frame.

    All corners of all visible body markers are expressed in the body frame and
    one PnP is solved over them. Starting guesses: both IPPE solutions of every
    visible marker (mapped to the body frame) plus a global SQPnP solution; each
    is refined with Levenberg-Marquardt and the one with the lowest reprojection
    error through the Aria model wins.

    Returns (T_cam_body, reproj_err_px, used_marker_ids, "fused" | "single") or None.
    """
    T_rm = body["T_ref_marker"]
    used = [m for m in T_rm if m in det["poses"]]
    if not used:
        return None
    if len(used) == 1:
        m = used[0]
        T_cam_m, err = det["poses"][m]
        return T_cam_m @ np.linalg.inv(T_rm[m]), err, used, "single"

    obj = np.concatenate([transform_points(T_rm[m], marker_object_points(TAG_SIZE_M[m]))
                          for m in used])
    px = np.concatenate([det["px"][m] for m in used])
    norm = unproject_normalized(cam_calib, px)

    guesses = [T_c @ np.linalg.inv(T_rm[m]) for m in used for T_c in det["cands"][m]]
    if norm is not None:
        try:
            ok, r, t = cv2.solvePnP(obj, norm, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
            if ok:
                guesses.append(rt_to_T(r, t))
        except cv2.error:
            pass

    best = None
    for G in guesses:
        T = G
        if norm is not None and G[2, 3] > 0:
            rvec = cv2.Rodrigues(G[:3, :3])[0]
            tvec = G[:3, 3].reshape(3, 1).copy()
            rvec, tvec = cv2.solvePnPRefineLM(obj, norm, np.eye(3), None, rvec, tvec)
            T = rt_to_T(rvec, tvec)
        err = reprojection_error(cam_calib, T, obj, px)
        if best is None or err < best[1]:
            best = (T, err)
    if best is None or not np.isfinite(best[1]):
        return None
    return best[0], best[1], used, "fused"


def body_ray_direction(body):
    """Ray direction in the body frame (unit vector)."""
    axis_dir = np.zeros(3)
    axis_dir[AXIS_INDEX[RAY_AXIS]] = RAY_AXIS_SIGN
    if RAY_DIRECTION == "axis":
        return axis_dir
    if RAY_DIRECTION == "marker_line":
        if RAY_LINE_TO_ID not in body["T_ref_marker"]:
            print(f"  WARNING: RAY_DIRECTION='marker_line' needs the offset of marker "
                  f"{RAY_LINE_TO_ID}, which isn't calibrated; casting along "
                  f"{'+' if RAY_AXIS_SIGN > 0 else '-'}{RAY_AXIS} of marker {body['ref']} instead.")
            return axis_dir
        d = body["T_ref_marker"][RAY_LINE_TO_ID][:3, 3].copy()
        print(f"  ray: from marker {body['ref']} through marker {RAY_LINE_TO_ID} "
              f"(body-frame direction {np.round(d / np.linalg.norm(d), 3)})")
        return d / np.linalg.norm(d)
    raise SystemExit(f"Unknown RAY_DIRECTION '{RAY_DIRECTION}'")


# ----------------------------------------------------------------------------- drawing
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


def pose_columns(T):
    return [*(f"{v:.6f}" for v in T[:3, 3]), *(f"{v:.6f}" for v in rot_to_quat_wxyz(T[:3, :3]))]


# ----------------------------------------------------------------------------- press classifier
class PressClassifier:
    """Plunger crop + pressed/not-pressed probability, reusing the training code
    (common.py for the crop, train_eval.py for the backbone features)."""

    def __init__(self, model_path, project_dir):
        sys.path.insert(0, str(Path(project_dir).resolve()))
        import joblib
        import common
        import config as press_config
        import train_eval
        self.common, self.press_cfg, self.te = common, press_config, train_eval

        model_path = (Path(model_path) if model_path is not None
                      else press_config.RESULTS_DIR / CLASSIFIER_RUN / "model.joblib")
        if not model_path.exists():
            raise SystemExit(f"Classifier not found: {model_path}. Run train_eval.py "
                             f"(--run-name {CLASSIFIER_RUN}), or set CLASSIFIER_PATH.")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.backbone_name = bundle["backbone"]
        self.te.INPUT_HW = tuple(bundle["input_hw"])   # preprocess() reads this global
        self.threshold = PRESS_THRESHOLD if PRESS_THRESHOLD is not None else bundle["threshold"]
        self.backbone, self.torch = self.te.load_backbone(self.backbone_name)
        if PRESS_CROP_SESSION is not None and PRESS_CROP_SESSION not in press_config.SESSIONS:
            raise SystemExit(f"PRESS_CROP_SESSION '{PRESS_CROP_SESSION}' is not in config.SESSIONS "
                             f"({list(press_config.SESSIONS)}); set it to None for the global box.")
        box = common.crop_box(PRESS_CROP_SESSION) if hasattr(common, "crop_box") else None
        if box:
            print(f"  crop box x {box[0]}, y {box[1]} m (up_sign {box[2]}) in marker "
                  f"{CROP_MARKER_ID}'s frame, from session {PRESS_CROP_SESSION}")
        print(f"Press classifier: {model_path} ({self.backbone_name}, "
              f"input {self.te.INPUT_HW}, threshold {self.threshold})")

    def crop(self, image, mapping, cam_calib, T_cam_marker=None, corners_frame_px=None):
        """Plunger crop as in extract_crops.py, on the sensor-orientation image.
        Uses T_cam_marker if given, otherwise the marker pose from its corners
        (common.estimate_pose)."""
        if mapping.rot:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        scale = image.shape[1] / int(cam_calib.get_image_size()[0])
        if T_cam_marker is None:
            c = mapping.to_calib(np.asarray(corners_frame_px).reshape(4, 2)) * scale
            pose = self.common.estimate_pose(c.reshape(1, 4, 2).astype(np.float32),
                                             self.press_cfg.TAG_SIZE_M, cam_calib, scale)
            if pose is None:
                return None
            T_cam_marker = pose["T"]
        if PRESS_CROP_SESSION is None:
            return self.common.rectified_crop(image, T_cam_marker, cam_calib, scale)
        return self.common.rectified_crop(image, T_cam_marker, cam_calib, scale,
                                          session=PRESS_CROP_SESSION)

    def predict(self, crops):
        """p(pressed) for a list of BGR crops."""
        if self.backbone is None:   # "pixels" baseline
            feats = np.stack([self.te.pixel_features(im) for im in crops])
        else:
            x = self.torch.from_numpy(np.stack([self.te.preprocess(im) for im in crops]))
            with self.torch.no_grad():
                feats = self.backbone(x).cpu().numpy()
        return self.model.predict_proba(feats)[:, 1]


def smooth_press(p, window):
    """Centred median over the frame sequence, ignoring frames that weren't
    classified (NaN); only classified frames get a value (others stay NaN)."""
    if window <= 1 or not np.isfinite(p).any():
        return p.copy()
    h = window // 2
    win = np.lib.stride_tricks.sliding_window_view(np.pad(p, h, constant_values=np.nan), window)
    with np.errstate(all="ignore"):
        out = np.nanmedian(win, axis=1)
    out[~np.isfinite(p)] = np.nan
    return out


def press_colour(label, default=None):
    return {"pressed": PRESSED_COLOUR, "not_pressed": NOT_PRESSED_COLOUR,
            "no_crop": UNKNOWN_COLOUR}.get(label, default)


def draw_press_status(image, label, p, cell, colour):
    """Banner in the top-left corner: PRESSED / not pressed, probability and cell."""
    text = {"pressed": "PRESSED", "not_pressed": "not pressed", "no_crop": "no crop"}[label]
    if p is not None and np.isfinite(p):
        text += f"  p={p:.2f}"
    if cell is not None:
        text += f"  cell (r{cell[0]}, c{cell[1]})"
    org = (30, 70)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, 3, cv2.LINE_AA)


def draw_inset(image, crop, colour, height=360):
    """Paste the classifier's input crop into the top-right corner with a coloured frame."""
    h, w = crop.shape[:2]
    ins = cv2.resize(crop, (max(1, int(w * height / h)), height), interpolation=cv2.INTER_AREA)
    ih, iw = ins.shape[:2]
    x0, y0 = image.shape[1] - iw - 20, 20
    if x0 < 0 or y0 + ih > image.shape[0]:
        return
    image[y0:y0 + ih, x0:x0 + iw] = ins
    cv2.rectangle(image, (x0 - 3, y0 - 3), (x0 + iw + 2, y0 + ih + 2), colour, 6)


# ----------------------------------------------------------------------------- main
def detect_frame(path, image, mapping, detector, cam_calib, wanted):
    """Detection + single-marker poses for one frame."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    det = {"path": path, "keep_c": [], "keep_i": [], "other_c": [], "other_i": [],
           "px": {}, "poses": {}, "cands": {}}
    if ids is None:
        return det
    for c, mid in zip(corners, ids.flatten()):
        mid = int(mid)
        if mid not in wanted:
            det["other_c"].append(c)
            det["other_i"].append(mid)
            continue
        det["keep_c"].append(c)
        det["keep_i"].append(mid)
        pts_calib = mapping.to_calib(c.reshape(4, 2))
        res = estimate_marker_pose(pts_calib, TAG_SIZE_M[mid], cam_calib)
        if res is None:
            continue
        T, err, cands = res
        if mid in det["poses"] and det["poses"][mid][1] <= err:
            continue   # same ID detected twice: keep the better one
        det["px"][mid], det["poses"][mid], det["cands"][mid] = pts_calib, (T, err), cands
    return det


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

    # ---- Pass 1: detect markers and single-marker poses in every frame
    frame_dets, mapping = [], None
    for idx, path in enumerate(tqdm(selected, desc="Detecting", unit="frame"), start=START_IDX):
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
        det = detect_frame(path, image, mapping, detector, cam_calib, wanted)
        det["idx"], det["mapping"] = idx, mapping
        frame_dets.append(det)

    # Detection summary: frames in which each marker was found / got a pose
    print(f"Detections in {len(frame_dets)} frames (found / with pose):")
    for mid in sorted(wanted):
        found = sum(mid in d["keep_i"] for d in frame_dets)
        posed = sum(mid in d["poses"] for d in frame_dets)
        errs = [d["poses"][mid][1] for d in frame_dets if mid in d["poses"]]
        med = f", median reproj {np.median(errs):.2f} px" if errs else ""
        print(f"  {mid:>4} ({MARKER_TO_OBJECT[mid]}): {found} / {posed}{med}")

    # ---- Rigid-body offsets (cached, or estimated from pass 1)
    print("Rigid bodies:")
    bodies = load_or_calibrate_bodies(frame_dets, cam_calib)
    ray_body = bodies.get(DISTANCE_BODY)
    ray_dir_body = body_ray_direction(ray_body) if ray_body is not None else None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated_dir = OUTPUT_DIR / "annotated"
    if SAVE_ANNOTATED:
        annotated_dir.mkdir(exist_ok=True)
    abstract_dir = OUTPUT_DIR / "abstract"
    if SAVE_ABSTRACT:
        abstract_dir.mkdir(exist_ok=True)

    n_det = 0
    T_global_cam = None   # inv(pose of marker 36), from the current or most recent frame
    anchor_counts = {"current": 0, "most_recent": 0}
    body_counts = {name: {"fused": 0, "single": 0, "none": 0} for name in bodies}
    # Plate corners cached in each plate marker's own frame, so the box can still
    # be drawn when only one of the two markers is visible.
    plate_in_marker = {}   # marker_id -> (8, 3) box corners in that marker's frame

    # ---- Pass 2: global frame, fused poses, plate box and ray for every frame
    results = []
    for det in tqdm(frame_dets, desc="Fusing", unit="frame"):
        idx, path = det["idx"], det["path"]
        T_cam = {mid: T for mid, (T, _) in det["poses"].items()}
        r = {"T_cam": T_cam}

        # 1) Centre the global frame on marker 36 in this frame, or the most recent one
        if ORIGIN_MARKER_ID in T_cam:
            T_global_cam, anchor = np.linalg.inv(T_cam[ORIGIN_MARKER_ID]), "current"
        elif T_global_cam is not None:
            anchor = "most_recent"
        else:
            raise SystemExit(
                f"Origin marker {ORIGIN_MARKER_ID} not found in first frame "
                f"{path.name} (index {idx}). Choose a different START_IDX."
            )
        anchor_counts[anchor] += 1
        r["anchor"], r["T_global_cam"] = anchor, T_global_cam

        # 2) Fused rigid-body poses
        body_poses = {}
        for name, body in bodies.items():
            res = estimate_body_pose(body, det, cam_calib)
            body_counts[name]["none" if res is None else res[3]] += 1
            if res is not None:
                body_poses[name] = res
        r["body_poses"] = body_poses

        # 3) Plate bounding box from markers 36 (top-right) and 26 (bottom-right)
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
        r["plate_corners"], r["plate_label"] = plate_corners, plate_label

        # 4) Cast a ray from the fused DISTANCE_BODY pose and intersect it with the plate box
        dist, p_origin, p_enter, hit_cell = None, None, None, None
        T_ray_body = body_poses[DISTANCE_BODY][0] if DISTANCE_BODY in body_poses else None
        if plate_corners is not None and T_ray_body is not None:
            p_origin = T_ray_body[:3, 3]
            ray_dir = T_ray_body[:3, :3] @ ray_dir_body
            ray_dir /= np.linalg.norm(ray_dir)
            hit = ray_box_intersection(p_origin, ray_dir, plate_corners)
            if hit is not None:
                dist = hit[0]
                p_enter = p_origin + dist * ray_dir
                if dist < HIT_DIST_THRESHOLD_M:
                    hit_cell = grid_cell_at(p_enter, plate_corners)
        r.update(T_ray_body=T_ray_body, dist=dist, p_origin=p_origin, p_enter=p_enter,
                 hit_cell=hit_cell, press_p_raw=np.nan, press_p=np.nan, press_label="",
                 crop=None)
        results.append(r)

    # ---- Pass 3: button-press classifier on frames where the ray hits the plate
    #      within HIT_DIST_THRESHOLD_M
    if CLASSIFY_PRESS:
        clf = PressClassifier(CLASSIFIER_PATH, PRESS_PROJECT_DIR)
        crops_dir = OUTPUT_DIR / "press_crops"
        if SAVE_PRESS_CROPS:
            crops_dir.mkdir(exist_ok=True)
        todo = [k for k, r in enumerate(results) if r["hit_cell"] is not None]
        batch_k, batch_crops = [], []

        def flush():
            if batch_crops:
                for k, p in zip(batch_k, clf.predict(batch_crops)):
                    results[k]["press_p_raw"] = float(p)
                batch_k.clear()
                batch_crops.clear()

        crop_body = None
        if CROP_POSE == "fused":
            crop_body = next((n for n, b in bodies.items() if b["ref"] == CROP_MARKER_ID), None)
            if crop_body is None:
                raise SystemExit(f"CROP_POSE='fused' needs a rigid body with ref {CROP_MARKER_ID}")
        elif CROP_POSE != "single":
            raise SystemExit(f"Unknown CROP_POSE '{CROP_POSE}'")

        for k in tqdm(todo, desc="Classifying", unit="frame"):
            det, r = frame_dets[k], results[k]
            crop = None
            if crop_body is not None:
                if crop_body in r["body_poses"]:
                    crop = clf.crop(cv2.imread(str(det["path"])), det["mapping"], cam_calib,
                                    T_cam_marker=r["body_poses"][crop_body][0])
            else:
                corners = [c for c, mid in zip(det["keep_c"], det["keep_i"]) if mid == CROP_MARKER_ID]
                if corners:
                    crop = clf.crop(cv2.imread(str(det["path"])), det["mapping"], cam_calib,
                                    corners_frame_px=corners[0])
            if crop is None:
                results[k]["press_label"] = "no_crop"
                continue
            if SAVE_PRESS_CROPS:
                cv2.imwrite(str(crops_dir / f"{det['path'].stem}.png"), crop)
            if DRAW_CROP_INSET:
                results[k]["crop"] = crop
            batch_k.append(k)
            batch_crops.append(crop)
            if len(batch_crops) >= PRESS_BATCH_SIZE:
                flush()
        flush()

        p_raw = np.array([r["press_p_raw"] for r in results])
        p_smooth = smooth_press(p_raw, PRESS_SMOOTH_WINDOW)
        for r, p in zip(results, p_smooth):
            if np.isfinite(p):
                r["press_p"] = float(p)
                r["press_label"] = "pressed" if p >= clf.threshold else "not_pressed"
        labels = [r["press_label"] for r in results if r["press_label"]]
        print(f"Press classifier: ray hit in {len(todo)} frames -> "
              + ", ".join(f"{lab}={labels.count(lab)}"
                          for lab in ("pressed", "not_pressed", "no_crop")))

    # ---- Pass 4: write rows and draw
    # marker_id is the tag ID for marker rows and e.g. "30+23" for body rows.
    # source: "marker" (single tag), "fused" (body from several tags), "single"
    # (body from one tag + calibrated offset). The ray / hit / press columns are
    # filled on the DISTANCE_BODY row only.
    header = ["frame_idx", "frame", "marker_id", "object", "source",
              "global_x", "global_y", "global_z",
              "global_qw", "global_qx", "global_qy", "global_qz",
              "anchor", "reproj_err_px",
              "cam_tx", "cam_ty", "cam_tz", "cam_qw", "cam_qx", "cam_qy", "cam_qz",
              "dev_tx", "dev_ty", "dev_tz", "dev_qw", "dev_qx", "dev_qy", "dev_qz",
              "ray_dist_to_plate_m",   # blank if the ray misses
              "hit_row", "hit_col",    # grid cell hit within HIT_DIST_THRESHOLD_M
              "press_p_raw", "press_p_smoothed", "press_label"]   # classifier output
    with open(OUTPUT_DIR / "poses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for det, r in tqdm(list(zip(frame_dets, results)), desc="Writing", unit="frame"):
            idx, path, mapping = det["idx"], det["path"], det["mapping"]
            T_cam, body_poses = r["T_cam"], r["body_poses"]
            plate_corners, plate_label = r["plate_corners"], r["plate_label"]
            dist, p_origin, p_enter, hit_cell = r["dist"], r["p_origin"], r["p_enter"], r["hit_cell"]
            T_ray_body, label = r["T_ray_body"], r["press_label"]

            def write_row(marker_id, obj, source, T_cam_x, err, extra=("",) * 6):
                writer.writerow([
                    idx, path.name, marker_id, obj, source,
                    *pose_columns(r["T_global_cam"] @ T_cam_x),
                    r["anchor"],
                    f"{err:.3f}",
                    *pose_columns(T_cam_x),
                    *pose_columns(T_device_camera @ T_cam_x),
                    *extra,
                ])

            for mid, (T_cam_marker, err) in det["poses"].items():
                n_det += 1
                write_row(mid, MARKER_TO_OBJECT[mid], "marker", T_cam_marker, err)
            for name, (T_cam_body, err, used, source) in body_poses.items():
                extra = ("",) * 6
                if name == DISTANCE_BODY:
                    fmt = lambda v: f"{v:.4f}" if np.isfinite(v) else ""
                    extra = (f"{dist:.6f}" if dist is not None else "",
                             *(hit_cell if hit_cell is not None else ("", "")),
                             fmt(r["press_p_raw"]), fmt(r["press_p"]), label)
                write_row("+".join(map(str, used)), name, source, T_cam_body, err, extra)

            if not (SAVE_ANNOTATED or SAVE_ABSTRACT):
                continue
            image = cv2.imread(str(path))
            cell_colour = press_colour(label, HIT_CELL_COLOUR)

            # 6) Annotated frame: marker axes, fused body axes, grid, hit cell, plate box, ray, press
            if SAVE_ANNOTATED:
                annotated = image.copy()
                for mid, T_cam_marker in T_cam.items():
                    draw_axes(annotated, T_cam_marker, TAG_SIZE_M[mid] * 0.75, cam_calib, mapping, 1)
                for name, (T_cam_body, _, used, source) in body_poses.items():
                    draw_axes(annotated, T_cam_body, 0.04, cam_calib, mapping, 4)
                if DRAW_WELL_GRID and plate_corners is not None:
                    draw_well_grid(annotated, plate_corners, cam_calib, mapping)
                if hit_cell is not None:
                    fill_grid_cell(annotated, plate_corners, *hit_cell, cam_calib, mapping,
                                   colour=cell_colour)
                if DRAW_PLATE_BOX and plate_corners is not None:
                    draw_plate_box(annotated, plate_corners, cam_calib, mapping, plate_label)
                if dist is not None:
                    draw_ray(annotated, p_origin, p_enter, dist, cam_calib, mapping)
                if det["keep_c"]:
                    cv2.aruco.drawDetectedMarkers(annotated, det["keep_c"],
                                                  np.array(det["keep_i"]).reshape(-1, 1))
                if det["other_c"]:
                    cv2.aruco.drawDetectedMarkers(annotated, det["other_c"],
                                                  np.array(det["other_i"]).reshape(-1, 1),
                                                  borderColor=OTHER_MARKER_COLOUR)
                if label:
                    draw_press_status(annotated, label, r["press_p"], hit_cell, cell_colour)
                if r["crop"] is not None:
                    draw_inset(annotated, r["crop"], cell_colour)
                cv2.imwrite(str(annotated_dir / f"{path.stem}.jpg"), annotated)

            # 7) Abstract view: white on black, same dimensions as the frame
            if SAVE_ABSTRACT:
                abstract = np.zeros_like(image)
                W = ABSTRACT_COLOUR
                if plate_corners is not None:
                    draw_well_grid(abstract, plate_corners, cam_calib, mapping,
                                   colour=W, alpha=ABSTRACT_GRID_ALPHA)
                    if hit_cell is not None:
                        fill_grid_cell(abstract, plate_corners, *hit_cell, cam_calib, mapping,
                                       colour=press_colour(label, ABSTRACT_CELL_COLOUR),
                                       alpha=ABSTRACT_CELL_ALPHA)
                    draw_plate_box(abstract, plate_corners, cam_calib, mapping, plate_label,
                                   colour=W)
                if dist is not None:
                    draw_ray(abstract, p_origin, p_enter, dist, cam_calib, mapping, colour=W)
                if T_ray_body is not None:
                    used = body_poses[DISTANCE_BODY][2]
                    draw_marker_position(abstract, T_ray_body[:3, 3],
                                         f"{DISTANCE_BODY} ({'+'.join(map(str, used))})",
                                         cam_calib, mapping, W)
                if label:
                    draw_press_status(abstract, label, r["press_p"], hit_cell,
                                      press_colour(label, W))
                cv2.imwrite(str(abstract_dir / f"{path.stem}.jpg"), abstract)

    print(f"Done: {n_det} marker poses written to {OUTPUT_DIR / 'poses.csv'}")
    print("Frames per anchor type: " + ", ".join(f"{k}={v}" for k, v in anchor_counts.items()))
    for name, c in body_counts.items():
        print(f"Body '{name}': " + ", ".join(f"{k}={v}" for k, v in c.items()))


if __name__ == "__main__":
    main()