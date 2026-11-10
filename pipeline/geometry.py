"""Camera-model and pose helpers shared by the AprilTag modules.

The Aria RGB camera uses a fisheye model that OpenCV doesn't support, so poses
are solved on normalised (z = 1) coordinates obtained by unprojecting pixels
through the Aria calibration, and 3D points are drawn by projecting through it.
"""
import cv2
import numpy as np


class FrameToCalibMapping:
    """Maps pixels between the processed frame and the calibration's image size.
    Frames come straight from the VRS in sensor orientation, so this is only a
    (normally 1:1) scale; it matters if frames are ever resized before tracking."""

    def __init__(self, frame_wh, calib_wh):
        self.frame_wh = tuple(frame_wh)
        self.sx = calib_wh[0] / frame_wh[0]
        self.sy = calib_wh[1] / frame_wh[1]
        if abs(self.sx - self.sy) > 1e-3:
            print(f"WARNING: non-uniform scale frame->calib ({self.sx:.4f}, {self.sy:.4f}); "
                  "were the frames cropped?")

    def to_calib(self, pts):
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2) * [self.sx, self.sy]

    def to_frame(self, pts):
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2) / [self.sx, self.sy]


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
        ray = np.asarray(cam_calib.unproject_no_checks(np.asarray(p, dtype=np.float64)),
                         dtype=np.float64).reshape(3)
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
