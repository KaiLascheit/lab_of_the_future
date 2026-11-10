"""AprilTag detection and single-marker 6-DoF poses for one frame."""
from dataclasses import dataclass, field

import cv2
import numpy as np

import settings as S
from geometry import marker_object_points, reprojection_error, rt_to_T, unproject_normalized


@dataclass
class MarkerDetections:
    """Tags found in one frame. Pose dicts only hold tags listed in TAG_SIZE_M."""
    tags: list = field(default_factory=list)     # (corners in frame px (4, 2), id) of TAG_SIZE_M tags
    others: list = field(default_factory=list)   # same for any other tag ids
    px: dict = field(default_factory=dict)       # id -> (4, 2) corners in calibration px
    poses: dict = field(default_factory=dict)    # id -> (T_cam_marker, reprojection error px)
    cands: dict = field(default_factory=dict)    # id -> both IPPE pose candidates

    def corners_of(self, marker_id):
        return next((c for c, mid in self.tags if mid == marker_id), None)


def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # better corners -> better poses
    return cv2.aruco.ArucoDetector(dictionary, params)


def estimate_marker_pose(corners_calib_px, size, cam_calib):
    """Pose of one square marker from its 4 corners (calibration pixels).

    Corners are unprojected with the Aria model, normalised to z=1, and PnP is
    solved with an identity intrinsic matrix. Both IPPE solutions of the planar
    "flip" ambiguity are returned so the rigid-body fusion can pick the one that
    agrees with the other tags. Returns (T_best, err_best, [T_candidates]) or None.
    """
    norm = unproject_normalized(cam_calib, corners_calib_px)
    if norm is None:
        return None
    obj = marker_object_points(size)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, norm, np.eye(3), None,
                                             flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not n:
        return None
    cands = [rt_to_T(r, t) for r, t in zip(rvecs, tvecs)]
    errs = [reprojection_error(cam_calib, T, obj, corners_calib_px) for T in cands]
    best = int(np.argmin(errs))
    if not np.isfinite(errs[best]):
        return None
    return cands[best], errs[best], cands


def detect_markers(image, mapping, detector, cam_calib) -> MarkerDetections:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    det = MarkerDetections()
    if ids is None:
        return det
    for c, mid in zip(corners, ids.flatten()):
        mid, c = int(mid), c.reshape(4, 2)
        if mid not in S.TAG_SIZE_M:
            det.others.append((c, mid))
            continue
        det.tags.append((c, mid))
        pts_calib = mapping.to_calib(c)
        res = estimate_marker_pose(pts_calib, S.TAG_SIZE_M[mid], cam_calib)
        if res is None:
            continue
        T, err, cands = res
        if mid in det.poses and det.poses[mid][1] <= err:
            continue   # same id detected twice: keep the better one
        det.px[mid], det.poses[mid], det.cands[mid] = pts_calib, (T, err), cands
    return det
