"""
Per-frame AprilTag tracking: marker poses, fused pipette pose, plate box, and the
ray from the pipette onto the plate (which grid cell it points at, and how far).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import settings as S
from geometry import FrameToCalibMapping, transform_points
from markers import MarkerDetections, detect_markers, make_detector
from plate import grid_cell_at, plate_corners_cam, ray_box_intersection
from rigid_body import RigidBodies


@dataclass
class TrackingResult:
    detections: MarkerDetections
    mapping: FrameToCalibMapping
    body_poses: dict = field(default_factory=dict)   # name -> (T_cam_body, err, used ids, source)
    plate_corners: Optional[np.ndarray] = None       # (8, 3) camera frame, see plate_corners_cam
    plate_source: Optional[str] = None               # "both markers" | "marker <id>"
    ray_origin: Optional[np.ndarray] = None          # camera frame, metres
    ray_dir: Optional[np.ndarray] = None
    hit_dist: Optional[float] = None                 # along the ray to where it enters the box
    hit_point: Optional[np.ndarray] = None
    hit_cell: Optional[tuple] = None                 # (row, col), only if hit_dist < threshold

    @property
    def pipette_pose(self):
        res = self.body_poses.get(S.RAY_BODY)
        return None if res is None else res[0]

    def summary(self):
        """Small JSON-friendly summary for the frame log."""
        body = self.body_poses.get(S.RAY_BODY)
        return {
            "markers": sorted(self.detections.poses),
            "pipette_pose": None if body is None else {
                "source": body[3], "markers": body[2], "reproj_err_px": round(body[1], 3)},
            "plate": self.plate_source,
            "ray_dist_m": None if self.hit_dist is None else round(float(self.hit_dist), 4),
            "hit_cell": None if self.hit_cell is None else list(self.hit_cell),
        }


class AprilTagTracker:
    def __init__(self, cam_calib):
        self.cam_calib = cam_calib
        self.calib_wh = tuple(int(v) for v in cam_calib.get_image_size())
        self.detector = make_detector()
        self.bodies = RigidBodies(cam_calib)
        self._mapping = None
        # Plate corners in each plate marker's own frame, so the box can still be
        # placed when only one of the two markers is visible.
        self._plate_in_marker = {}

    def process(self, image) -> TrackingResult:
        mapping = self._mapping_for(image)
        det = detect_markers(image, mapping, self.detector, self.cam_calib)
        self.bodies.observe(det)
        result = TrackingResult(det, mapping)
        for name in self.bodies.bodies:
            res = self.bodies.pose(name, det)
            if res is not None:
                result.body_poses[name] = res
        self._locate_plate(result)
        self._cast_ray(result)
        return result

    def finish(self):
        self.bodies.finalise()

    # ------------------------------------------------------------------------- internals
    def _mapping_for(self, image):
        wh = (image.shape[1], image.shape[0])
        if self._mapping is None or self._mapping.frame_wh != wh:
            self._mapping = FrameToCalibMapping(wh, self.calib_wh)
        return self._mapping

    def _locate_plate(self, r: TrackingResult):
        T_cam = {mid: T for mid, (T, _) in r.detections.poses.items()}
        top, bottom = S.PLATE_TOP_ID, S.PLATE_BOTTOM_ID
        if top in T_cam and bottom in T_cam:
            r.plate_corners = plate_corners_cam(T_cam[top], T_cam[bottom])
            r.plate_source = "both markers"
            for mid in (top, bottom):
                self._plate_in_marker[mid] = transform_points(np.linalg.inv(T_cam[mid]),
                                                              r.plate_corners)
            return
        for mid in (top, bottom):
            if mid in T_cam and mid in self._plate_in_marker:
                r.plate_corners = transform_points(T_cam[mid], self._plate_in_marker[mid])
                r.plate_source = f"marker {mid}"
                return

    def _cast_ray(self, r: TrackingResult):
        T_body = r.pipette_pose
        if r.plate_corners is None or T_body is None:
            return
        r.ray_origin = T_body[:3, 3]
        d = T_body[:3, :3] @ self.bodies.ray_direction(S.RAY_BODY)
        r.ray_dir = d / np.linalg.norm(d)
        hit = ray_box_intersection(r.ray_origin, r.ray_dir, r.plate_corners)
        if hit is None:
            return
        r.hit_dist = float(hit[0])
        r.hit_point = r.ray_origin + r.hit_dist * r.ray_dir
        if r.hit_dist < S.HIT_DIST_THRESHOLD_M:
            r.hit_cell = grid_cell_at(r.hit_point, r.plate_corners)
