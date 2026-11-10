"""
Multi-marker rigid bodies (pipette = markers 30 + 23).

Calibration: the transform T_ref_m from the reference tag to every other tag of a
body is constant. It is loaded from BODY_CALIB_FILE, or estimated online from the
first frames in which all of the body's tags are seen (robust medoid + inlier mean,
optionally refined by bundle adjustment) and then cached in that file.

Fusion (every frame): all corners of all visible body tags are expressed in the
body frame (= the reference tag's frame) and one PnP is solved over them. With
only one tag visible, the pose comes from that tag plus its calibrated offset.
"""
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import settings as S
from geometry import (marker_object_points, project_to_so3, reprojection_error,
                      rotation_angle_deg, rt_to_T, transform_points, unproject_normalized)

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ----------------------------------------------------------------------------- fused pose
def estimate_body_pose(body, det, cam_calib):
    """Fused pose of one rigid body in the camera frame.
    Returns (T_cam_body, reproj_err_px, used_marker_ids, "fused" | "single") or None."""
    T_rm = body["T_ref_marker"]
    used = [m for m in T_rm if m in det.poses]
    if not used:
        return None
    if len(used) == 1:
        m = used[0]
        T_cam_m, err = det.poses[m]
        return T_cam_m @ np.linalg.inv(T_rm[m]), err, used, "single"

    obj = np.concatenate([transform_points(T_rm[m], marker_object_points(S.TAG_SIZE_M[m]))
                          for m in used])
    px = np.concatenate([det.px[m] for m in used])
    norm = unproject_normalized(cam_calib, px)

    # Starting guesses: both IPPE solutions of every tag + a global SQPnP solution;
    # each is refined with Levenberg-Marquardt, lowest reprojection error wins.
    guesses = [T_c @ np.linalg.inv(T_rm[m]) for m in used for T_c in det.cands[m]]
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


# ----------------------------------------------------------------------------- calibration
def refine_offset_ba(T_init, ref, mid, frame_dets, cam_calib):
    """Bundle adjustment of T_ref_mid jointly with one body pose per frame, so that
    all 8 corners of both tags reproject onto their detections (robust loss)."""
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
        if ref not in det.poses or mid not in det.poses:
            continue
        nr = unproject_normalized(cam_calib, det.px[ref])
        nm = unproject_normalized(cam_calib, det.px[mid])
        res = estimate_body_pose(body, det, cam_calib)
        if nr is None or nm is None or res is None:
            continue
        obs.append(np.concatenate([nr, nm]))
        poses.append(res[0])
        errs.append(res[1])
    if len(obs) < S.CALIB_MIN_FRAMES_HARD:
        return T_init, None
    keep = np.array(errs) < 3 * np.median(errs) + 1.0   # drop gross failures of the init
    obs = np.array(obs)[keep]
    poses = [p for p, k in zip(poses, keep) if k]
    n = len(poses)

    # Approximate focal length, so residuals (normalised coords) read as pixels
    f_px = np.linalg.norm(
        np.asarray(cam_calib.project_no_checks(np.array([0.01, 0.0, 1.0]))).ravel()
        - np.asarray(cam_calib.project_no_checks(np.array([0.0, 0.0, 1.0]))).ravel()) / 0.01
    obj_ref = marker_object_points(S.TAG_SIZE_M[ref])
    obj_mid = marker_object_points(S.TAG_SIZE_M[mid])

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
    """Estimate T_ref_marker for every marker of a rigid body from frames in which it
    is seen together with the reference marker. Returns (T_ref_marker, stats)."""
    ref = cfg["ref"]
    T_ref_marker = {ref: np.eye(4)}
    stats = {}
    for mid in cfg["markers"]:
        if mid == ref:
            continue
        covis = [(max(d.poses[ref][1], d.poses[mid][1]),
                  np.linalg.inv(d.poses[ref][0]) @ d.poses[mid][0])
                 for d in frame_dets if ref in d.poses and mid in d.poses]
        if len(covis) < S.CALIB_MIN_FRAMES_HARD:
            print(f"  WARNING {name}: markers {ref} and {mid} seen together in only "
                  f"{len(covis)} frames; marker {mid} can't be calibrated yet.")
            continue
        covis.sort(key=lambda s: s[0])
        errs = np.array([e for e, _ in covis])
        good = int((errs < S.CALIB_MAX_REPROJ_PX).sum())
        n_use = max(good, min(len(covis), max(S.CALIB_MIN_FRAMES, len(covis) // 2)))
        S_ = np.array([T for _, T in covis[:n_use]])
        if len(S_) > 1000:   # keep the O(n^2) medoid search cheap
            S_ = S_[np.linspace(0, len(S_) - 1, 1000).astype(int)]
        R, t = S_[:, :3, :3], S_[:, :3, 3]

        cos = np.clip((np.einsum("nij,mij->nm", R, R) - 1) / 2, -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        dist = np.linalg.norm(t[:, None] - t[None], axis=2)
        score = ang / S.CALIB_INLIER_DEG + dist / S.CALIB_INLIER_M
        medoid = int(np.argmin(np.median(score, axis=1)))

        inl = (ang[medoid] < S.CALIB_INLIER_DEG) & (dist[medoid] < S.CALIB_INLIER_M)
        T = np.eye(4)
        T[:3, :3] = project_to_so3(R[inl].mean(axis=0))   # chordal L2 rotation mean
        T[:3, 3] = t[inl].mean(axis=0)
        ba_err = None
        if S.CALIB_REFINE_BA:
            T, ba_err = refine_offset_ba(T, ref, mid, frame_dets, cam_calib)
        T_ref_marker[mid] = T

        rot_spread = np.mean([rotation_angle_deg(T[:3, :3].T @ Ri) for Ri in R[inl]])
        trans_spread = np.linalg.norm(t[inl] - T[:3, 3], axis=1).mean()
        stats[mid] = {"frames": int(len(S_)), "inliers": int(inl.sum()),
                      "mean_rot_dev_deg": round(float(rot_spread), 3),
                      "mean_trans_dev_mm": round(float(trans_spread) * 1000, 3),
                      "centre_distance_mm": round(float(np.linalg.norm(T[:3, 3])) * 1000, 2),
                      "ba_median_corner_err_px": None if ba_err is None else round(ba_err, 3)}
        print(f"  {name}: T_{ref}_{mid} from {inl.sum()}/{len(S_)} inlier frames, spread "
              f"{rot_spread:.2f} deg / {trans_spread * 1000:.2f} mm, centre distance "
              f"{np.linalg.norm(T[:3, 3]) * 1000:.1f} mm (check this with a ruler)")
    return T_ref_marker, stats


# ----------------------------------------------------------------------------- registry
class RigidBodies:
    """Holds the calibrated rigid bodies, calibrates missing ones online and gives
    fused poses and ray directions."""

    def __init__(self, cam_calib, configs=None, calib_file=None, recalibrate=None):
        self.cam_calib = cam_calib
        self.configs = configs if configs is not None else S.RIGID_BODIES
        self.calib_file = Path(calib_file if calib_file is not None else S.BODY_CALIB_FILE)
        recalibrate = S.RECALIBRATE_BODIES if recalibrate is None else recalibrate
        self._stored = {}
        if self.calib_file.exists() and not recalibrate:
            self._stored = json.loads(self.calib_file.read_text())

        self.bodies, self.pending = {}, set()
        for name, cfg in self.configs.items():
            s = self._stored.get(name)
            if (s is not None and s["ref"] == cfg["ref"]
                    and sorted(int(k) for k in s["T_ref_marker"]) == sorted(cfg["markers"])):
                T_ref_marker = {int(k): np.array(v) for k, v in s["T_ref_marker"].items()}
                print(f"Rigid body '{name}': offsets loaded from {self.calib_file}")
            else:
                T_ref_marker = {cfg["ref"]: np.eye(4)}
                self.pending.add(name)
                print(f"Rigid body '{name}': not calibrated; will calibrate after "
                      f"{S.CALIB_WARMUP_FRAMES} frames with all of {cfg['markers']} visible "
                      f"(until then: marker {cfg['ref']} only)")
            self.bodies[name] = {"ref": cfg["ref"], "T_ref_marker": T_ref_marker}
        self._buffer = deque(maxlen=S.CALIB_MAX_BUFFER)
        self._ray_dirs = {}

    # -- calibration
    def observe(self, det):
        """Feed one frame's detections; calibrates pending bodies once enough
        co-visible frames have been collected."""
        if not self.pending:
            return
        if not any(sum(m in det.poses for m in self.configs[n]["markers"]) >= 2
                   for n in self.pending):
            return
        self._buffer.append(det)
        for name in list(self.pending):
            cfg = self.configs[name]
            n_covis = min(sum(cfg["ref"] in d.poses and m in d.poses for d in self._buffer)
                          for m in cfg["markers"] if m != cfg["ref"])
            if n_covis >= S.CALIB_WARMUP_FRAMES:
                self._calibrate(name)
        if not self.pending:
            self._buffer.clear()

    def finalise(self):
        """End of run: calibrate still-pending bodies from whatever was collected,
        so the next run starts calibrated."""
        for name in list(self.pending):
            self._calibrate(name)

    def _calibrate(self, name):
        cfg = self.configs[name]
        print(f"Calibrating rigid body '{name}' from {len(self._buffer)} frames")
        T_ref_marker, stats = calibrate_body(name, cfg, list(self._buffer), self.cam_calib)
        if len(T_ref_marker) != len(cfg["markers"]):
            return
        self.bodies[name]["T_ref_marker"] = T_ref_marker
        self.pending.discard(name)
        self._ray_dirs.pop(name, None)
        self._stored[name] = {"ref": cfg["ref"],
                              "T_ref_marker": {str(k): v.tolist() for k, v in T_ref_marker.items()},
                              "stats": {str(k): v for k, v in stats.items()}}
        self.calib_file.write_text(json.dumps(self._stored, indent=2))
        print(f"  offsets saved to {self.calib_file}")

    # -- per frame
    def pose(self, name, det):
        return estimate_body_pose(self.bodies[name], det, self.cam_calib)

    def ray_direction(self, name):
        """Unit ray direction in the body frame (cached until the body is recalibrated)."""
        if name not in self._ray_dirs:
            self._ray_dirs[name] = self._compute_ray_direction(self.bodies[name])
        return self._ray_dirs[name]

    @staticmethod
    def _compute_ray_direction(body):
        axis_dir = np.zeros(3)
        axis_dir[AXIS_INDEX[S.RAY_AXIS]] = S.RAY_AXIS_SIGN
        if S.RAY_DIRECTION == "axis":
            return axis_dir
        if S.RAY_DIRECTION != "marker_line":
            raise ValueError(f"Unknown RAY_DIRECTION '{S.RAY_DIRECTION}'")
        if S.RAY_LINE_TO_ID not in body["T_ref_marker"]:
            print(f"  ray: marker {S.RAY_LINE_TO_ID} not calibrated yet; casting along "
                  f"{'+' if S.RAY_AXIS_SIGN > 0 else '-'}{S.RAY_AXIS} of marker {body['ref']}")
            return axis_dir
        d = body["T_ref_marker"][S.RAY_LINE_TO_ID][:3, 3].copy()
        d /= np.linalg.norm(d)
        print(f"  ray: from marker {body['ref']} through marker {S.RAY_LINE_TO_ID} "
              f"(body-frame direction {np.round(d, 3)})")
        return d
