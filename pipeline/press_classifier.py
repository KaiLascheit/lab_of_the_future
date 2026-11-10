"""
Pipette button press: the trained classifier (crop of the plunger -> p(pressed))
and an online event detector that turns per-frame probabilities into one
"dispense" per press.

The classifier reuses your training code in PRESS_PROJECT_DIR (common.py for the
crop, config.py, train_eval.py for the backbone), exactly as detect_apriltags.py did.
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np

import settings as S


class PressClassifier:
    def __init__(self, model_path=S.CLASSIFIER_PATH, project_dir=S.PRESS_PROJECT_DIR):
        project_dir = Path(project_dir).resolve()
        if not project_dir.exists():
            raise FileNotFoundError(f"PRESS_PROJECT_DIR not found: {project_dir}")
        sys.path.insert(0, str(project_dir))
        import joblib
        import common
        import config as press_config
        import train_eval
        self.common, self.press_cfg, self.te = common, press_config, train_eval

        model_path = (Path(model_path) if model_path is not None
                      else Path(press_config.RESULTS_DIR) / S.CLASSIFIER_RUN / "model.joblib")
        if not model_path.exists():
            raise FileNotFoundError(f"Classifier not found: {model_path}. Run train_eval.py "
                                    f"(--run-name {S.CLASSIFIER_RUN}), or set CLASSIFIER_PATH.")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.backbone_name = bundle["backbone"]
        self.te.INPUT_HW = tuple(bundle["input_hw"])   # preprocess() reads this global
        self.threshold = S.PRESS_THRESHOLD if S.PRESS_THRESHOLD is not None else bundle["threshold"]
        self.backbone, self.torch = self.te.load_backbone(self.backbone_name)
        if S.PRESS_CROP_SESSION is not None and S.PRESS_CROP_SESSION not in press_config.SESSIONS:
            raise ValueError(f"PRESS_CROP_SESSION '{S.PRESS_CROP_SESSION}' is not in "
                             f"config.SESSIONS ({list(press_config.SESSIONS)})")

        # Body whose pose places the crop when CROP_POSE == "fused"
        self.crop_body = None
        if S.CROP_POSE == "fused":
            self.crop_body = next((n for n, b in S.RIGID_BODIES.items()
                                   if b["ref"] == S.CROP_MARKER_ID), None)
            if self.crop_body is None:
                raise ValueError(f"CROP_POSE='fused' needs a rigid body with ref {S.CROP_MARKER_ID}")
        elif S.CROP_POSE != "single":
            raise ValueError(f"Unknown CROP_POSE '{S.CROP_POSE}'")
        print(f"Press classifier: {model_path} ({self.backbone_name}, "
              f"input {self.te.INPUT_HW}, threshold {self.threshold})")

    def crop_for(self, image, tracking, cam_calib):
        """Plunger crop for one frame (as in extract_crops.py), or None."""
        scale = image.shape[1] / int(cam_calib.get_image_size()[0])
        if self.crop_body is not None:
            body = tracking.body_poses.get(self.crop_body)
            if body is None:
                return None
            T_cam_marker = body[0]
        else:
            corners = tracking.detections.corners_of(S.CROP_MARKER_ID)
            if corners is None:
                return None
            c = tracking.mapping.to_calib(corners) * scale
            pose = self.common.estimate_pose(c.reshape(1, 4, 2).astype(np.float32),
                                             self.press_cfg.TAG_SIZE_M, cam_calib, scale)
            if pose is None:
                return None
            T_cam_marker = pose["T"]
        if S.PRESS_CROP_SESSION is None:
            return self.common.rectified_crop(image, T_cam_marker, cam_calib, scale)
        return self.common.rectified_crop(image, T_cam_marker, cam_calib, scale,
                                          session=S.PRESS_CROP_SESSION)

    def predict(self, crops):
        """p(pressed) for a list of BGR crops."""
        if self.backbone is None:   # "pixels" baseline
            feats = np.stack([self.te.pixel_features(im) for im in crops])
        else:
            x = self.torch.from_numpy(np.stack([self.te.preprocess(im) for im in crops]))
            with self.torch.no_grad():
                feats = self.backbone(x).cpu().numpy()
        return self.model.predict_proba(feats)[:, 1]


class PressEventDetector:
    """Online press detection.

    update() takes p(pressed) for the current frame, or None when the frame wasn't
    classified (the ray isn't on the plate, or no crop). The smoothed probability
    is the median of the last `window` classified values; a press event fires on
    the first smoothed "pressed" frame, and the detector re-arms only after
    `rearm_frames` frames that are not pressed (unclassified frames count as not
    pressed), so holding the plunger down or losing the pose for a single frame
    doesn't count twice.
    """

    def __init__(self, threshold, window=S.PRESS_SMOOTH_WINDOW, rearm_frames=S.PRESS_REARM_FRAMES):
        self.threshold = threshold
        self.history = deque(maxlen=max(1, window))
        self.rearm_frames = rearm_frames
        self.armed = True
        self.released_for = 0

    def update(self, p_raw):
        """Returns (p_smoothed or None, label, press_started)."""
        self.history.append(np.nan if p_raw is None else float(p_raw))
        p_smooth, label = None, ""
        if p_raw is not None:
            p_smooth = float(np.nanmedian(np.array(self.history)))
            label = "pressed" if p_smooth >= self.threshold else "not_pressed"

        started = False
        if label == "pressed":
            self.released_for = 0
            if self.armed:
                started, self.armed = True, False
        else:
            self.released_for += 1
            if self.released_for >= self.rearm_frames:
                self.armed = True
        return p_smooth, label, started
