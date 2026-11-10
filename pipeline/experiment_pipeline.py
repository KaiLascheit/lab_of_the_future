"""
Per-frame orchestration. For every 5 Hz frame:

  1. object detection (YOLO-World + MobileSAM)
  2. if a pipette is detected (gate currently disabled, see REQUIRE_PIPETTE_FOR_TRACKING):
       AprilTag tracking -> pipette ray onto the plate -> hit cell
       volume counter reading
  3. press classifier on frames where the ray hits the plate; a new press
     records a dispense into the hit well, with the current stable volume
  4. every VLM_EVERY_N_FRAMES-th frame: VLM update of the running report,
     and the object-detection visualisation
  5. tracking + abstract visualisations of every frame

Everything per frame is appended to frames.jsonl for debugging.
"""
import json
from pathlib import Path

from tqdm import tqdm

import settings as S
from apriltag_tracker import AprilTagTracker
from experiment_state import WellPlateState, format_time
from object_detector import ObjectDetector, objects_to_dicts
from plate import well_name
from press_classifier import PressClassifier, PressEventDetector
from report import ExperimentReport
from visualisation import Visualiser
from volume_reader import VolumeTracker


class ExperimentPipeline:
    def __init__(self, cam_calib, output_dir, recording_name, use_vlm=S.RUN_VLM):
        self.cam_calib = cam_calib
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.detector = ObjectDetector() if S.RUN_OBJECT_DETECTION else None
        self.tracker = AprilTagTracker(cam_calib)
        self.volume = VolumeTracker()
        self.press = PressClassifier() if S.CLASSIFY_PRESS else None
        self.press_events = PressEventDetector(self.press.threshold if self.press else 0.5)
        self.plate = WellPlateState()
        self.report = ExperimentReport(recording_name)
        self.vlm = None
        if use_vlm:
            from vlm_reporter import VLMReporter   # only needs google-genai when used
            self.vlm = VLMReporter()
        self.vis = Visualiser(self.out, cam_calib)
        self.frame_log = open(self.out / "frames.jsonl", "w")

    # ------------------------------------------------------------------------- per frame
    def process(self, frame):
        # 1) Objects
        objects = self.detector.detect(frame.image) if self.detector else []

        # 2) Tracking + volume, gated on a detected pipette
        pipette_seen = any(o.label == S.PIPETTE_LABEL for o in objects)
        tracking, reading = None, None
        if pipette_seen or not S.REQUIRE_PIPETTE_FOR_TRACKING:
            tracking = self.tracker.process(frame.image)
            reading = self.volume.update(frame.image)

        # 3) Press -> dispense
        p_raw, crop = self._classify_press(frame.image, tracking)
        p_smooth, press_label, press_started = self.press_events.update(p_raw)
        event = None
        if press_started and tracking is not None and tracking.hit_cell is not None:
            stable = self.volume.current
            event = self.plate.record_dispense(
                frame.time_s, frame.idx, *tracking.hit_cell,
                volume_ul=None if stable is None else stable.volume_ul, press_p=p_smooth)
            vol = "unknown volume" if event.volume_ul is None else f"{event.volume_ul:g} µL"
            tqdm.write(f"[{format_time(frame.time_s)}] dispense -> {event.well} ({vol})")

        # 4) Report + object visualisation (last frame of every group of VLM_EVERY_N_FRAMES)
        if (frame.idx + 1) % S.VLM_EVERY_N_FRAMES == 0:
            self._update_report(frame, objects)
            self.vis.save_objects(frame, objects)

        # 5) Log + tracking visualisation
        self._log(frame, objects, pipette_seen, tracking, reading, p_raw, p_smooth,
                  press_label, event)
        self._visualise_tracking(frame, tracking, press_label, p_smooth, crop, event)

    def finish(self):
        self.tracker.finish()
        self.report.save(self.out, self.plate)
        self.frame_log.close()
        self.vis.finish()
        print(f"{len(self.plate.events)} dispenses into {len(self.plate.wells())} wells. "
              f"Report: {self.out / 'report.md'}, visualisations: {self.vis.root}")

    # ------------------------------------------------------------------------- steps
    def _classify_press(self, image, tracking):
        """(p(pressed), crop). p is None when the ray doesn't hit the plate or
        there's no crop; crop is None when it couldn't be made."""
        if self.press is None or tracking is None or tracking.hit_cell is None:
            return None, None
        crop = self.press.crop_for(image, tracking, self.cam_calib)
        if crop is None:
            return None, None
        return float(self.press.predict([crop])[0]), crop

    def _update_report(self, frame, objects):
        if self.vlm is not None:
            update = self.vlm.request_update(frame.image, frame.time_s, objects,
                                             self.plate, self.report)
            if update is not None:
                self.report.apply(update, frame.time_s)
        self.report.save(self.out, self.plate)

    def _visualise_tracking(self, frame, tracking, press_label, p_smooth, crop, event):
        # "no_crop": the ray is on the plate but the plunger crop couldn't be made
        if (not press_label and self.press is not None and tracking is not None
                and tracking.hit_cell is not None and crop is None):
            press_label = "no_crop"
        stable = self.volume.current
        volume_text = None if stable is None else f"{stable.volume_ul:g} uL"
        self.vis.save_tracking(frame, tracking, press_label, p_smooth, crop, volume_text,
                               event, self.plate)

    def _log(self, frame, objects, pipette_seen, tracking, reading, p_raw, p_smooth,
             press_label, event):
        stable = self.volume.current
        rec = {
            "frame_idx": frame.idx, "time_s": round(frame.time_s, 3),
            "timestamp_ns": frame.timestamp_ns,
            "objects": objects_to_dicts(objects, frame.image.shape),
            "pipette_detected": pipette_seen,
            "tracking": None if tracking is None else tracking.summary(),
            "well": (None if tracking is None or tracking.hit_cell is None
                     else well_name(*tracking.hit_cell)),
            "volume_reading": None if reading is None else reading.get("digits"),
            "volume_stable_ul": None if stable is None else stable.volume_ul,
            "press_p_raw": p_raw, "press_p": p_smooth, "press_label": press_label,
            "dispense": None if event is None else event.to_dict(),
        }
        self.frame_log.write(json.dumps(rec) + "\n")