"""
Object detection for the VLM context (and the pipette gate for tracking).

MobileSAM segments whatever it is prompted with but doesn't name objects, so an
open-vocabulary detector (YOLO-World, prompted with OBJECT_LABELS) finds and
labels boxes, and MobileSAM turns each box into a mask; the box sent onwards is
the tight box around that mask. Both models come from `ultralytics` and their
weights download automatically on first use.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import settings as S


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: list                          # [x1, y1, x2, y2] in frame pixels
    mask: Optional[np.ndarray] = None   # bool (H, W), if USE_SAM_MASKS
    mask_area_frac: Optional[float] = None   # mask area / image area

    def to_dict(self, image_shape):
        """JSON-friendly form. box_2d is [ymin, xmin, ymax, xmax] normalised to
        0-1000 (Gemini's native box convention, independent of image size)."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = self.bbox
        d = {"label": self.label, "confidence": round(self.confidence, 2),
             "box_2d": [int(round(1000 * y1 / h)), int(round(1000 * x1 / w)),
                        int(round(1000 * y2 / h)), int(round(1000 * x2 / w))]}
        if self.mask_area_frac is not None:
            d["area_frac"] = round(self.mask_area_frac, 4)
        return d


def objects_to_dicts(objects, image_shape):
    return [o.to_dict(image_shape) for o in sorted(objects, key=lambda o: -o.confidence)]


class ObjectDetector:
    def __init__(self, labels=None, detector_weights=S.DETECTOR_WEIGHTS,
                 sam_weights=S.SAM_WEIGHTS, conf=S.DETECTION_CONF,
                 use_masks=S.USE_SAM_MASKS, device=S.DEVICE):
        from ultralytics import SAM, YOLOWorld
        self.labels = list(labels if labels is not None else S.OBJECT_LABELS)
        self.conf = conf
        self.kwargs = {"verbose": False, **({"device": device} if device else {})}
        self.detector = YOLOWorld(detector_weights)
        self.detector.set_classes(self.labels)
        self.sam = SAM(sam_weights) if use_masks else None
        print(f"Object detector: {detector_weights} ({len(self.labels)} labels)"
              + (f" + {sam_weights} masks" if use_masks else ""))

    def detect(self, image) -> list:
        res = self.detector.predict(image, conf=self.conf, **self.kwargs)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        objects = [DetectedObject(label=res.names[int(c)], confidence=float(p), bbox=b.tolist())
                   for b, c, p in zip(boxes, res.boxes.cls.cpu().numpy(),
                                      res.boxes.conf.cpu().numpy())]
        if self.sam is not None:
            self._add_masks(image, boxes, objects)
        return objects

    def _add_masks(self, image, boxes, objects):
        h, w = image.shape[:2]
        res = self.sam.predict(image, bboxes=boxes.tolist(), **self.kwargs)[0]
        if res.masks is None:
            return
        for obj, m in zip(objects, res.masks.data.cpu().numpy()):
            m = m > 0.5
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            obj.mask = m
            obj.mask_area_frac = float(len(xs)) / (h * w)
            obj.bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
