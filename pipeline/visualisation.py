"""
Visualisations, saved to <out>/visus/ (drawing code from detect_apriltags.py):

  tracking/  every frame (5 Hz): camera image with tag outlines + IDs, marker axes,
             fused pipette axes (thicker), plate box + dashed well grid, hit cell
             (green = not pressed, red = pressed, orange = no crop), ray with its
             distance, status banner (press, well, volume, dispense) and the
             plunger crop fed to the classifier as an inset
  abstract/  every frame: same size, white on black: plate box, grid, hit cell,
             pipette origin, ray, banner, plus a map of the wells filled so far
  objects/   VLM frames only (1 Hz): detector boxes + labels and SAM masks
  tracking_stacked.mp4   tracking/ above abstract/ (make_video.py), at the end of the run
"""
import cv2
import numpy as np

import settings as S
from experiment_state import format_time
from geometry import project_points
from make_video import make_stacked_video
from plate import grid_cell_quad, well_name

# ----------------------------------------------------------------------------- colours (BGR)
PLATE_BOX_COLOUR = (0, 215, 255)       # yellow-orange
GRID_COLOUR = (0, 255, 255)            # yellow
GRID_ALPHA = 0.45
GRID_DASH_SAMPLES = 60                 # samples per grid line; every other segment drawn
HIT_CELL_COLOUR = (0, 255, 0)          # green
HIT_CELL_ALPHA = 0.5
RAY_COLOUR = (255, 0, 255)             # magenta
PRESSED_COLOUR = (0, 0, 255)           # red
NOT_PRESSED_COLOUR = HIT_CELL_COLOUR
UNKNOWN_COLOUR = (0, 200, 255)         # orange: ray hits but no crop
OTHER_MARKER_COLOUR = (0, 0, 255)      # tags not in TAG_SIZE_M
ABSTRACT_COLOUR = (255, 255, 255)
ABSTRACT_GRID_ALPHA = 0.45
ABSTRACT_CELL_ALPHA = 0.6
OBJECT_MASK_ALPHA = 0.4
OBJECT_PALETTE = [(255, 128, 0), (0, 200, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255),
                  (255, 255, 0), (128, 0, 255), (0, 0, 255), (128, 255, 128), (255, 128, 255)]


def press_colour(label, default=None):
    return {"pressed": PRESSED_COLOUR, "not_pressed": NOT_PRESSED_COLOUR,
            "no_crop": UNKNOWN_COLOUR}.get(label, default)


# ----------------------------------------------------------------------------- 3D drawing primitives
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


def draw_line_3d(image, p0, p1, cam_calib, mapping, colour, thickness, samples=20):
    """Straight 3D segment, sampled so it curves correctly under fisheye distortion."""
    uv = project_points(cam_calib, np.linspace(p0, p1, samples))
    if np.isnan(uv).any():
        return
    uv = mapping.to_frame(uv).round().astype(np.int32)
    cv2.polylines(image, [uv.reshape(-1, 1, 2)], False, colour, thickness, cv2.LINE_AA)


def draw_dashed_line_3d(image, p0, p1, cam_calib, mapping, colour, thickness,
                        samples=GRID_DASH_SAMPLES):
    uv = project_points(cam_calib, np.linspace(p0, p1, samples + 1))
    if np.isnan(uv).any():
        return
    uv = mapping.to_frame(uv).round().astype(int)
    for k in range(0, samples, 2):
        cv2.line(image, tuple(uv[k]), tuple(uv[k + 1]), colour, thickness, cv2.LINE_AA)


def draw_well_grid(image, plate_points, cam_calib, mapping, colour=GRID_COLOUR, alpha=GRID_ALPHA):
    """Semi-transparent dashed GRID_ROWS x GRID_COLS grid on the box's top face."""
    g0, g1, _, g3 = plate_points[0:4]
    ev, eu = g1 - g0, g3 - g0
    overlay = image.copy()
    for i in range(S.GRID_ROWS + 1):
        a = g0 + (i / S.GRID_ROWS) * ev
        draw_dashed_line_3d(overlay, a, a + eu, cam_calib, mapping, colour, 1)
    for j in range(S.GRID_COLS + 1):
        a = g0 + (j / S.GRID_COLS) * eu
        draw_dashed_line_3d(overlay, a, a + ev, cam_calib, mapping, colour, 1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def draw_plate_box(image, corners_cam, cam_calib, mapping, label, colour=PLATE_BOX_COLOUR):
    top, bottom = corners_cam[:4], corners_cam[4:8]
    for i in range(4):
        j = (i + 1) % 4
        draw_line_3d(image, top[i], top[j], cam_calib, mapping, colour, 3)
        if S.PLATE_HEIGHT_M > 0:
            draw_line_3d(image, bottom[i], bottom[j], cam_calib, mapping, colour, 1)
            draw_line_3d(image, top[i], bottom[i], cam_calib, mapping, colour, 1)
    uv = project_points(cam_calib, top[3:4])
    if not np.isnan(uv).any():
        x, y = mapping.to_frame(uv)[0].round().astype(int)
        cv2.putText(image, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)


def fill_grid_cell(image, plate_points, row, col, cam_calib, mapping, samples=8,
                   colour=HIT_CELL_COLOUR, alpha=HIT_CELL_ALPHA):
    quad = grid_cell_quad(plate_points, row, col)
    pts = np.concatenate([np.linspace(quad[k], quad[(k + 1) % 4], samples, endpoint=False)
                          for k in range(4)])
    uv = project_points(cam_calib, pts)
    if np.isnan(uv).any():
        return
    uv = mapping.to_frame(uv).round().astype(np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [uv.reshape(-1, 1, 2)], colour, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def draw_ray(image, p_marker, p_enter, dist, cam_calib, mapping, colour=RAY_COLOUR):
    draw_line_3d(image, p_marker, p_enter, cam_calib, mapping, colour, 2)
    uv = project_points(cam_calib, np.array([p_marker, p_enter, (p_marker + p_enter) / 2]))
    if np.isnan(uv).any():
        return
    uv = mapping.to_frame(uv).round().astype(int)
    cv2.circle(image, tuple(uv[0]), 6, colour, -1, cv2.LINE_AA)
    cv2.circle(image, tuple(uv[1]), 4, colour, -1, cv2.LINE_AA)
    put_text(image, f"{dist * 100:.1f} cm", (int(uv[2][0]) + 10, int(uv[2][1])), colour, 0.9)


def draw_marker_position(image, p, label, cam_calib, mapping, colour):
    uv = project_points(cam_calib, np.array([p]))
    if np.isnan(uv).any():
        return
    x, y = mapping.to_frame(uv)[0].round().astype(int)
    cv2.circle(image, (x, y), 8, colour, -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), 14, colour, 1, cv2.LINE_AA)
    cv2.putText(image, label, (x + 18, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------- 2D overlays
def put_text(image, text, org, colour, scale, thickness=2):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def draw_status(image, lines, colour):
    """Banner in the top-left corner, one entry per line."""
    for k, text in enumerate(lines):
        put_text(image, text, (30, 70 + 60 * k), colour, 1.6, 3)


def draw_footer(image, text):
    put_text(image, text, (30, image.shape[0] - 30), (255, 255, 255), 1.0)


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


def draw_plate_map(image, plate, cell_px=28, colour=ABSTRACT_COLOUR):
    """Map of the wells filled so far (bottom-right corner); the latest is red."""
    rows, cols = plate.rows, plate.cols
    filled = plate.wells()
    last = plate.events[-1].well if plate.events else None
    x0 = image.shape[1] - (cols + 1) * cell_px - 20
    y0 = image.shape[0] - (rows + 1) * cell_px - 20
    if x0 < 0 or y0 < 0:
        return
    for c in range(cols):
        cv2.putText(image, str(c + 1), (x0 + (c + 1) * cell_px + 4, y0 + cell_px - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    for r in range(rows):
        letter = chr(ord("A") + r)
        y = y0 + (r + 1) * cell_px
        cv2.putText(image, letter, (x0 + 8, y + cell_px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour, 1, cv2.LINE_AA)
        for c in range(cols):
            x = x0 + (c + 1) * cell_px
            name = f"{letter}{c + 1}"
            centre, rad = (x + cell_px // 2, y + cell_px // 2), cell_px // 2 - 3
            if name in filled:
                cv2.circle(image, centre, rad, PRESSED_COLOUR if name == last else HIT_CELL_COLOUR,
                           -1, cv2.LINE_AA)
            cv2.circle(image, centre, rad, colour, 1, cv2.LINE_AA)
    put_text(image, f"{len(filled)} wells / {len(plate.events)} dispenses",
             (x0, y0 - 10), colour, 0.6, 1)


def draw_objects(image, objects):
    """Detector boxes + labels, SAM masks filled and outlined."""
    overlay = image.copy()
    for o in objects:
        colour = _object_colour(o.label)
        if o.mask is not None:
            overlay[o.mask] = colour
    cv2.addWeighted(overlay, OBJECT_MASK_ALPHA, image, 1 - OBJECT_MASK_ALPHA, 0, dst=image)
    for o in objects:
        colour = _object_colour(o.label)
        if o.mask is not None:
            contours, _ = cv2.findContours(o.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, colour, 3, cv2.LINE_AA)
        x1, y1, x2, y2 = (int(round(v)) for v in o.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        put_text(image, f"{o.label} {o.confidence:.2f}", (x1 + 4, max(30, y1 - 10)), colour, 1.0)


def _object_colour(label):
    idx = S.OBJECT_LABELS.index(label) if label in S.OBJECT_LABELS else sum(map(ord, label))
    return OBJECT_PALETTE[idx % len(OBJECT_PALETTE)]


# ----------------------------------------------------------------------------- frame views
def status_lines(tracking, press_label, press_p, volume_text, event):
    lines = []
    if press_label:
        text = {"pressed": "PRESSED", "not_pressed": "not pressed", "no_crop": "no crop"}[press_label]
        if press_p is not None and np.isfinite(press_p):
            text += f"  p={press_p:.2f}"
        lines.append(text)
    if tracking is not None and tracking.hit_cell is not None:
        lines.append(f"pointing at {well_name(*tracking.hit_cell)}")
    if volume_text:
        lines.append(f"volume {volume_text}")
    if event is not None:
        lines.append(f"DISPENSE -> {event.well}")
    return lines


def tracking_view(image, cam_calib, tracking, press_label, press_p, crop, lines):
    """Annotated camera frame (the 'annotated' view of detect_apriltags.py)."""
    out = image.copy()
    colour = press_colour(press_label, HIT_CELL_COLOUR)
    if tracking is None:
        draw_status(out, ["tracking skipped (no pipette detected)"], UNKNOWN_COLOUR)
        return out
    m, det = tracking.mapping, tracking.detections
    for mid, (T, _) in det.poses.items():
        draw_axes(out, T, S.TAG_SIZE_M[mid] * 0.75, cam_calib, m, 1)
    for T, *_ in tracking.body_poses.values():
        draw_axes(out, T, 0.04, cam_calib, m, 4)
    pc = tracking.plate_corners
    if pc is not None:
        draw_well_grid(out, pc, cam_calib, m)
        if tracking.hit_cell is not None:
            fill_grid_cell(out, pc, *tracking.hit_cell, cam_calib, m, colour=colour)
        draw_plate_box(out, pc, cam_calib, m, f"plate ({tracking.plate_source})")
    if tracking.hit_dist is not None:
        draw_ray(out, tracking.ray_origin, tracking.hit_point, tracking.hit_dist, cam_calib, m)
    for group, border in ((det.tags, (0, 255, 0)), (det.others, OTHER_MARKER_COLOUR)):
        if group:
            cv2.aruco.drawDetectedMarkers(
                out, [c.reshape(1, 4, 2).astype(np.float32) for c, _ in group],
                np.array([i for _, i in group]).reshape(-1, 1), borderColor=border)
    draw_status(out, lines, colour)
    if crop is not None:
        draw_inset(out, crop, colour)
    return out


def abstract_view(shape, cam_calib, tracking, press_label, press_p, plate, lines):
    """White-on-black view of plate box, grid, hit cell, pipette origin and ray."""
    out = np.zeros(shape, np.uint8)
    W = ABSTRACT_COLOUR
    if tracking is not None:
        m, pc = tracking.mapping, tracking.plate_corners
        if pc is not None:
            draw_well_grid(out, pc, cam_calib, m, colour=W, alpha=ABSTRACT_GRID_ALPHA)
            if tracking.hit_cell is not None:
                fill_grid_cell(out, pc, *tracking.hit_cell, cam_calib, m,
                               colour=press_colour(press_label, HIT_CELL_COLOUR),
                               alpha=ABSTRACT_CELL_ALPHA)
            draw_plate_box(out, pc, cam_calib, m, f"plate ({tracking.plate_source})", colour=W)
        if tracking.hit_dist is not None:
            draw_ray(out, tracking.ray_origin, tracking.hit_point, tracking.hit_dist,
                     cam_calib, m, colour=W)
        body = tracking.body_poses.get(S.RAY_BODY)
        if body is not None:
            draw_marker_position(out, body[0][:3, 3], f"{S.RAY_BODY} ({'+'.join(map(str, body[2]))})",
                                 cam_calib, m, W)
    draw_status(out, lines, press_colour(press_label, W))
    draw_plate_map(out, plate)
    return out


def objects_view(image, objects, header):
    out = image.copy()
    draw_objects(out, objects)
    put_text(out, header, (30, 70), (255, 255, 255), 1.4, 3)
    return out


# ----------------------------------------------------------------------------- saving
class Visualiser:
    """Writes the visualisations of each frame to <out>/visus/<view>/frame_XXXXXX.jpg."""

    def __init__(self, out_dir, cam_calib):
        self.cam_calib = cam_calib
        self.root = out_dir / S.VIS_DIR_NAME
        self.dirs = {name: self.root / name for name, on in (
            ("tracking", S.VIS_TRACKING), ("abstract", S.VIS_ABSTRACT), ("objects", S.VIS_OBJECTS)) if on}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def _path(self, view, frame):
        return self.dirs[view] / f"frame_{frame.idx:06d}.jpg"

    def save_tracking(self, frame, tracking, press_label, press_p, crop, volume_text, event, plate):
        if "tracking" not in self.dirs and "abstract" not in self.dirs:
            return
        lines = status_lines(tracking, press_label, press_p, volume_text, event)
        footer = f"{format_time(frame.time_s)}  frame {frame.idx}"
        if "tracking" in self.dirs:
            img = tracking_view(frame.image, self.cam_calib, tracking, press_label, press_p, crop, lines)
            draw_footer(img, footer)
            cv2.imwrite(str(self._path("tracking", frame)), img)
        if "abstract" in self.dirs:
            img = abstract_view(frame.image.shape, self.cam_calib, tracking, press_label, press_p,
                                plate, lines)
            draw_footer(img, footer)
            cv2.imwrite(str(self._path("abstract", frame)), img)

    def save_objects(self, frame, objects):
        if "objects" not in self.dirs:
            return
        header = f"{format_time(frame.time_s)}  frame {frame.idx}  {len(objects)} objects"
        cv2.imwrite(str(self._path("objects", frame)), objects_view(frame.image, objects, header))

    def finish(self):
        if S.VIS_VIDEO and {"tracking", "abstract"} <= self.dirs.keys():
            make_stacked_video(self.dirs["tracking"], self.dirs["abstract"],
                               self.root / "tracking_stacked.mp4", fps=S.FRAME_RATE_HZ,
                               slowdown=S.VIDEO_SLOWDOWN, scale=S.VIDEO_SCALE,
                               interp_steps=S.VIDEO_INTERP_STEPS)