"""
Label plunger crops as pressed / not pressed by marking where each segment starts.

Usage:  python label_tool.py <session> [--start FRAME]

Labels are stored as change points in data/labels/<session>.csv: pressing a label key
at a frame means "from this frame on, the label is X" (until the next change point).
So for each press you only mark two frames: its first frame (p) and the first frame
after it (n). Mark a few blurry/half-pressed frames around transitions with i (ignore)
if you like; build_dataset.py also drops a margin around every transition.
Every change is saved immediately.

Keys
  right / d        next frame             left / a     previous frame
  up / w           +10 frames             down / s     -10 frames
  ] / [            next / previous change point
  space            play / pause
  p / n / i        pressed / not pressed / ignore, from this frame on
  x                delete the change point at this frame
  q / esc          quit
"""
import argparse
from collections import OrderedDict

import cv2
import numpy as np

import common
import config as C

VIEW_H = 720                     # display height of the main crop
PANEL_W = 420
TIMELINE_H = 26
PLAY_DELAY_MS = 60
COLOURS = {"pressed": (0, 200, 0), "not_pressed": (200, 120, 0), "ignore": (0, 200, 255),
           C.UNLABELLED: (70, 70, 70)}

KEYS_NEXT = {ord("d"), 65363, 2555904, 63235}
KEYS_PREV = {ord("a"), 65361, 2424832, 63234}
KEYS_UP10 = {ord("w"), 65362, 2490368, 63232}
KEYS_DOWN10 = {ord("s"), 65364, 2621440, 63233}


class CropCache:
    def __init__(self, session, size=64):
        self.session, self.size, self.data = session, size, OrderedDict()

    def get(self, row):
        key = row["frame_idx"]
        if key not in self.data:
            img = cv2.imread(str(C.CROPS_DIR / self.session / row["crop"])) if row["marker_found"] else None
            self.data[key] = img
            if len(self.data) > self.size:
                self.data.popitem(last=False)
        return self.data[key]


def put(img, s, org, colour=(255, 255, 255), scale=0.6, th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, th, cv2.LINE_AA)


def timeline(labels, pos, width, lo, hi):
    """Colour bar of labels[lo:hi] stretched to `width`, with a cursor at pos."""
    bar = np.zeros((TIMELINE_H, width, 3), np.uint8)
    n = hi - lo
    for x in range(width):
        k = lo + min(int(x * n / width), n - 1)
        bar[:, x] = COLOURS.get(labels[k], (70, 70, 70))
    cx = int((pos - lo + 0.5) * width / n)
    cv2.line(bar, (cx, 0), (cx, TIMELINE_H - 1), (255, 255, 255), 2)
    return bar


def render(session, index, labels, points, pos, cache, playing):
    row = index[pos]
    xs, ys = common.crop_axes()
    view_w = int(round(VIEW_H * len(xs) / len(ys)))
    crop = cache.get(row)
    if crop is None:
        main = np.full((VIEW_H, view_w, 3), 35, np.uint8)
        put(main, "no tag in this frame", (20, VIEW_H // 2), (180, 180, 180), 0.8, 2)
    else:
        main = cv2.resize(crop, (view_w, VIEW_H), interpolation=cv2.INTER_LINEAR)

    # Side panel: previous / next frames for context + status
    panel = np.full((VIEW_H, PANEL_W, 3), 25, np.uint8)
    tw = (PANEL_W - 30) // 2
    tht = int(round(tw * len(ys) / len(xs)))
    for k, (off, title) in enumerate(((-1, "previous"), (1, "next"))):
        j = pos + off
        if 0 <= j < len(index):
            t = cache.get(index[j])
            if t is not None:
                panel[30:30 + tht, 10 + k * (tw + 10):10 + k * (tw + 10) + tw] = cv2.resize(t, (tw, tht))
            put(panel, f"{title} ({index[j]['frame_idx']})", (10 + k * (tw + 10), 22), scale=0.45)
    y = 30 + tht + 35
    fi = row["frame_idx"]
    label = labels[pos]
    put(panel, f"{session}", (10, y), scale=0.55)
    put(panel, f"frame {fi}   ({pos + 1}/{len(index)})", (10, y + 30), scale=0.6)
    put(panel, f"label: {label}", (10, y + 62), COLOURS.get(label, (255, 255, 255)), 0.75, 2)
    if fi in points:
        put(panel, "change point here", (10, y + 92), (255, 255, 255), 0.5)
    if row["marker_found"]:
        put(panel, f"tag: dist {float(row['dist_m']) * 100:.0f} cm, view {float(row['view_deg']):.0f} deg",
            (10, y + 122), scale=0.45)
        amb = float(row["ambiguity"])
        put(panel, f"ambiguity {amb:.2f}" + ("  (pose may flip)" if amb > 0.5 else ""),
            (10, y + 146), (0, 200, 255) if amb > 0.5 else (255, 255, 255), 0.45)
    if playing:
        put(panel, "PLAYING", (PANEL_W - 110, VIEW_H - 15), (0, 255, 255), 0.6, 2)
    put(panel, "p/n/i label  x delete  space play  q quit", (10, VIEW_H - 40), scale=0.42)
    put(panel, "arrows/a,d step  w,s +-10  [ ] jump", (10, VIEW_H - 18), scale=0.42)

    top = np.hstack([main, panel])
    W = top.shape[1]
    full = timeline(labels, pos, W, 0, len(index))
    lo, hi = max(0, pos - 150), min(len(index), pos + 150)
    local = timeline(labels, pos, W, lo, hi)
    put(local, f"{index[lo]['frame_idx']}", (4, 18), scale=0.45)
    put(local, f"{index[hi - 1]['frame_idx']}", (W - 60, 18), scale=0.45)
    gap = np.full((4, W, 3), 0, np.uint8)
    return np.vstack([top, gap, local, gap, full])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--start", type=int, default=None, help="frame index to start at")
    args = ap.parse_args()

    index = common.read_index(args.session)
    frame_ids = [r["frame_idx"] for r in index]
    points = common.load_change_points(args.session)
    labels = common.expand_labels(points, frame_ids)
    cache = CropCache(args.session)
    pos = 0
    if args.start is not None:
        pos = int(np.clip(np.searchsorted(frame_ids, args.start), 0, len(index) - 1))
    playing = False
    win = f"label_tool - {args.session}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(win, render(args.session, index, labels, points, pos, cache, playing))
        key = cv2.waitKeyEx(PLAY_DELAY_MS if playing else 0)
        if key == -1:                       # timeout while playing, or window closed
            if not playing or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if pos < len(index) - 1:
                pos += 1
            else:
                playing = False
            continue
        k8 = key & 0xFF
        changed = False
        if key in KEYS_NEXT:
            pos = min(pos + 1, len(index) - 1)
        elif key in KEYS_PREV:
            pos = max(pos - 1, 0)
        elif key in KEYS_UP10:
            pos = min(pos + 10, len(index) - 1)
        elif key in KEYS_DOWN10:
            pos = max(pos - 10, 0)
        elif k8 == ord(" "):
            playing = not playing
        elif k8 == ord("]"):
            later = [k for k in sorted(points) if k > frame_ids[pos]]
            if later:
                pos = frame_ids.index(later[0]) if later[0] in frame_ids else pos
        elif k8 == ord("["):
            earlier = [k for k in sorted(points) if k < frame_ids[pos]]
            if earlier:
                pos = frame_ids.index(earlier[-1]) if earlier[-1] in frame_ids else pos
        elif chr(k8) in C.LABELS:
            points[frame_ids[pos]] = C.LABELS[chr(k8)]
            changed = True
        elif k8 == ord("x"):
            changed = points.pop(frame_ids[pos], None) is not None
        elif k8 in (ord("q"), 27):
            break
        if changed:
            common.save_change_points(args.session, points)
            labels = common.expand_labels(points, frame_ids)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    print(f"Saved {len(points)} change points to {common.labels_path(args.session)}")


if __name__ == "__main__":
    main()
