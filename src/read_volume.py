"""
Read the pipette's volume counter (digit wheels in a window on the handle) from
camera frames. Lightweight: OpenCV + NumPy only, no OCR engine.

How it works
  1. Find the counter window: bright, elongated patches enclosed by the dark
     pipette body; the one with the most red (the red top wheel) wins.
  2. Straighten it: rotate the window so the digits stand upright, red wheel on
     top (the window reads top -> bottom when the pipette points tip-down).
  3. Read each of the N_DIGITS cells by template matching against digits 0-9.
     Templates come from TEMPLATES_DIR if it exists (your own labelled cells,
     see --collect below), otherwise from a rendered sans-serif font.
  4. Over a sequence of frames, the most common confident reading is reported
     as the stable volume (the setting rarely changes mid-run).

Usage
  python read_volume.py frame.png                 # one image
  python read_volume.py frames_dir/ --csv out.csv # every image in a folder
  python read_volume.py frames_dir/ --debug dbg/  # save the straightened windows
  python read_volume.py frames_dir/ --collect cells/
      saves every digit cell as cells/<frame>_<k>.png. Sort a few of each digit
      into TEMPLATES_DIR/0 ... TEMPLATES_DIR/9 to replace the font templates;
      this makes reading much more reliable.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------- CONFIG
N_DIGITS = 3                 # wheels visible in the window (top = red wheel)
UL_PER_UNIT = 10.0           # volume in µL = (digits read as an integer) * UL_PER_UNIT
                             # Gilson-style P1000: "025" = 250 µL -> 10.0 (check your model)
# Digits each wheel can show, top to bottom (None = any). On a P1000 the red top
# wheel is only ever 0 or 1, which removes most misreads of the faint red digit.
POSITION_DIGITS = ("01", None, None)
TEMPLATES_DIR = Path("volume_templates")   # optional: <dir>/<digit>/*.png

BRIGHT_V = 105               # wheel faces are brighter than this (HSV value)
DARK_V = 80                  # pipette body around the window is darker than this
MIN_AREA, MAX_AREA = 150, 8000   # window size range in pixels (depends on distance)
MIN_ELONGATION, MAX_ELONGATION = 1.8, 6.0
MIN_DARK_RING = 0.7          # fraction of the window's surroundings that must be dark
MIN_RED_FRAC, MAX_RED_FRAC = 0.02, 0.3   # red pixels inside the window: one wheel, not all
MIN_RED_AT_END = 0.7         # share of the red pixels lying in one end third of the window
MIN_CREAM_FRAC = 0.5         # bright window pixels that are cream (low saturation) wheel faces
CREAM_MAX_S = 70

CELL_W, CELL_H = 32, 40      # size of one straightened digit cell
MIN_SCORE = 0.5              # template correlation below this -> digit reported as "?"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# ----------------------------------------------------------------------------- /CONFIG


# ----------------------------------------------------------------------------- 1) find the window
def red_mask(img):
    b, g, r = (img[..., k].astype(np.int16) for k in range(3))
    return ((r - np.maximum(g, b)) > 40) & (r > 90)


def find_window(img):
    """Return (box, score) for the most likely counter window, or (None, 0).
    box = 4 corners (float32) of the window's rotated rectangle."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    bright = (V > BRIGHT_V).astype(np.uint8)
    dark = V < DARK_V
    red = red_mask(img)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        pts = np.column_stack(np.nonzero(lab[y:y + h, x:x + w] == i))[:, ::-1] + [x, y]
        rect = cv2.minAreaRect(pts.astype(np.float32))
        rw, rh = rect[1]
        elong = max(rw, rh) / max(1.0, min(rw, rh))
        if not MIN_ELONGATION <= elong <= MAX_ELONGATION:
            continue
        # Surroundings must be the dark pipette body
        pad = max(4, int(0.25 * np.sqrt(area)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        m = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = cv2.dilate(m, np.ones((2 * pad + 1,) * 2, np.uint8)) > m
        dark_frac = dark[y0:y1, x0:x1][ring].mean()
        if dark_frac < MIN_DARK_RING:
            continue
        # The counter window has a red top wheel: some red, concentrated at one end
        hull = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull((pts - [x0, y0]).astype(np.int32)), 1)
        red_in = red[y0:y1, x0:x1] & (hull > 0)
        red_frac = red_in.sum() / max(1, hull.sum())
        if not MIN_RED_FRAC <= red_frac <= MAX_RED_FRAC:
            continue
        ry, rx = np.nonzero(red_in)
        (cx, cy), _, _ = rect
        bp = cv2.boxPoints(rect)
        e1, e2 = bp[1] - bp[0], bp[2] - bp[1]
        long_edge = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        u = long_edge / np.linalg.norm(long_edge)
        t = ((np.column_stack([rx + x0, ry + y0]) - [cx, cy]) @ u) / (max(rw, rh) / 2)  # -1..1
        red_at_end = max((t < -1 / 3).mean(), (t > 1 / 3).mean())
        if red_at_end < MIN_RED_AT_END:
            continue
        # Wheel faces are cream, not a saturated coloured object
        blob = lab[y0:y1, x0:x1] == i
        cream = (S[y0:y1, x0:x1][blob] < CREAM_MAX_S).mean()
        if cream < MIN_CREAM_FRAC:
            continue
        score = dark_frac + red_at_end + cream
        if score > best_score:
            best, best_score = cv2.boxPoints(rect), score
    return best, best_score


# ----------------------------------------------------------------------------- 2) straighten
def straighten(img, box, pad_frac=0.08):
    """Warp the window into an upright strip (N_DIGITS cells stacked vertically),
    red wheel on top. Returns the BGR strip."""
    box = np.asarray(box, np.float32)
    # Same winding as the destination corners (clockwise on screen), otherwise the
    # warp would mirror the digits
    x, y = box[:, 0], box[:, 1]
    if (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() < 0:
        box = box[::-1].copy()
    # order corners so that edge 0->1 is a SHORT side (the strip's top/bottom)
    if np.linalg.norm(box[1] - box[0]) > np.linalg.norm(box[2] - box[1]):
        box = np.roll(box, -1, axis=0)
    # pad the box a little along both axes so edge strokes aren't cut off
    c = box.mean(axis=0)
    box = c + (box - c) * (1 + pad_frac)
    W, H = CELL_W, CELL_H * N_DIGITS
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    strip = cv2.warpPerspective(img, cv2.getPerspectiveTransform(box, dst), (W, H),
                                flags=cv2.INTER_CUBIC)
    # red wheel on top
    red = red_mask(strip)
    if red.any() and np.nonzero(red)[0].mean() > H / 2:
        strip = cv2.rotate(strip, cv2.ROTATE_180)
    return strip


def ink_image(cell_bgr):
    """Digit strokes (black or red) as bright-on-dark, contrast-normalised float.
    Red digits are measured by redness, which ignores the dark window corners."""
    b, g, r = (cell_bgr[..., k].astype(np.float32) for k in range(3))
    redness = np.clip(r - np.maximum(g, b), 0, None)
    if np.percentile(redness, 98) > 35:      # this cell holds a red digit
        ink = redness
        ink -= np.percentile(ink, 20)
        ink /= max(1e-6, np.percentile(ink, 99))
        return np.clip(ink, 0, 1)
    ink = 255 - cell_bgr.min(axis=2).astype(np.float32)    # black ink is dark in min()
    # Remove full-width bands (gaps between wheels) and full-height bands (window
    # edges): they are dark across the whole row/column, digit strokes are not.
    ink = np.clip(ink - np.median(ink, axis=1, keepdims=True), 0, None)
    ink = np.clip(ink - np.median(ink, axis=0, keepdims=True), 0, None)
    ink -= np.percentile(ink, 20)
    ink /= max(1e-6, np.percentile(ink, 99))
    return np.clip(ink, 0, 1)


# ----------------------------------------------------------------------------- 3) read the digits
def font_templates():
    """Digits 0-9 rendered in two sans-serif Hershey fonts."""
    out = {d: [] for d in range(10)}
    for font, thick in ((cv2.FONT_HERSHEY_SIMPLEX, 2), (cv2.FONT_HERSHEY_DUPLEX, 2),
                        (cv2.FONT_HERSHEY_SIMPLEX, 3)):
        for d in range(10):
            canvas = np.zeros((CELL_H * 3, CELL_W * 3), np.uint8)
            cv2.putText(canvas, str(d), (CELL_W // 2, CELL_H * 2), font, 2.0, 255, thick * 2,
                        cv2.LINE_AA)
            # blur + downsample like a ~10 px tall camera digit, then isolate it
            small = cv2.resize(canvas, (CELL_W, int(CELL_H * 1.0)), interpolation=cv2.INTER_AREA)
            g = isolate_glyph(cv2.GaussianBlur(small.astype(np.float32) / 255, (0, 0), 0.8))
            if g is not None:
                out[d].append(g)
    return out


def file_templates(root):
    out = {d: [] for d in range(10)}
    for d in range(10):
        for p in sorted((root / str(d)).glob("*.png")):
            cell = cv2.imread(str(p))
            if cell is not None:
                g = isolate_glyph(ink_image(cv2.resize(cell, (CELL_W, CELL_H))))
                if g is not None:
                    out[d].append(g)
    return out


def load_templates():
    """Labelled cells from TEMPLATES_DIR where available, rendered font for the
    digits that have no labelled examples yet."""
    font = font_templates()
    labelled = file_templates(TEMPLATES_DIR) if TEMPLATES_DIR.exists() else {}
    have = sorted(d for d, v in labelled.items() if v)
    t = {d: labelled[d] if d in have else font[d] for d in range(10)}
    if have:
        print(f"Templates: labelled cells for digits {have} from {TEMPLATES_DIR}"
              + (f", font for {sorted(set(range(10)) - set(have))}" if len(have) < 10 else ""))
    else:
        print(f"Templates: rendered font only. For reliable reading, run with --collect and "
              f"sort a few cells of each digit into {TEMPLATES_DIR}/0 ... /9")
    return t, have


GLYPH_W, GLYPH_H = 16, 24    # size digits are compared at


def isolate_glyph(ink, thr=0.35):
    """Crop the digit out of a cell's ink image: the tallest stroke component
    (digits are taller than glare blobs), plus anything inside its bounding box,
    resized to GLYPH_W x GLYPH_H. None if nothing digit-like is found."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((ink > thr).astype(np.uint8), 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_HEIGHT]))
    x, y, w, h = st[k, :4]
    if h < 0.25 * ink.shape[0]:
        return None
    # widen narrow glyphs (a "1") to the usual aspect so they aren't stretched
    cx, w = x + w / 2, max(w, h * GLYPH_W / GLYPH_H)
    x0, x1 = int(max(0, cx - w / 2)), int(min(ink.shape[1], cx + w / 2 + 1))
    g = ink[y:y + h, x0:x1]
    return cv2.resize(g, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA)


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def read_cell(cell_bgr, templates, allowed=None):
    """(digit, score): normalised cross-correlation between the isolated digit
    and every template glyph of the allowed digits."""
    g = isolate_glyph(ink_image(cell_bgr))
    if g is None:
        return None, 0.0
    best_d, best_s = None, -1.0
    for d, ts in templates.items():
        if allowed is not None and str(d) not in allowed:
            continue
        for t in ts:
            s = ncc(g, t)
            if s > best_s:
                best_d, best_s = d, s
    return best_d, best_s


def read_frame(img, templates):
    """{'found', 'digits', 'scores', 'value', 'strip'} for one frame."""
    box, _ = find_window(img)
    if box is None:
        return {"found": False}
    strip = straighten(img, box)
    digits, scores = [], []
    for k in range(N_DIGITS):
        allowed = POSITION_DIGITS[k] if POSITION_DIGITS and k < len(POSITION_DIGITS) else None
        d, s = read_cell(strip[k * CELL_H:(k + 1) * CELL_H], templates, allowed)
        digits.append(str(d) if s >= MIN_SCORE else "?")
        scores.append(s)
    text = "".join(digits)
    value = int(text) * UL_PER_UNIT if "?" not in text else None
    return {"found": True, "digits": text, "scores": scores, "value": value,
            "strip": strip, "box": box}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="image file or folder of frames")
    ap.add_argument("--csv", type=Path, help="write per-frame readings here")
    ap.add_argument("--debug", type=Path, help="save straightened windows + overlay here")
    ap.add_argument("--collect", type=Path, help="save every digit cell here (to build templates)")
    args = ap.parse_args()

    paths = ([args.input] if args.input.is_file() else
             sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS))
    templates, _ = load_templates()
    for d in (args.debug, args.collect):
        if d:
            d.mkdir(parents=True, exist_ok=True)

    rows, confident = [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        r = read_frame(img, templates)
        if not r["found"]:
            rows.append([p.name, 0, "", "", ""])
            continue
        rows.append([p.name, 1, r["digits"], " ".join(f"{s:.2f}" for s in r["scores"]),
                     "" if r["value"] is None else f"{r['value']:g}"])
        if r["value"] is not None:
            confident.append(r["digits"])
        if len(paths) == 1 or args.debug:
            print(f"{p.name}: {r['digits']}  (scores {' '.join(f'{s:.2f}' for s in r['scores'])})"
                  + (f"  -> {r['value']:g} µL" if r["value"] is not None else ""))
        if args.debug:
            big = cv2.resize(r["strip"], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            for k, ch in enumerate(r["digits"]):
                cv2.putText(big, ch, (big.shape[1] - 30, (k + 1) * CELL_H * 4 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(str(args.debug / f"{p.stem}_strip.png"), big)
            ov = img.copy()
            cv2.polylines(ov, [r["box"].astype(np.int32)], True, (0, 255, 0), 2)
            x, y = r["box"].min(axis=0).astype(int)
            cv2.putText(ov, r["digits"], (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(str(args.debug / f"{p.stem}_overlay.jpg"), ov)
        if args.collect:
            for k in range(N_DIGITS):
                cv2.imwrite(str(args.collect / f"{p.stem}_{k}.png"),
                            r["strip"][k * CELL_H:(k + 1) * CELL_H])

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "window_found", "digits", "scores", "volume_ul"])
            w.writerows(rows)
        print(f"Wrote {args.csv}")
    found = sum(r[1] for r in rows)
    print(f"Window found in {found}/{len(rows)} frames, confident reading in {len(confident)}")
    if confident:
        (value, count), = Counter(confident).most_common(1)
        print(f"Most common reading: {value} ({count}/{len(confident)} confident frames)"
              f" -> {int(value) * UL_PER_UNIT:g} µL")


if __name__ == "__main__":
    main()