"""
Turn crops + label change points into a dataset table with events and a train/test split.

Usage:  python build_dataset.py            # uses every session that has a labels file

Output: data/dataset/dataset.csv, one row per frame of every labelled session:
  session, frame_idx, timestamp_ns, crop (path relative to data/), label, target
  (1 pressed, 0 not pressed, -1 other), event_id (run of identical pressed/not_pressed
  labels), split (train/test), usable (crop exists, quality OK, not near a transition,
  label is a class), drop_reason, train_pick (subsampled training frame), tag quality.

Splitting never cuts through an event. Use whole sessions for testing whenever you can
(TEST_SESSIONS); otherwise the last TEST_FRACTION of each session becomes test data.
"""
import csv
from collections import Counter, defaultdict

import numpy as np

import common
import config as C

# ----------------------------------------------------------------------------- SETTINGS
TEST_SESSIONS = []            # e.g. ["p1000_154354"]; empty -> time split within sessions
TEST_FRACTION = 0.3           # used when TEST_SESSIONS is empty
TRANSITION_MARGIN = 2         # drop up to this many frames at each end of every event ...
MIN_FRAMES_KEPT = 1           # ... but always keep at least this many in its middle, so
                              # short presses aren't dropped entirely
TRAIN_FRAMES_PER_EVENT = 5    # evenly spaced training frames per event (None = all)
MAX_REPROJ_PX = 3.0           # drop frames whose tag fit is worse than this (None = off)
MAX_AMBIGUITY = 0.8           # drop frames whose tag orientation may flip (None = off)
# ----------------------------------------------------------------------------- /SETTINGS

FIELDS = ["session", "frame_idx", "timestamp_ns", "crop", "label", "target", "event_id",
          "split", "usable", "drop_reason", "train_pick", "reproj_px", "ambiguity",
          "dist_m", "view_deg"]


def build_session(session):
    index = common.read_index(session)
    points = common.load_change_points(session)
    labels = common.expand_labels(points, [r["frame_idx"] for r in index])
    n = len(index)

    # Events: runs of identical class labels (unlabelled / ignore break runs)
    event_ids, k, prev = [None] * n, -1, None
    for j, lab in enumerate(labels):
        if lab in C.CLASSES:
            if lab != prev:
                k += 1
            event_ids[j] = f"{session}:{k:04d}"
        prev = lab

    # Frames near a label change: trim up to TRANSITION_MARGIN frames from each end of
    # every event, shrinking the margin for short events so that at least
    # MIN_FRAMES_KEPT frames in the middle survive.
    near_change = np.zeros(n, bool)
    start = 0
    for j in range(1, n + 1):
        if j == n or labels[j] != labels[start]:
            if labels[start] in C.CLASSES:
                length = j - start
                m = min(TRANSITION_MARGIN, max(0, (length - MIN_FRAMES_KEPT) // 2))
                near_change[start:start + m] = True
                near_change[j - m:j] = True
            start = j

    rows = []
    for j, (r, lab) in enumerate(zip(index, labels)):
        reason = ""
        if lab not in C.CLASSES:
            reason = lab
        elif not r["marker_found"]:
            reason = "no_tag"
        elif near_change[j]:
            reason = "transition"
        elif MAX_REPROJ_PX is not None and float(r["reproj_px"]) > MAX_REPROJ_PX:
            reason = "reproj"
        elif MAX_AMBIGUITY is not None and float(r["ambiguity"]) > MAX_AMBIGUITY:
            reason = "ambiguous"
        rows.append({
            "session": session, "frame_idx": r["frame_idx"], "timestamp_ns": r["timestamp_ns"],
            "crop": f"crops/{session}/{r['crop']}" if r["marker_found"] else "",
            "label": lab, "target": C.CLASSES.index(lab) if lab in C.CLASSES else -1,
            "event_id": event_ids[j] or "", "usable": int(reason == ""), "drop_reason": reason,
            "train_pick": 0, "reproj_px": r["reproj_px"], "ambiguity": r["ambiguity"],
            "dist_m": r["dist_m"], "view_deg": r["view_deg"],
        })
    return rows, points


def assign_split(session, rows, points):
    if TEST_SESSIONS:
        split = "test" if session in TEST_SESSIONS else "train"
        for r in rows:
            r["split"] = split
        return
    # Time split: cut at the label change point closest to (1 - TEST_FRACTION) of the
    # labelled frames, so that no event is split between train and test.
    labelled = [r["frame_idx"] for r in rows if r["label"] != C.UNLABELLED]
    if not labelled:
        for r in rows:
            r["split"] = "train"
        return
    target = labelled[int(len(labelled) * (1 - TEST_FRACTION))]
    cands = [k for k in points if labelled[0] < k <= labelled[-1]]
    cut = min(cands, key=lambda k: abs(k - target)) if cands else labelled[-1] + 1
    for r in rows:
        r["split"] = "test" if r["frame_idx"] >= cut else "train"


def pick_training_frames(rows):
    by_event = defaultdict(list)
    for r in rows:
        if r["split"] == "train" and r["usable"]:
            by_event[r["event_id"]].append(r)
    for ev_rows in by_event.values():
        if TRAIN_FRAMES_PER_EVENT is None or len(ev_rows) <= TRAIN_FRAMES_PER_EVENT:
            chosen = ev_rows
        else:
            idx = np.linspace(0, len(ev_rows) - 1, TRAIN_FRAMES_PER_EVENT).round().astype(int)
            chosen = [ev_rows[i] for i in np.unique(idx)]
        for r in chosen:
            r["train_pick"] = 1


def summarise(rows):
    print(f"\n{'split':6} {'class':12} {'events':>7} {'usable':>7} {'picked':>7}")
    for split in ("train", "test"):
        for cls in C.CLASSES:
            sel = [r for r in rows if r["split"] == split and r["label"] == cls]
            events = {r["event_id"] for r in sel if r["usable"]}
            usable = sum(r["usable"] for r in sel)
            picked = sum(r["train_pick"] for r in sel)
            print(f"{split:6} {cls:12} {len(events):7d} {usable:7d} {picked:7d}")
    print(f"\n{'session':28} {'split':6} {'pressed ev':>10} {'not ev':>7} {'usable fr':>9}")
    for s in dict.fromkeys(r["session"] for r in rows):
        for split in ("train", "test"):
            sel = [r for r in rows if r["session"] == s and r["split"] == split and r["usable"]]
            if not sel:
                continue
            ev = lambda c: len({r["event_id"] for r in sel if r["label"] == c})
            print(f"{s:28} {split:6} {ev('pressed'):10d} {ev('not_pressed'):7d} {len(sel):9d}")
    reasons = Counter(r["drop_reason"] for r in rows if not r["usable"])
    print("\nDropped frames:", dict(reasons))
    all_press = {r["event_id"] for r in rows if r["label"] == "pressed"}
    used_press = {r["event_id"] for r in rows if r["label"] == "pressed" and r["usable"]}
    if len(used_press) < len(all_press):
        print(f"{len(all_press) - len(used_press)} of {len(all_press)} labelled press events "
              f"have no usable frame (no tag / poor tag fit / transition margin).")
    n_test_press = len({r["event_id"] for r in rows
                        if r["split"] == "test" and r["label"] == "pressed" and r["usable"]})
    if n_test_press < 20:
        print(f"Note: only {n_test_press} usable pressed events in the test split - "
              f"metrics will be noisy (aim for 20-30+).")


def main():
    sessions = [s for s in C.SESSIONS if common.labels_path(s).exists()]
    if not sessions:
        raise SystemExit(f"No label files in {C.LABELS_DIR}. Label with label_tool.py first.")
    all_rows = []
    for s in sessions:
        rows, points = build_session(s)
        assign_split(s, rows, points)
        all_rows += rows
        print(f"[{s}] {len(rows)} frames, {len(points)} change points")
    pick_training_frames(all_rows)

    C.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out = C.DATASET_DIR / "dataset.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    summarise(all_rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()