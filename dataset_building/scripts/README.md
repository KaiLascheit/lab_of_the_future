# Pipette press dataset tools

Classify whether the pipette plunger is pressed, from upright crops around the plunger
that are cut out using the pose of the pipette AprilTag.

```
dataset_building/
  scripts/   these tools (settings in config.py, plus a SETTINGS block in each script)
  data/      everything they produce
```

## Setup

```bash
pip install opencv-contrib-python numpy tqdm projectaria-tools \
            pandas scikit-learn matplotlib joblib torch torchvision
```

Edit `config.py`:
- `SESSIONS`: one entry per recording: the VRS path (relative to the folder containing
  `dataset_building/`), an optional frame range, and `marker_id` if a recording uses a different
  pipette tag (e.g. 30 for the single-channel P1000 recording).
- `PIPETTE_MARKER_ID`, `TAG_SIZE_M`, `BOX_X`, `BOX_Y`: as verified with `check_pipette_marker.py`
  (tag 43, 2 cm, box 10 x 15 cm with "up the pipette" = -Y).

## Workflow

| step | command | output |
|---|---|---|
| 1. crops | `python extract_crops.py [session ...]` | `data/crops/<session>/` + `index.csv` |
| 2. overview | `python make_contact_sheets.py [session ...]` | `data/contact_sheets/<session>/` |
| 3. label | `python label_tool.py <session>` | `data/labels/<session>.csv` |
| 4. dataset | `python build_dataset.py` | `data/dataset/dataset.csv` |
| 5. train/eval | `python train_eval.py [--backbone dinov2_vits14] [--curve]` | `data/results/<backbone>/` |

### 1. Crops
Reads frames straight from the VRS, detects the pipette tag, estimates its pose with the
Aria fisheye calibration and resamples the plunger region into an upright 300 x 450 px crop
(3 px/mm). `index.csv` lists every frame in range, including frames without the tag, with
tag quality (`reproj_px`, `ambiguity` = how well the flipped orientation fits; > 0.5 means
the pose may flip, `dist_m`, `view_deg`). Existing crops are skipped unless `--overwrite`.

### 2. Contact sheets
Numbered thumbnail grids (frame index under each crop, grey tiles where the tag was missing,
a coloured bar for the current label once labels exist). Good for finding presses quickly,
and for checking labels after labelling.

### 3. Labelling
Labels are **change points**: pressing a key at a frame means "from here on, the label is X".
For each press, mark its first frame with `p` and the first frame after it with `n`.
Use `i` (ignore) for stretches you can't judge. Everything before the first mark is unlabelled.
Saved after every change; re-open any time to continue or fix.

| key | action |
|---|---|
| right / `d`, left / `a` | next / previous frame |
| up / `w`, down / `s` | +10 / -10 frames |
| `]` / `[` | next / previous change point |
| space | play / pause |
| `p` / `n` / `i` | pressed / not pressed / ignore from this frame on |
| `x` | delete the change point at this frame |
| `q` / esc | quit |

### 4. Dataset
`build_dataset.py` (settings at the top):
- **events** = runs of identical pressed / not_pressed labels; all evaluation and splitting
  works on events, because neighbouring frames are near-duplicates
- drops `TRANSITION_MARGIN` frames either side of every label change, frames without the tag,
  and frames with poor tag poses (`MAX_REPROJ_PX`, `MAX_AMBIGUITY`)
- **split**: whole recordings in `TEST_SESSIONS` (preferred), otherwise the last
  `TEST_FRACTION` of each recording, cut at a label change so no event is split
- **training frames**: `TRAIN_FRAMES_PER_EVENT` evenly spaced frames per event; test keeps all
- prints events and frames per split and class, and warns if the test set has < 20 presses

### 5. Training and evaluation
Frozen pretrained features (default DINOv2-small, or `resnet18`, `resnet50`, or `pixels` as a
no-deep-learning baseline) + standardised, class-balanced logistic regression. The regularisation
strength is chosen by cross-validation grouped by event. Features are cached in `data/features/`.

Reported on the test split:
- **frame level** (usable frames): balanced accuracy, precision / recall / F1 for "pressed",
  confusion matrix, ROC AUC; raw and after a median filter over `SMOOTH_WINDOW` frames
- **event level**: labelled presses found, predicted presses (runs >= `MIN_EVENT_FRAMES`),
  event recall / precision, median onset error in frames
- `timeline_<session>.png`: labels vs. predicted probability over time - the quickest way to
  see what goes wrong
- `--curve`: learning curve over the number of training events per class (repeated random
  subsets). If it is still rising at the largest size, label more; if it is flat, improve the
  crops or the model instead.

`model.joblib` holds the fitted classifier (plus backbone name and input size) for later use.

## How much to label
Count **events**, not frames. A reasonable first target: 30-50 press events for training,
20-30 for testing from a separate recording, with hard negatives (thumb resting on the plunger
without pressing, fast pipette movement). Then use `--curve` to decide whether more helps.
