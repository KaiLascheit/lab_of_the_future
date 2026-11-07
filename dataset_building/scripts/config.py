"""
Shared settings for the pipette-press dataset tools.

Layout (relative to this file):
  dataset_building/scripts/   <- these tools
  dataset_building/data/      <- everything they produce
      crops/<session>/<frame>.png      upright 10 x 15 cm crops around the plunger
      crops/<session>/index.csv        one row per frame (incl. frames without the tag)
      contact_sheets/<session>/...     numbered thumbnail grids
      labels/<session>.csv             label change points from label_tool.py
      dataset/dataset.csv              per-frame labels, events, splits, usage flags
      features/<backbone>.npz          cached image features
      results/<run_name>/...           metrics, predictions, timeline plots
"""
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
PROJECT_ROOT = SCRIPTS_DIR.parent.parent      # folder that contains dataset_building/

CROPS_DIR = DATA_DIR / "crops"
SHEETS_DIR = DATA_DIR / "contact_sheets"
LABELS_DIR = DATA_DIR / "labels"
DATASET_DIR = DATA_DIR / "dataset"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = DATA_DIR / "results"

# ----------------------------------------------------------------------------- sessions
# One entry per recording. Paths are relative to PROJECT_ROOT (or absolute).
# start/end are RGB frame indices (end exclusive, None = to the end); every frame
# index used anywhere in these tools is the RGB frame index in that VRS.
SESSIONS = {
    "multi_p300_154736": {
        "vrs": "streams/96-well_plate_pipetting_multi_p300_QR_20261105_154736.vrs",
        "start": 0,
        "end": None,
    },
    # "p1000_154354": {
    #     "vrs": "streams/96-well_plate_pipetting_p1000_QR_20261105_154354.vrs",
    #     "start": 0, "end": None,
    #     "marker_id": 30,          # per-session override of PIPETTE_MARKER_ID
    # },
}

# ----------------------------------------------------------------------------- camera / marker
CAMERA_LABEL = "camera-rgb"
PIPETTE_MARKER_ID = 43            # default; override per session with "marker_id"
TAG_SIZE_M = 0.020                # black-border side length (check with check_pipette_marker.py)

# Crop box in the pipette marker's frame (metres). The tag is mounted upside down:
# "up the pipette" is -Y, the thumb side (image right) is -X.
BOX_X = (-0.06, 0.04)             # 10 cm wide
BOX_Y = (-0.14, 0.01)             # 15 cm tall
CROP_PLANE_Z = 0.0
UP_SIGN = -1                      # crops are rendered with "up the pipette" at the top
CROP_PX_PER_MM = 3                # 10 x 15 cm -> 300 x 450 px
CROP_GRID_STEP_MM = 5             # exact projection every 5 mm, interpolated in between

# ----------------------------------------------------------------------------- labels
LABELS = {"p": "pressed", "n": "not_pressed", "i": "ignore"}   # label_tool key -> label
UNLABELLED = "unlabelled"
CLASSES = ["not_pressed", "pressed"]                            # class 0, class 1
