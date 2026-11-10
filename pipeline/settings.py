"""
All pipeline settings in one place: paths, rates, marker/plate geometry, model
choices and thresholds. The input recording and time window can also be set from
the command line (see `python main.py --help`).

Paths are relative to the project root (the folder that contains pipeline/), so
the pipeline finds streams/, dataset_building/, volume_templates/ and
rigid_body_calibration.json wherever you launch it from.

NOTE: this file is deliberately not called config.py - the press-classifier code
in PRESS_PROJECT_DIR has its own config.py, and two modules with the same name
would shadow each other.
"""
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PIPELINE_DIR.parent

# ============================================================================ input / output
STREAM = "Well_plate_p1000_2QR_2_20261107_162219"
VRS_PATH = PROJECT_DIR / "streams" / f"{STREAM}.vrs"
OUTPUT_ROOT = PROJECT_DIR / "outputs"   # results go to OUTPUT_ROOT/<vrs file name>/
WINDOW_START = None        # "m:ss" from the start of the recording; None = beginning
WINDOW_END = None          # "m:ss"; None = end of the recording
SAVE_FRAMES = True         # raw frames -> <out>/frames/ (input for volume_reader.py --collect)

# ============================================================================ visualisation
# Saved to <out>/VIS_DIR_NAME/ (see visualisation.py)
VIS_DIR_NAME = "visus"
VIS_TRACKING = True        # tracking/: annotated camera frames, every frame (5 Hz)
VIS_ABSTRACT = True        # abstract/: white-on-black plate/ray view + well map, every frame
VIS_OBJECTS = True         # objects/: detector boxes + SAM masks, VLM frames only (1 Hz)
VIS_VIDEO = True           # tracking_stacked.mp4 (tracking above abstract) at the end of the run
VIDEO_SLOWDOWN = 1         # play this many times slower than real time
VIDEO_SCALE = 0.5          # resize each panel
VIDEO_INTERP_STEPS = 0     # cross-faded frames between frames (smoother 5 Hz playback)

# ============================================================================ frame extraction
CAMERA_LABEL = "camera-rgb"
FRAME_RATE_HZ = 5.0
VLM_EVERY_N_FRAMES = 5     # last frame of every group of N goes to the VLM (5 Hz / 5 = 1 Hz)

# ============================================================================ object detection
# SAM itself segments but doesn't name things, so objects are found by an
# open-vocabulary detector (YOLO-World, prompted with OBJECT_LABELS) and MobileSAM
# turns each box into a mask (and a tight box from that mask).
RUN_OBJECT_DETECTION = True
OBJECT_LABELS = [
    "pipette", "well plate", "pipette tip box", "test tube", "tube rack",
    "reagent reservoir", "bottle", "beaker", "glove", "hand",
]
PIPETTE_LABEL = "pipette"
# AprilTag tracking (and the volume reading) only run on frames where PIPETTE_LABEL
# was detected. Off for now: tracking runs on every frame.
REQUIRE_PIPETTE_FOR_TRACKING = False
DETECTOR_WEIGHTS = "yolov8s-worldv2.pt"   # downloaded by ultralytics on first use
SAM_WEIGHTS = "mobile_sam.pt"             # downloaded by ultralytics on first use
DETECTION_CONF = 0.15
USE_SAM_MASKS = True       # False: detector boxes only (faster)
DEVICE = None              # "cuda", "cuda:0", "cpu", "mps" or None (ultralytics picks)

# ============================================================================ AprilTags
# Black-border side length of each tag in METRES (measure your printed tags!).
TAG_SIZE_M = {
    30: 0.020, 23: 0.020,     # pipette
    26: 0.020, 36: 0.020,     # 96-well plate
}
MARKER_TO_OBJECT = {30: "pipette", 23: "pipette", 26: "plate", 36: "plate"}

# Rigid bodies tracked with several markers. The body frame is the frame of "ref";
# the offsets of the other markers are loaded from BODY_CALIB_FILE (written by
# detect_apriltags.py or by this pipeline). If missing, they are estimated online
# once CALIB_WARMUP_FRAMES frames with all tags visible have been seen; until then
# the body pose comes from the ref marker alone.
RIGID_BODIES = {
    "pipette": {"markers": [30, 23], "ref": 30},
}
BODY_CALIB_FILE = PROJECT_DIR / "rigid_body_calibration.json"
RECALIBRATE_BODIES = False     # True: ignore BODY_CALIB_FILE and re-estimate online
CALIB_WARMUP_FRAMES = 30       # co-visible frames to collect before calibrating online
CALIB_MAX_BUFFER = 1000        # frames kept for online calibration
CALIB_MAX_REPROJ_PX = 1.5
CALIB_MIN_FRAMES = 10
CALIB_MIN_FRAMES_HARD = 3
CALIB_REFINE_BA = True
CALIB_INLIER_DEG = 10.0
CALIB_INLIER_M = 0.010

# 96-well plate box from its two markers. Seen from above, PLATE_TOP_ID and
# PLATE_BOTTOM_ID sit at the upper and lower right corners, and the plate body
# extends to the LEFT of the line between them.
PLATE_TOP_ID = 36
PLATE_BOTTOM_ID = 26
PLATE_EDGE_EXTENSION_M = 0.015
PLATE_LENGTH_M = 0.12776
PLATE_RIGHT_MARGIN_M = 0.010
PLATE_HEIGHT_M = 0.0144
GRID_ROWS, GRID_COLS = 8, 12

# Well names. Grid row 0 is at PLATE_TOP_ID's end of the marker edge, grid column 0
# along the marker edge itself (the plate's right side seen from above).
ROW_A_AT_TOP_MARKER = True        # True: grid row 0 is row "A"
COLUMN_1_AT_MARKER_EDGE = False   # False: grid column 0 is column 12 (A1 top-left)

# Ray cast from the pipette body towards the plate
RAY_BODY = "pipette"
RAY_DIRECTION = "marker_line"     # "marker_line" (ref marker -> RAY_LINE_TO_ID) or "axis"
RAY_LINE_TO_ID = 23
RAY_AXIS = "y"                    # used for "axis", and as fallback before calibration
RAY_AXIS_SIGN = +1
HIT_DIST_THRESHOLD_M = 0.30       # hit cell only counts closer than this

# ============================================================================ press classifier
CLASSIFY_PRESS = True
PRESS_PROJECT_DIR = PROJECT_DIR / "dataset_building" / "scripts"  # common.py / config.py / train_eval.py
CLASSIFIER_RUN = "dinov2_vits14"
CLASSIFIER_PATH = None            # None: <press config.RESULTS_DIR>/<CLASSIFIER_RUN>/model.joblib
CROP_MARKER_ID = 30
PRESS_CROP_SESSION = "p1000_162006"
CROP_POSE = "fused"               # "fused" (body pose) or "single" (CROP_MARKER_ID alone)
PRESS_THRESHOLD = None            # None: threshold saved with the model
# Online press detection at FRAME_RATE_HZ: trailing median over the last
# PRESS_SMOOTH_WINDOW classified frames, then one dispense per press. A new press
# can only be counted after PRESS_REARM_FRAMES frames without a press.
PRESS_SMOOTH_WINDOW = 3
PRESS_REARM_FRAMES = 2

# ============================================================================ volume counter
VOLUME_N_DIGITS = 3
VOLUME_UL_PER_UNIT = 10.0         # Gilson-style P1000: "025" = 250 µL
VOLUME_POSITION_DIGITS = ("01", None, None)
VOLUME_TEMPLATES_DIR = PROJECT_DIR / "volume_templates"
VOLUME_HISTORY = 25               # confident readings considered (25 = 5 s at 5 Hz)
VOLUME_MIN_SUPPORT = 2            # readings needed before a value counts as stable

# ============================================================================ VLM report
RUN_VLM = True
VLM_MODEL = "gemini-3.5-flash-lite"
VLM_PROMPT_FILE = PIPELINE_DIR / "prompts" / "report_update.txt"
VLM_IMAGE_MAX_SIDE = 768          # frames are downscaled (aspect kept) before sending
VLM_RECENT_STEPS = 10             # procedure steps shown to the VLM as context