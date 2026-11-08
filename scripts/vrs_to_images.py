"""
Extract images from a VRS file (e.g. a Project Aria recording) and save them as PNGs.

Install:  pip install projectaria-tools opencv-python
Usage:    python extract_vrs_images.py recording.vrs [output_folder]
"""
import os
import sys
import cv2
from tqdm import tqdm
from pathlib import Path
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import SensorDataType

vrs_path = sys.argv[1] if len(sys.argv) > 1 else "recording.vrs"
out_dir = sys.argv[2] if len(sys.argv) > 2 else Path("frames") / Path(vrs_path).stem

# 1. Open the VRS file
provider = data_provider.create_vrs_data_provider(vrs_path)
if provider is None:
    sys.exit(f"Could not open {vrs_path}")

# 2. Go through every stream and keep only the image streams
for stream_id in provider.get_all_streams():
    if provider.get_sensor_data_type(stream_id) != SensorDataType.IMAGE:
        continue

    label = provider.get_label_from_stream_id(stream_id)  # e.g. "camera-rgb"
    if label != "camera-rgb":
        continue
    
    num_frames = provider.get_num_data(stream_id)
    stream_dir = out_dir # os.path.join(out_dir, label)
    os.makedirs(stream_dir, exist_ok=True)
    print(f"{label} ({stream_id}): {num_frames} frames")

    # 3. Read each frame and save it
    for i in tqdm(range(num_frames)):
        image_data, record = provider.get_image_data_by_index(stream_id, i)
        img = image_data.to_numpy_array()          # numpy array, RGB or grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR

        timestamp_ns = record.capture_timestamp_ns
        cv2.imwrite(os.path.join(stream_dir, f"{i:06d}_{timestamp_ns}.png"), img)

print(f"Done. Images saved under '{out_dir}/'")