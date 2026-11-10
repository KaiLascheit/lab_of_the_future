"""Read RGB frames and the RGB camera calibration from an Aria VRS recording."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions


@dataclass
class Frame:
    idx: int             # running index of the extracted frame (0, 1, 2, ...)
    timestamp_ns: int    # capture time, device clock
    time_s: float        # seconds since the start of the recording
    image: np.ndarray    # BGR, raw sensor orientation (the orientation the calibration describes)


class VRSReader:
    def __init__(self, vrs_path, camera_label="camera-rgb"):
        vrs_path = Path(vrs_path)
        if not vrs_path.exists():
            raise FileNotFoundError(f"VRS file not found: {vrs_path}")
        self.path = vrs_path
        self.provider = data_provider.create_vrs_data_provider(str(vrs_path))
        if self.provider is None:
            raise RuntimeError(f"Could not open VRS file: {vrs_path}")
        self.stream_id = self.provider.get_stream_id_from_label(camera_label)
        if self.stream_id is None:
            raise RuntimeError(f"No '{camera_label}' stream in {vrs_path}")
        device_calib = self.provider.get_device_calibration()
        self.camera_calibration = device_calib.get_camera_calib(camera_label)
        if self.camera_calibration is None:
            raise RuntimeError(f"No calibration for '{camera_label}'. "
                               f"Available: {device_calib.get_camera_labels()}")
        self.start_ns = self.provider.get_first_time_ns_all_streams(TimeDomain.DEVICE_TIME)
        self.end_ns = self.provider.get_last_time_ns_all_streams(TimeDomain.DEVICE_TIME)

    def to_stream_time(self, time_str: Optional[str], default_ns: int) -> int:
        """"m:ss" (or plain seconds) from the start of the recording -> device time in ns."""
        if time_str is None:
            return default_ns
        parts = [float(p) for p in str(time_str).split(":")]
        seconds = parts[-1] + (60 * parts[-2] if len(parts) > 1 else 0)
        return self.start_ns + int(seconds * 1e9)

    def query_times(self, start=None, end=None, frequency_hz=5.0) -> range:
        begin_ns = self.to_stream_time(start, self.start_ns)
        end_ns = self.to_stream_time(end, self.end_ns)
        return range(begin_ns, end_ns, int(round(1e9 / frequency_hz)))

    def frames(self, start=None, end=None, frequency_hz=5.0,
               save_dir: Optional[Path] = None) -> Iterator[Frame]:
        """Frames closest to every 1/frequency_hz between start and end."""
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        idx, last_ts = 0, None
        for query_ns in self.query_times(start, end, frequency_hz):
            image_data, record = self.provider.get_image_data_by_time_ns(
                self.stream_id, query_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
            ts = int(record.capture_timestamp_ns)
            if ts == last_ts:          # same camera frame as before (gap in the stream)
                continue
            last_ts = ts
            array = image_data.to_numpy_array()
            if array is None or array.size == 0:
                continue
            image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            if save_dir is not None:
                cv2.imwrite(str(save_dir / f"frame_{idx:06d}.jpg"), image)
            yield Frame(idx, ts, (ts - self.start_ns) / 1e9, image)
            idx += 1
