import pdb
import os
import cv2
import math
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
import whisper

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions


class VRSExtractor:
    
    def __init__(self, dataset, whisper_model_type="base", output_dir="outputs", data_dir="data"):
        self.dataset = dataset
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.whisper_model = whisper.load_model(whisper_model_type)
        self.save_dir = Path(os.path.join(output_dir, dataset, "vrs_extraction"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.vrs_data_provider = self.load_vrs_data_provider()
        return
    
    def extract_window(self, begin_str, end_str, frequency_hz, included=["camera", "gaze", "speech"]):
        begin_ns, end_ns = self.human_to_stream_time(begin_str), self.human_to_stream_time(end_str)
        step_ns = int(1e9 * 1.0 / frequency_hz)
        images_with_timestamps = self.extract_images(begin_ns, end_ns, step_ns) if "camera" in included else None
        eyegazes_with_images_with_timestamps = self.extract_eyegazes(images_with_timestamps) if "camera" in included and "gaze" in included else None
        speech = self.extract_speech(begin_ns, end_ns) if "speech" in included else None
        # return timestamp, images, gazes, speech (any of these will be None if their not in the 'included' list)
        if "camera" in included and "gaze" in included:
            return [elt[1] for elt in eyegazes_with_images_with_timestamps], [elt[0] for elt in eyegazes_with_images_with_timestamps], [elt[2] for elt in eyegazes_with_images_with_timestamps], speech
        elif "camera" in included:
            return [elt[1] for elt in images_with_timestamps], [elt[0] for elt in images_with_timestamps], None, speech
        elif "gaze" in included:
            return None, None, None, speech
        return None, None, None, None
    
    def load_vrs_data_provider(self):
        vrs_path = os.path.join(self.data_dir, self.dataset, "video.vrs")
        vrs_data_provider = data_provider.create_vrs_data_provider(vrs_path)
        return vrs_data_provider
    
    def human_to_stream_time(self, time_str):
        minutes, seconds = map(float, time_str.split(":"))
        offset_ns = int((minutes * 60 + seconds) * 1e9)
        start_ns = self.vrs_data_provider.get_first_time_ns_all_streams(TimeDomain.DEVICE_TIME)
        return start_ns + offset_ns
    
    def extract_images(self, begin_ns, end_ns, step_ns):
        image_save_dir = Path(os.path.join(self.save_dir, "images"))
        image_save_dir.mkdir(parents=True, exist_ok=True)
        stream_id = self.vrs_data_provider.get_stream_id_from_label("camera-rgb")
        images_with_timestamps = []     
        for idx, query_time_ns in enumerate(tqdm(range(begin_ns, end_ns, step_ns), desc="extracting images from vrs")):
            image_data, image_record = self.vrs_data_provider.get_image_data_by_time_ns(stream_id, query_time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
            image = cv2.cvtColor(image_data.to_numpy_array(), cv2.COLOR_RGB2BGR)
            output_path = image_save_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(output_path), image)
            images_with_timestamps.append([image, image_record.capture_timestamp_ns])
        print(f"Saved images to {image_save_dir}")
        return images_with_timestamps
        
    def extract_eyegazes(self, images_with_timestamps, time_diff_tol_ns=1e6, draw=True):
        eyegazes_with_images_with_timestamps = []
        stream_id = self.vrs_data_provider.get_stream_id_from_label("eyegaze")
        for idx, (image, image_timestamp) in enumerate(tqdm(images_with_timestamps, desc="extracting eye gazes from vrs")):
            found_valid_eyegaze = False
            eyegaze_timestamp = image_timestamp
            gaze_image_coords = [np.nan, np.nan]
            while not (found_valid_eyegaze or abs(eyegaze_timestamp - image_timestamp) > time_diff_tol_ns):
                sensor_data = self.vrs_data_provider.get_sensor_data_by_time_ns(stream_id, image_timestamp, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
                eyegaze_timestamp = sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)                
                gaze = sensor_data.eye_gaze_data()
                if gaze.combined_gaze_valid and gaze.spatial_gaze_point_valid:
                    found_valid_eyegaze = True
                    gaze_image_coords = self.project_gaze_to_rgb(gaze)
                    if draw:
                        self.draw_gaze(idx, image, gaze_image_coords)
            eyegazes_with_images_with_timestamps.append([image, image_timestamp, gaze_image_coords, eyegaze_timestamp])
        if draw:
            print(f"Saved gaze visualizations to {os.path.join(self.save_dir, 'gazes')}")
        return eyegazes_with_images_with_timestamps
            
    def project_gaze_to_rgb(self, gaze):
        device_calib = self.vrs_data_provider.get_device_calibration()
        camera_calib = device_calib.get_camera_calib("camera-rgb")
        T_device_cpf = device_calib.get_transform_device_cpf()
        gaze_point_device = (T_device_cpf @ gaze.spatial_gaze_point_in_cpf) # cpf to device
        gaze_point_camera = (camera_calib.get_transform_device_camera().inverse() @ gaze_point_device) # device to camera
        pixel = camera_calib.project(gaze_point_camera) # camera to pixel
        if pixel is None:
            return None
        return int(round(pixel[0])), int(round(pixel[1]))
    
    def draw_gaze(self, idx, image, gaze_coords, circle_radius=30, outline=8):
        drawing_save_dir = Path(os.path.join(self.save_dir, "gazes"))
        drawing_save_dir.mkdir(parents=True, exist_ok=True)
        x, y = gaze_coords
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), circle_radius, (0, 0, 255), outline)
        output_path = drawing_save_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(output_path), image)
        return
    
    def extract_speech(self, begin_ns, end_ns):
        stream_id = self.vrs_data_provider.get_stream_id_from_label("mic")
        timestamps_ns = self.vrs_data_provider.get_timestamps_ns(stream_id, TimeDomain.DEVICE_TIME)
        min_idx, max_idx = np.searchsorted(timestamps_ns, begin_ns), np.searchsorted(timestamps_ns, end_ns)
        audio_snippets = []
        for idx in tqdm(range(min_idx, max_idx), desc="extracting audio"):
            sensor_data = self.vrs_data_provider.get_sensor_data_by_index(stream_id, idx)
            audio_snippet = np.asarray(sensor_data.audio_data_and_record()[0].data)
            audio_snippets.append(audio_snippet)
        audio = np.concatenate(audio_snippets).reshape(-1, 8).astype(np.int16)
        audio_save_dir = Path(os.path.join(self.save_dir, "audio"))
        audio_save_dir.mkdir(parents=True, exist_ok=True)
        wavfile.write(os.path.join(audio_save_dir, "audio.wav"), 16000, audio[:, 0]) # NOTE: only keeping first of 8 channels (mono audio)
        speech = self.audio_to_text(os.path.join(audio_save_dir, "audio.wav"))
        Path(os.path.join(audio_save_dir, "transcription.txt")).write_text(speech)
        print(f"Saved audio and transcription to {audio_save_dir}")
        return speech

    def audio_to_text(self, audio_path):
        print("transcribing audio...")
        result = self.whisper_model.transcribe(str(audio_path))
        print("done")
        return result["text"].strip()