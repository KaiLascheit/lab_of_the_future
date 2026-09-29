from pathlib import Path
import os

from vrs_extraction import VRSExtractor
from vlm_prompting import VLM


DATASET = "eat_2"

vrs_extractor = VRSExtractor(DATASET)
vlm = VLM()

timestamps, images, gazes, speech = vrs_extractor.extract_window("3:20", "3:30", frequency_hz=1.0)
vlm_description = vlm.describe_scene(images, gazes, speech)

print(vlm_description)
Path(os.path.join("outputs", DATASET, "scene_description.md")).write_text(vlm_description)