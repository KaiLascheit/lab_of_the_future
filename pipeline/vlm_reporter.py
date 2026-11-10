"""Prompt Gemini with a frame, the detected objects and the well-plate state, and
get back a structured update for the running experiment report."""
import json
import os
from string import Template

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

import settings as S
from experiment_state import format_time
from object_detector import objects_to_dicts


class ReportUpdate(BaseModel):
    summary: str               # 1-3 sentences: the experiment as understood so far
    current_activity: str      # what is happening in this frame
    new_steps: list[str]       # procedure steps that started since the last listed step
    new_materials: list[str]   # reagents / equipment not listed yet
    note: str                  # anything noteworthy, "" if nothing


class VLMReporter:
    def __init__(self, model=S.VLM_MODEL, prompt_file=S.VLM_PROMPT_FILE):
        load_dotenv()
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self.template = Template(prompt_file.read_text())

    def request_update(self, image, time_s, objects, plate, report):
        """One VLM call. Returns a ReportUpdate, or None if the call failed."""
        prompt = self.template.substitute(
            time=format_time(time_s),
            objects_json=json.dumps(objects_to_dicts(objects, image.shape), indent=1),
            plate_json=json.dumps(plate.to_dict(), indent=1),
            summary=report.summary or "(none yet)",
            materials=", ".join(report.materials) or "(none yet)",
            recent_steps=report.recent_steps_text(S.VLM_RECENT_STEPS),
        )
        contents = [types.Part.from_bytes(data=self._encode(image), mime_type="image/jpeg"), prompt]
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   response_schema=ReportUpdate))
            update = response.parsed
            if update is None:
                update = ReportUpdate.model_validate_json(response.text)
            return update
        except Exception as e:   # a failed call shouldn't stop a long recording
            print(f"WARNING: VLM update at {format_time(time_s)} failed: {e}")
            return None

    @staticmethod
    def _encode(image, max_side=S.VLM_IMAGE_MAX_SIDE):
        h, w = image.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()
