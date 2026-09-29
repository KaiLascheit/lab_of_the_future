import pdb
import os
import cv2
from pathlib import Path

from google.genai import types

from dotenv import load_dotenv
from google import genai


class VLM:
    
    def __init__(self):
        load_dotenv()
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.prompt_dir = "prompts"
        return

    def describe_scene(self, images, gazes, speech):
        images, gazes = self.downscale_media(images, gazes)
        message = []
        for idx, (image, gaze) in enumerate(zip(images, gazes)):
            message.append(f"Frame {idx}, gaze position: ({gaze[0]}, {gaze[1]}) pixels.")
            _, buffer = cv2.imencode(".jpg", image)
            message.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"))
        prompt = Path(os.path.join(self.prompt_dir, "scene_description.txt")).read_text().format(transcribed_speech=speech)
        message.append(prompt)
        print("Prompting VLM")
        response = self.client.models.generate_content(model="gemini-3.5-flash-lite", contents=message)
        return response.text
    
    def downscale_media(self, images, gazes, target_size=(480, 640)):
        h, w = images[0].shape[:2]
        target_h, target_w = target_size
        images = [cv2.resize(image, (target_w, target_h)) for image in images]
        gazes = [(int(x * target_w / w), int(y * target_h / h)) for x, y in gazes]
        return images, gazes