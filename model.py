import os
import random
import requests
from PIL import Image
from io import BytesIO

from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_local_path
import logging
# URL with host
# LS_URL =  "http://127.0.0.1:8080"
LS_URL = "http://label_studio_ui:8080"
LS_API_TOKEN = "7150236b0895d2fdb6370d90c517019a7b3df0a1"


# Initialize class inhereted from LabelStudioMLBase
class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        # Initialize self variables
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        self.labels = ['anus', 'make_love','nipple','penis','vagina']
        # Load model
        self.model = YOLO(("/app/inferrence_model/yolov8s_v3/openvino_fp16/best_openvino_model"),task="detect")

    # Function to predict
    def predict(self, tasks, **kwargs):
        """ This is where inference happens: model returns
                    the list of predictions based on input list of tasks
                """
        task = tasks[0]

        predictions = []
        score = 0

        # header = {
        #     "Authorization": "Token " + LS_API_TOKEN}
        # print(str(LS_URL + task['data']['image']), flush=True)
        # image_url=LS_URL + task['data']['image']
        # image_bytes= requests.get(image_url, headers = header).content
        # image = Image.open(BytesIO(image_bytes))
        image_uri="/app/files/"+task['data']['image'][21:]
        print(image_uri, flush=True)
        image = Image.open(image_uri)
        original_width, original_height = image.size
        results = self.model.predict(image)

        for result in results:
            for i, prediction in enumerate(result.boxes):
                xyxy = prediction.xyxy[0].tolist()
                predictions.append({
                    "id": str(i),
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "rectanglelabels",
                    "score": prediction.conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": xyxy[0] / original_width * 100,
                        "y": xyxy[1] / original_height * 100,
                        "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                        "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                        "rectanglelabels": [self.labels[int(prediction.cls.item())]]
                    }
                })
                score += prediction.conf.item()

        return [{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": "v8n",  # all predictions will be differentiated by model version
        }]
