import os
import random
import requests
from PIL import Image
from io import BytesIO
import cv2 as cv
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_local_path

# URL with host
LS_URL =  "http://localhost:8080"
#LS_URL = "http://192.168.100.3:8080"
LS_API_TOKEN = "5018625469bbb652b072e6547b29fe7f85cced92"


# Initialize class inhereted from LabelStudioMLBase
class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        # Initialize self variables
        print("self.parsed_label_config ", self.parsed_label_config)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        self.labels = ['anus','make_love','nipple','penis','vagina']
        # Load model
        self.model = YOLO("best.pt")
        print(self.model)

    def _get_image_url(self, task):

        save_dir = "/Users/minhquan/label-studio-yolov8-backend/LBdata/media/upload/1"
        # save_dir = "/home/htdung167/Documents/BadmintonProject/BadmintonLabel/mydata/media"

        image_url = task['data'].get(self.value)
        image_url = image_url[1:].split("/", 1)[1]
        image_url = os.path.join(save_dir, image_url)
        return image_url

    # Function to predict
    def predict(self, tasks, **kwargs):
        """
        Returns the list of predictions based on input list of tasks for 1 image
        """
        print("@"*20)
        task = tasks[0]
        print(task)

        # Getting URL of the image
        image_url = self._get_image_url(task)
        # .task['data'][self.value]

        full_url = LS_URL + image_url
        print("FULL URL: ", full_url)

        # Header to get request
        # header = {
        #     "Authorization": "Token " + LS_API_TOKEN}
        
        # Getting URL and loading image
        # image = Image.open(BytesIO(requests.get(
        #     image_url).content))
        image = cv.imread(image_url)
        # # Height and width of image
        # original_width, original_height = image.size
        height, width, channels = image.shape
        # Creating list for predictions and variable for scores
        predictions = []
        score = 0
        

        # Getting prediction using model
        results = self.model.predict(image)
        print(results)
        

        # Getting mask segments, boxes from model prediction
        for result in results:
            for i, (box, segm) in enumerate(zip(result.boxes, result.masks.segments)):
                
                # 2D array with poligon points 
                rectangle_points = (segm * 100).tolist()

                # Adding dict to prediction
                predictions.append({
                    "from_name" : self.from_name,
                    "to_name" : self.to_name,
                    "id": str(i),
                    "type": "rectanglelabels",
                    "score": box.conf.item(),
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "points": rectangle_points,
                        "rectanglelabels": [self.labels[int(box.cls.item())]]
                    }})

                # Calculating score
                score += box.conf.item()


        print(10*"#", "Returned Prediction", 10*"#")

        # Dict with final dicts with predictions
        final_prediction = [{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": "v8x"
        }]

        return final_prediction
    #
    # def fit(self, completions, workdir=None, **kwargs):
    #     """
    #     Dummy function to train model
    #     """
    #     return {'random': random.randint(1, 10)}