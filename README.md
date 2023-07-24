# How to start:
This model is trained from Yolov8x model from Ultralytics

## Installation:
Clone into the git repository using command:

```bash
git clone https://github.com/minhquan24/Yolov8xNSFW.git
```
#Explain files:
**Dockerfile**: The Dockerfile for building the backend container.
**docker-compose.yml**: The docker-compose file for running the backend.
**model.py**: The Python code for the ML backend model.
**best.pt**: The pre-trained YOLOv8 model for n*de images detection.
**uwsgi.ini**: The uWSGI configuration file for running the backend.
**supervisord.conf**: The supervisord configuration file for running the backend processes.
**requirements.txt**: The list of Python dependencies for the backend.

# Download the requirements dependencies:

```bash
pip install -r requirments.txt
```

## Docker:
Please make sure that Docker-compose is downloaded and ready to use!!!

```bash
docker-compose build
docker-compose up
```

