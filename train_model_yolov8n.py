//
from roboflow import Roboflow
import os
from ultralytics import YOLO

#truy cập dataset ở trang Roboflow
rf = Roboflow(api_key="hyWj56r4YyxOfvHC1BIZ")
project = rf.workspace("vietnameselicenseplate").project("vietnamese-license-plate-tptd0-y4gwu")
dataset = project.version(1).download("yolov8")
#Mô hình loyov8n
model = YOLO("yolov8n.pt")

#Train
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device='cuda'
)

