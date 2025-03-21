from roboflow import Roboflow
from ultralytics import YOLO


rf = Roboflow(api_key="56I24vFGcTeGPDzhYA3N")
project = rf.workspace("rofand-aaqbf").project("newdataset-qeiw4")
version = project.version(3)
dataset = version.download("yolov8")

# Load the model
model = YOLO("yolov8s.pt")

# Train the model
model.train(data=f"{dataset.location}/data.yaml", epochs=25, imgsz=800,batch=8, plots=True)