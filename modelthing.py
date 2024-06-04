from ultralytics import YOLO

# Generate the model

model = YOLO("yolov8n.pt")
model.export(format="openvino", imgsz=640)

# model.export(format="openvino", imgsz=640, half=True)
# model.export(format="openvino", imgsz=640, int8=True, data="coco128.yaml")


# Run the model.
openvino_model = YOLO("yolov8n_openvino_model/", task="detect")
openvino_model.predict(source=1, show=True, imgsz=640)
