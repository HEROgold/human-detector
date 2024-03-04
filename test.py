import os
import cv2
import supervision as sv
from ultralytics import YOLO


model = YOLO("yolov8n.pt")
image = cv2.imread(os.path.abspath("./humans.png"))
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    model.model.names[class_id]
    for class_id
    in detections.class_id
]

annotated_image = bounding_box_annotator.annotate(
    scene=image,
    detections=detections,
)
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels,
)
cv2.imwrite(os.path.abspath("./humans_annotated.png"), annotated_image)
