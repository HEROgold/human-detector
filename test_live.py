import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import time
import sqlite3

# Initialize the model, don't output processing speed
model = YOLO("yolov8n.pt", verbose=False)

# Initialize the annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Start the webcam
cap = cv2.VideoCapture(0)

selected_classes = [0]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Apply the model
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Apply Filters
    detections = detections[np.isin(detections.class_id, selected_classes)]
    detections = detections[detections.confidence > 0.5]
    detections = detections[detections.class_id == selected_classes]
    print(len(detections))

    labels = [
        model.model.names[class_id]
        for class_id
        in detections.class_id
    ]

    labels_with_confidence = [
        f"{label}: {confidence:.2f}"
        for label, confidence
        in zip(labels, detections.confidence)
    ]

    if labels_with_confidence:
        print("HUMAN FOUND")

    # Annotate the image
    annotated_image = bounding_box_annotator.annotate(
        scene=frame,
        detections=detections,
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels_with_confidence,
    )

    time.sleep(1)
    # Display the resulting frame
    cv2.imshow('Human reconition', annotated_image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
