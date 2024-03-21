from datetime import datetime, timedelta
import cv2
import keyboard
import numpy as np
import supervision as sv
from ultralytics import YOLO

from database.tables import Room, Session, engine

SHOW_VIDEO = True
DETECT_COOLDOWN_PERIOD = 0
camera_indexes = [0]

# set up some settings
model = YOLO("yolov8n.pt", verbose=False)
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
selected_classes = [0] # see https://stackoverflow.com/a/77479465
room_cooldown: dict[str, datetime] = {}
captures: dict[int, cv2.VideoCapture] = {}


def detect_room(target_number: int):
    """
    Detects the amount of ppl in a given room/camera, and updates the corresponding database table
    
    Parameters
    -----------
    target_number: int
        room or camera number to detect
    """

    if target_number in captures:
        cap = captures[target_number]
    else:
        cap = captures[target_number] = cv2.VideoCapture(target_number)

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Apply the model
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Apply Filters
    detections = detections[np.isin(detections.class_id, selected_classes)]
    detections = detections[detections.confidence > 0.7]
    detections = detections[detections.class_id == selected_classes]
    detected_amount = len(detections)

    if detected_amount > 0:
        # Human detected
        with Session(engine) as session:
            if room := session.query(Room).where(Room.id == target_number).first():
                room.human_count = detected_amount
            else:
                session.add(Room(id=target_number, human_count=detected_amount))
            session.commit()

    # Annotate the image
    if SHOW_VIDEO:
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

        annotated_image = bounding_box_annotator.annotate(
            scene=frame,
            detections=detections,
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels_with_confidence,
        )

        # Display the resulting frame
        cv2.imshow('IT Hub Human Reconition Frame', annotated_image)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
    # cap.release()


def reset_cooldown():
    """
    Reset the cooldown variable to 0 in datetime
    """
    for target in camera_indexes:
        room_cooldown[target] = datetime.fromtimestamp(0)


def on_cooldown(target: int):
    return room_cooldown[target] + timedelta(seconds=DETECT_COOLDOWN_PERIOD) >= datetime.now()


def set_cooldown(target: int):
    room_cooldown[target] = datetime.now()


def main() -> None:
    reset_cooldown()
    run = True

    while run:
        if keyboard.is_pressed('q'):
            run = False
            break

        for target in camera_indexes:
            if on_cooldown(target):
                continue
            detect_room(target)
            set_cooldown(target)

    # When everything done, release the capture
    for cap in captures.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
