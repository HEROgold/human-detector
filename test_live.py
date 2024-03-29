from datetime import datetime
import cv2
import keyboard
import numpy as np
import supervision as sv
from ultralytics import YOLO

from database.tables import Room

SHOW_VIDEO = True
TRACKER = True
DETECT_COOLDOWN_PERIOD = 0
videos_or_cameras = [0]

# set up some settings
model = YOLO("yolov8n.pt", verbose=False)
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
tracker = cv2.TrackerMIL.create()
selected_classes = [0] # see https://stackoverflow.com/a/77479465

# Empty things for storage
room_cooldown: dict[str, datetime] = {}
captures: dict[int, cv2.VideoCapture] = {}
tracker_initialized = False


def detect_room(target_number: int | str):
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
    # detections = detections[detections.confidence > 0.7]
    detections = detections[detections.class_id == selected_classes]
    detections: sv.Detections
    detected_amount = len(detections)

    if detected_amount > 0:
        # Human detected
        Room.add_counter(room_id=target_number, count=detected_amount)

        annotated_image = annotate_frame(frame, detections)
        track_detections(frame, detections)
        draw_middle_line(frame, annotated_image)

        # Display the resulting frame
        cv2.imshow('IT Hub Human Recognition Frame', annotated_image)
        cv2.waitKey(1)


def draw_middle_line(frame: cv2.typing.MatLike, annotated_image: np.ndarray):
    height, width, _ = frame.shape
    middle_y = height // 2
    cv2.line(annotated_image, (0, middle_y), (width, middle_y), (255, 0, 0), 5)


def annotate_frame(frame: cv2.typing.MatLike, detections: sv.Detections) -> np.ndarray:
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

        return annotated_image


def track_detections(frame: cv2.typing.MatLike, detections: sv.Detections):
    global tracker_initialized
    tracker_initialized = tracker_initialized or False

    if TRACKER:
        for i, bbox in enumerate(detections.xyxy.tolist()):
            if i > 0:
                break

            # roi = cv2.selectROI(frame, True)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            w, h = x2 - x1, y2 - y1

            # Validate the bounding box
            if (
                x1 < 0 
                or y1 < 0 
                or w <= 0 
                or h <= 0
                or x2 <= 0
                or y2 <= 0
            ):
                continue

            if tracker_initialized is False:
                # Initialize tracker with first frame and bounding box
                track = tracker.init(frame, [x1, y1, w, h])
                tracker_initialized = True
            else:
                track = tracker.update(frame)

            if track:
                rect = track[1]
                cv2.putText(
                    frame,
                    f"tracking {i}",
                    (rect[0], rect[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,(0,0,255),
                    2
                )
                cv2.rectangle(
                    frame,
                    (rect[0], rect[1]),
                    (rect[2], rect[3]), # Add +10 to either x2 or y2 to visualize the tracking box
                    (255,125,0),
                    2,
                    1
                )
                # print("Tracking success")
            else:
                # Tracking failure
                # cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                print("Tracking failure")


def main() -> None:
    run = True

    while run:
        if keyboard.is_pressed('q'):
            run = False
            break

        for target in videos_or_cameras:
            detect_room(target)

    # When everything done, release the capture
    for cap in captures.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
