import cv2
from cv2.typing import MatLike
import keyboard
import numpy as np
import supervision as sv
from ultralytics import YOLO

from database.tables import Room


class Camera:
    model = YOLO("yolov8n.pt", verbose=False)
    confidence_threshold = 0.5
    selected_classes = [0] # see https://stackoverflow.com/a/77479465
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def __init__(self, camera_id: int, name: str):
        self.camera_id = camera_id
        self.name = name
        self.capture = cv2.VideoCapture(camera_id)
        self._show_live = False

    def __str__(self):
        return f"Camera {self.camera_id}: {self.name}"

    def get_live_feed(self):
        """
        When the after Camera.start() has been called, this will return the frames of the camera
        """
        while self._show_live:
            if keyboard.is_pressed("q"):
                self.stop()

            ret, frame = self.get_image()
            if not ret:
                break
            yield frame

    def stop(self):
        """
        Stops the live feed
        """
        self._show_live = False

    def start(self):
        """
        Starts the live feed
        """
        self._show_live = True

    def get_image(self):
        return self.capture.read()

    def show_image(self, frame: MatLike | None = None):
        if frame is not None:
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)
            return
        ret, frame = self.get_image()
        cv2.imshow(self.name, frame)
        cv2.waitKey(1)

    def get_detections(self, frame):
        # Apply the model
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Apply Filters
        detections = detections[np.isin(detections.class_id, self.selected_classes)]
        detections = detections[detections.confidence > self.confidence_threshold]
        detections = detections[detections.class_id == self.selected_classes]
        detections: sv.Detections
        return detections

    def count_detections(self, frame):
        """
        Count the amount of humans inside the frame, and update database for this camera
        """
        Room.add_counter(room_id=self.camera_id, count=self.get_detections(frame))
        return len(self.get_detections(frame))

    @classmethod
    def annotate_frame(cls, frame: cv2.typing.MatLike, detections: sv.Detections) -> np.ndarray:
        labels = [
            cls.model.names[class_id]
            for class_id
            in detections.class_id
        ]

        labels_with_confidence = [
            f"{label}: {confidence:.2f}"
            for label, confidence
            in zip(labels, detections.confidence)
        ]

        annotated_image = cls.bounding_box_annotator.annotate(
            scene=frame,
            detections=detections,
        )
        annotated_image = cls.label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels_with_confidence,
        )

        return annotated_image


def main() -> None:
    YOLO("yolov8n.pt", verbose=False)

    c = Camera(0, "Room 0")
    c2 = Camera(1, "Room 1")
    c3 = Camera(2, "Room 2")

    c.start()
    c2.start()
    c3.start()
    for frame in c.get_live_feed():
        labeled_img = c.annotate_frame(frame, c.get_detections(frame))

        c.show_image(labeled_img)


if __name__ == "__main__":
    main()
