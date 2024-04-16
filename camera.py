import random
import cv2
import keyboard
import numpy as np
import schedule
import supervision as sv
from cv2.typing import MatLike
from ultralytics import YOLO
from ultralytics.solutions import object_counter

from database.tables import Camera as DbCamera


class Camera:
    model = YOLO("yolov8n.pt", verbose=False)
    confidence_threshold = 0.5
    selected_classes = [0] # see https://stackoverflow.com/a/77479465
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def __init__(self, camera_id: int, name: str, room_id: int, scheduler: schedule.Scheduler):
        self.camera_id = camera_id
        self.name = name
        self.room_id = room_id
        self._show_live = False
        self._auto_update_db = True
        self.capture = cv2.VideoCapture(camera_id)
        self.counter = object_counter.ObjectCounter()
        self.scheduler = scheduler
        self.video_writer = cv2.VideoWriter(
            "ml-video-output.avi",
            cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.CAP_PROP_FPS,
            (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.counter.set_args(
            view_img=False,
            view_in_counts=True,
            view_out_counts=True,
            reg_pts=[
                # (self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, 0),
                # (self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                (0, 240), (640, 240)
            ],
            classes_names=self.model.names,
            draw_tracks=True
        )

        self.scheduler.every(5).seconds.do(self.update_db_counter)
        print(f"{self.__dict__=}, {cv2.CAP_PROP_FRAME_HEIGHT=}, {cv2.CAP_PROP_FRAME_WIDTH=}")

    def __str__(self):
        return f"Camera {self.camera_id}: {self.name}"

    def get_live_feed(self):
        """
        When the after Camera.start() has been called, this will return the frames of the camera
        """
        while self._show_live:
            if keyboard.is_pressed("q"):
                self.stop()
                cv2.destroyWindow(self.name)
                return

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

    def track(self):
        _, im0 = self.get_image()
        tracks = self.model.track(im0, persist=True, show=False, classes=self.selected_classes)
        self.counter.start_counting(im0, tracks)
        self.video_writer.write(im0)
        return im0

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
        Count the amount of humans inside the frame
        """
        return len(self.get_detections(frame))

    def update_db_counter(self):
        """
        Update the database with the new count of humans in the room
        """
        DbCamera.update_counter(self)

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

    @property
    def total_count(self):
        # return random.randint(0, 100) # For testing purposes
        return self.counter.in_counts - self.counter.out_counts
