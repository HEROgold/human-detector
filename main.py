from functools import partial
import cv2
import tkinter as tk
from PIL import Image, ImageTk

from camera import Camera


class CameraSelector(tk.Tk):
    _max_cam_count = 5

    def __init__(self):
        super().__init__()
        self.title("Camera Selector")
        self.buttons: list[tk.Button] = []
        self.cameras: list[Camera] = []
        self._selected_camera = 0

        count = self.get_max_cam_count()
        for i in range(count):  # Adjust range as needed
            cam = Camera(camera_id=i, name=f"Camera {i}")
            cap = cam.capture

            self.cameras.append(cam)

            if not cap.isOpened():
                continue

            ret, frame = cap.read()
            if not ret:
                continue

            # Convert frame to PIL Image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            # Create a button with the frame as its image
            imgtk = ImageTk.PhotoImage(image=img)
            btn = tk.Button(self, image=imgtk, command=partial(self.button_click, i))
            btn.image = imgtk  # Keep a reference to prevent GarbageCollection
            btn.pack(side="left")
            self.buttons.append(btn)

    def button_click(self, index):
        self.set_active_camera(index)
        print(f"Selected camera {self.get_active_camera()}")
        self.show_selected_camera(self.cameras[self.get_active_camera()])


    def get_active_camera(self):
        return self._selected_camera

    def set_active_camera(self, index):
        self._selected_camera = index

    def get_camera_count(self):
        i = 0
        max_idx = self._max_cam_count
        while True:
            if i >= max_idx:
                break

            cam = cv2.VideoCapture(i)
            if not cam.isOpened():
                break

            ret, frame = cam.read()

            if not ret:
                break
            i += 1

        print(f"Found {i} cameras")
        return i

    def get_max_cam_count(self):
        return self._max_cam_count

    @staticmethod
    def show_selected_camera(cam: Camera):
        cam.start()
        cam.count_detections()
        for frame in cam.get_live_feed():
            detections = cam.get_detections(frame)
            annotated_frame = cam.annotate_frame(frame, detections)
            cam.show_image(annotated_frame)


def main() -> None:
    app = CameraSelector()
    app.mainloop()


if __name__ == "__main__":
    main()
