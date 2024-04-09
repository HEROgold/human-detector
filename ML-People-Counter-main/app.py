from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt", verbose=False)
# model.to("cuda")

# MP4 Capture
cap = cv2.VideoCapture("assets/video.mp4")

# Camera Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define line points
# TopLeft > BottomRight
line_points = [(320, 720), (360, 0)]

# Video writer
video_writer = cv2.VideoWriter(
    "ml-video-output.avi",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h)
)

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(
    view_img=True,
    view_in_counts=True,
    view_out_counts=True,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=True
)

classes_to_count = [0]

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)
    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)
    # print fps
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Count: {counter.in_counts - counter.out_counts}")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
