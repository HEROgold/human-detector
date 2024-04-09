# Machine Learning People Counter

This repository contains a Python script for object counting using the YOLO (You Only Look Once) object detection model implemented in the `ultralytics` library along with OpenCV.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- [Python](https://www.python.org/downloads/) (>= 3.6)
- [OpenCV](https://opencv.org/) (>= 4.5.1)
- [Ultralytics](https://github.com/ultralytics/yolov5) (>= 0.0.7)

## Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/PandaMcGame/ML-People-Counter.git
    cd ml-people-counter
    ```

2. Install the required Python dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
   
3. Download the YOLOv8n pre-trained weights file (`yolov8n.pt`) from [here](https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov8n.pt) and place it in the root directory of the repository.

## Usage

Run the following command to execute the object counting script:

```bash
python app.py
```

## Script Description

The provided Python script (`app.py`) performs object counting on either a video file (`assets/video.mp4`) or live camera feed. Here's a brief overview of the script:

1. Imports necessary libraries: `YOLO` from `ultralytics`, `object_counter` from `ultralytics.solutions`, and `cv2` from OpenCV.
2. Initializes the YOLO model with the pre-trained weights (`yolov8n.pt`).
3. Sets up the video capture from either a video file or camera.
4. Defines line points for counting objects.
5. Initializes the video writer to save the output.
6. Initializes the object counter with specified arguments.
7. Starts the main loop for processing video frames:
    - Reads frames from the video.
    - Detects and tracks objects using the YOLO model.
    - Performs object counting and draws bounding boxes and tracks on the frames.
    - Writes the annotated frames to the output video file.
8. Releases video resources and closes windows.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to modify and use this script according to your requirements! If you have any questions or suggestions, please open an issue or contact the repository owner.