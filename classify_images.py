import cv2 as cv
from ultralytics import YOLO
import math

from utilities import max_screen

# This is the index of the used camera it can be 0, 1, ..
cam_index = 0
def classify_images():
    video_capture = cv.VideoCapture(cam_index)
    # model
    model = YOLO("yolo-Weights/yolov8n.pt")
    # object classes
    class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                   "teddy bear", "hair drier", "toothbrush"
                   ]

    window_name = "AILabo"
    max_screen(window_name, video_capture)
    while True:
        ret, frame = video_capture.read()
        results = model(frame, stream=True)
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # put box in cam
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", class_names[cls])

                # object details
                org = [x1, y1]
                font = cv.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (0, 0, 0)
                thickness = 2

                cv.putText(frame, class_names[cls], org, font, fontScale, color, thickness)

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv.imshow(window_name, frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    video_capture.release()
    cv.destroyAllWindows()
