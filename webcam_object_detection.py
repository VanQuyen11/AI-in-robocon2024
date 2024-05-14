import cv2
import time

from yolov8 import YOLOv8, draw_detections
from tracking import BYTETracker

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv7 object detector
model_path = "weights/yolov8n.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)
tracker = BYTETracker()

#cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    start_time = time.time()
    boxes, scores, class_ids = yolov8_detector(frame)
    print(f'{(time.time() - start_time)*1000:.2f} ms')
    
    boxes, scores, class_ids, ids = tracker.predict(frame, boxes, scores, class_ids)

    combined_img = draw_detections(frame, boxes, scores, class_ids, ids)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
