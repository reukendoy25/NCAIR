import cv2
import torch
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Queue to store landmarks for multiple persons
landmarks_queue = {i: deque(maxlen=10) for i in range(5)}

# Function to calculate average color of the torso
def get_torso_color(frame, x, y, w, h):
    torso_roi = frame[y + int(h * 0.4):y + int(h * 0.7), x:x + w]
    if torso_roi.size == 0:
        return (0, 0, 0)
    average_color = cv2.mean(torso_roi)[:3]
    return average_color

# Function to detect persons and draw bounding boxes
def detect_persons_and_draw_boxes(frame, person_id, trackers):
    results = model(frame)  # Perform inference

    new_trackers = []
    ids_detected = []

    for result in results:
        if result.boxes.cls[0] == 0:  # Class ID 0 corresponds to 'person'
            x, y, x2, y2 = map(int, result.boxes.xyxy[0])
            w, h = x2 - x, y2 - y
            cx, cy = x + w // 2, y + h // 2

            torso_color = get_torso_color(frame, x, y, w, h)

            # Find if this person matches a previously detected person
            min_dist = float('inf')
            min_id = -1
            for tracker in trackers:
                tracker_id, (tx, ty, tw, th), (tcx, tcy), tcolor = tracker
                color_diff = np.linalg.norm(np.array(tcolor) - np.array(torso_color))
                d = dist.euclidean((cx, cy), (tcx, tcy)) + color_diff * 10  # Adjust the weight as needed
                if d < min_dist:
                    min_dist = d
                    min_id = tracker_id

            if min_dist < 50:  # Threshold to determine if it's the same person
                ids_detected.append(min_id)
                new_trackers.append((min_id, (x, y, w, h), (cx, cy), torso_color))
            else:
                ids_detected.append(person_id)
                new_trackers.append((person_id, (x, y, w, h), (cx, cy), torso_color))
                person_id += 1

            # Draw a green rectangle around the person
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Put a unique ID for each person and display torso color
            cv2.putText(frame, f'ID: {ids_detected[-1]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            color_text = f'Color: {tuple(map(int, torso_color))}'
            cv2.putText(frame, color_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f'Person ID: {ids_detected[-1]}, Torso Color: {tuple(map(int, torso_color))}')

    return frame, person_id, new_trackers

# Open webcam
cap = cv2.VideoCapture(0)
person_id = 0
trackers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_boxes, person_id, trackers = detect_persons_and_draw_boxes(frame, person_id, trackers)

    # Display the resulting frame
    cv2.imshow("Frame", frame_with_boxes)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
