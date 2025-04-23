# Functions of IOU Tracker and Kalman Filter
import torch
import numpy as np
import cv2

class_names = ['car', 'bus', 'van', 'others']

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1, x1_2)
    inter_y1 = max(y1, y1_2)
    inter_x2 = min(x2, x2_2)
    inter_y2 = min(y2, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    iou_value = inter_area / (box1_area + box2_area - inter_area)

    # Additional centroid-based distance metric
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
    cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
    centroid_dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    # Penalize IOU if centroid distance is large
    return iou_value - centroid_dist * 0.001

# Kalman filter initialization function
def create_kalman_filter(x, y,z,h):
    kf = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, vx, vy), 2 measurement variables (x, y)

    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)

    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)

    initial_velocity_x = (z - x) / 2
    initial_velocity_y = (h - y) / 2
    kf.statePre = np.array([x, y, initial_velocity_x, initial_velocity_y], dtype=np.float32)

    # Initialize the error covariance matrix
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1000

    # Noise
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1

    return kf

def update_direction(track):
    """Determine the vehicle direction based on the speed of the Kalman filter"""
    kf = track['kalman']
    velocity_x = kf.statePost[2]
    velocity_y = kf.statePost[3]

    print(f"Track {track['id']} - Velocity: (x: {velocity_x}, y: {velocity_y})")

    if velocity_y < -0.5:
        return 'north'
    elif velocity_y > 0.5:
        return 'south'
    return None

class IOUTracker:
    def __init__(self, iou_threshold=0.3, max_age=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        self.class_ids = {i: 1 for i in range(len(class_names))}
        self.north_count = 0
        self.south_count = 0

    def update(self, detections):
        new_tracks = []
        for det in detections:
            matched = False
            for track in self.tracks:
                if iou(det[:4], track['bbox']) > self.iou_threshold:
                    kf = track['kalman']
                    prediction = kf.predict()
                    predicted_x, predicted_y = int(prediction[0]), int(prediction[1])
                    # print(predicted_x, predicted_y)

                    # Update target position
                    track['bbox'] = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]
                    track['lost'] = 0
                    track['bbox'] = det[:4]
                    track['last_bbox'] = track['bbox']

                    # Correct the prediction using the detected box
                    measurement = np.array([[det[0]], [det[1]]], dtype=np.float32)
                    kf.correct(measurement)
                    matched = True
                    new_tracks.append(track)
                    break

            if not matched:
                cls = det[4]
                new_id = self.class_ids[cls]
                self.class_ids[cls] += 1
                # For each new detection, create a new Kalman filter
                new_kf = create_kalman_filter(det[0], det[1], det[2], det[3])
                new_tracks.append(
                    {'id': new_id, 'bbox': det[:4], 'cls': cls, 'lost': 0, 'last_bbox': det[:4], 'kalman': new_kf})

        # Handle lost tracks
        for track in self.tracks:
            if track not in new_tracks:
                track['lost'] += 1
                if track['lost'] >= self.max_age:
                    continue
                new_tracks.append(track)

        self.tracks = new_tracks

        # Count the vehicles going north and south
        self.north_count = 0
        self.south_count = 0
        for track in self.tracks:
            if track['lost'] < self.max_age:
                direction = update_direction(track)  # Get direction using Kalman filter's speed
                if direction == 'north':
                    self.north_count += 1
                elif direction == 'south':
                    self.south_count += 1

            track['last_bbox'] = track['bbox']

        return self.tracks