import tkinter as tk
import numpy as np
from tkinter import filedialog
from tkinter import messagebox
import cv2
from threading import Thread
from PIL import Image, ImageTk
import torch
from iouCommon import IOUTracker
from weather_prediction import predict

# Load YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('.', 'custom', path=r'runs/train/exp/weights/best.pt', source='local', force_reload=True)
model = model.to(device)
model.eval()
class_names = ['car', 'bus', 'van', 'others']
class_colors = {'car': (0, 255, 0), 'bus': (0, 0, 255), 'van': (255, 0, 0), 'others': (0, 255, 255)}

# Initialize IOU Tracker
iou_tracker = IOUTracker(iou_threshold=0.5, max_age=1)

def detect_lane_lines(frame):
    """ Detect lane lines """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Define ROI. In this project, the ROI area needs to be manually adjusted according to different videos.
    height, width = frame.shape[:2]
    roi_mask = np.zeros_like(edges)
    polygon = np.array([[
        (596, 347),
        (596, 720),
        (1150, 720),
        (1150, 347)
    ]], np.int32)
    cv2.fillPoly(roi_mask, polygon, 255)
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    # Lane detection using Hough transform
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=110, minLineLength=100, maxLineGap=20)

    return lines if lines is not None else [], roi_mask

def is_vehicle_crossing_lane(vehicle_bbox, lane_lines):
    """ Determine if the vehicle is significantly crossing the lane lines. """
    x1, y1, x2, y2 = vehicle_bbox

    for line in lane_lines:
        x1_lane, y1_lane, x2_lane, y2_lane = line[0]

        # Calculate the Euclidean distance between the center of the vehicle and the lane line
        line_center_x = (x1_lane + x2_lane) / 2
        vehicle_center_x = (x1 + x2) / 2
        distance = abs(line_center_x - vehicle_center_x)

        # Define a threshold for crossing the lane line
        distance_threshold = 10

        if distance < distance_threshold:
            return True  # crossing
    return False

class VideoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Detection")

        # Video display area
        self.video_frame = tk.Frame(root, width=1280, height=720, bg="black")
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)
        self.canvas = tk.Canvas(self.video_frame, width=1280, height=720, bg="black")
        self.canvas.pack()

        # Control button area
        self.control_frame = tk.Frame(root, width=200, height=720, bg="white")
        self.control_frame.grid(row=0, column=1, padx=10, pady=10)

        # Mouse coordinates display
        self.coord_label = tk.Label(self.control_frame, text="Mouse Coordinates: (x, y)", bg="white")
        self.coord_label.pack(pady=10)

        # Button
        self.import_btn = tk.Button(self.control_frame, text="Import Video", command=self.import_video)
        self.import_btn.pack(pady=20)

        self.start_btn = tk.Button(self.control_frame, text="Start detection", command=self.start_detection)
        self.start_btn.pack(pady=20)

        self.exit_btn = tk.Button(self.control_frame, text="Exit", command=self.exit)
        self.exit_btn.pack(pady=20)

        # Southbound traffic flow threshold input box
        self.south_threshold_label = tk.Label(self.control_frame, text="Southbound Traffic Thresholds:")
        self.south_threshold_label.pack(pady=10)
        self.south_threshold_entry = tk.Entry(self.control_frame)
        self.south_threshold_entry.pack(pady=10)

        # Northbound traffic flow threshold input box
        self.north_threshold_label = tk.Label(self.control_frame, text="Northbound Traffic Thresholds:")
        self.north_threshold_label.pack(pady=10)
        self.north_threshold_entry = tk.Entry(self.control_frame)
        self.north_threshold_entry.pack(pady=10)

        # Automatic thresholds based on weather
        self.auto_threshold_var = tk.BooleanVar(value=False)
        self.auto_threshold_checkbox = tk.Checkbutton(self.control_frame, text="Auto Thresholds Based on Weather",
                                                      variable=self.auto_threshold_var,
                                                      command=self.update_threshold_mode)
        self.auto_threshold_checkbox.pack(pady=10)

        self.set_threshold_btn = tk.Button(self.control_frame, text="Set", command=self.set_threshold)
        self.set_threshold_btn.pack(pady=20)

        # Set the default threshold to infinity
        self.south_threshold = float('inf')
        self.north_threshold = float('inf')

        self.video_path = None
        self.cap = None
        self.running = False

        self.canvas.bind("<Motion>", self.update_coordinates)

    def update_threshold_mode(self):
        """Traffic flow threshold update mode"""
        if self.auto_threshold_var.get():
            # If the automatic threshold option is selected, adjust the threshold according to the weather
            if self.current_weather is not None:
                self.south_threshold, self.north_threshold = self.get_weather_based_threshold(self.current_weather)
                messagebox.showinfo("Info", f"Thresholds set based on weather: {self.current_weather}")
        else:
            # If unchecked, manually set the threshold
            self.set_threshold()

    def get_weather_based_threshold(self, weather):
        """Returns the traffic flow threshold based on weather"""
        weather_thresholds = {
            "sunny": (12, 12),
            "rainy": (7, 7),
            "snow": (5, 5),
            "cloudy": (10, 10),
        }
        return weather_thresholds.get(weather, (float('inf'), float('inf')))

    def update_coordinates(self, event):
        """ Update mouse coordinates """
        x, y = event.x, event.y
        self.coord_label.config(text=f"Mouse Coordinates: ({x}, {y})")
    def import_video(self):
        # Pop up the file selection window
        self.video_path = filedialog.askopenfilename(filetypes=[("video file", "*.mp4 *.avi *.mov")])
        if self.video_path:
            messagebox.showinfo("Prompt", f"Selected Videos: {self.video_path}")

            # Capture the first frame as the cover image
            self.cap = cv2.VideoCapture(self.video_path)
            ret, frame = self.cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (1280, 720))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.imgtk = imgtk

            self.cap.release()
        else:
            messagebox.showwarning("Warning", "Video file not selected！")

    def set_threshold(self):
        """ Read the traffic flow threshold in the input box and set it """
        self.south_threshold = self.get_threshold(self.south_threshold_entry)
        self.north_threshold = self.get_threshold(self.north_threshold_entry)

        messagebox.showinfo("Prompt", "Traffic flow thresholds have been set！")

    def start_detection(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please import the video file first！")
            return

        self.running = True
        self.cap = cv2.VideoCapture(self.video_path)
        Thread(target=self.detect_video).start()

    def get_threshold(self, entry_widget):
        """ Get the traffic flow threshold in the input box """
        input_value = entry_widget.get().strip()
        if input_value == "":
            return float('inf')

        try:
            threshold = float(input_value)
            if threshold < 0:
                messagebox.showerror("Error", "The threshold cannot be negative！")
                return float('inf')
            return threshold
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid value！")
            return float('inf')

    def detect_video(self):
        """ Implementation of main detection functions """
        frame_count = 0  # Frame counter
        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (1280, 720))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Update weather every 500 frames
            if frame_count % 500 == 0:
                self.current_weather = predict(frame_rgb, modelPath="weather_model/weather-2022-10-14-07-36-57.pth")
                if self.auto_threshold_var.get():
                    self.south_threshold, self.north_threshold = self.get_weather_based_threshold(self.current_weather)
            cv2.putText(frame_resized, f"Weather: {self.current_weather}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

            # Detect lane lines (not enabled by default, It is necessary to manually adjust the ROI area according to
            # different videos and then uncomment it.)
            # lane_lines, roi_mask = detect_lane_lines(frame_resized)

            # Visualize ROI area (not enabled by default, you can uncomment it if needed)
            # roi_color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
            # frame_resized = cv2.addWeighted(frame_resized, 0.8, roi_color, 0.2, 0)

            # Visualize detected lane lines (not enabled by default, you can uncomment it if needed)
            # for line in lane_lines:
            #     x1_lane, y1_lane, x2_lane, y2_lane = line[0]
            #     cv2.line(frame_resized, (x1_lane, y1_lane), (x2_lane, y2_lane), (0, 255, 0), 2)

            # Vehicle Detection
            results = model(frame_rgb)
            detections = []
            for result in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = result
                if conf > 0.5:
                    detections.append([int(x1), int(y1), int(x2), int(y2), int(cls)])

            tracks = iou_tracker.update(detections)

            for track in tracks:
                x1, y1, x2, y2 = track['bbox']
                cls = track['cls']
                label = class_names[cls]
                color = class_colors[label]

                # Determine whether the vehicle crosses the solid line (It is not enabled by default. If you need to
                # use it, please uncomment the lane detection related code.)
                # if is_vehicle_crossing_lane((x1, y1, x2, y2), lane_lines):
                    # If a vehicle changes lanes across a solid line, mark and warn
                    # cv2.putText(frame_resized, f"Violation: Crossing the line!", (x1, y1 - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # If the vehicle's lost frames are less than the maximum lost frames
                if track['lost'] < iou_tracker.max_age:
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Statistics display
            cv2.putText(frame_resized, f"North: {iou_tracker.north_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
            cv2.putText(frame_resized, f"South: {iou_tracker.south_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)

            cv2.putText(frame_resized, f"Northbound Traffic Thresholds: {self.north_threshold}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            cv2.putText(frame_resized, f"Southbound Traffic Thresholds: {self.south_threshold}", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

            # Check if the traffic volume exceeds the threshold
            if iou_tracker.north_count > self.north_threshold:
                messagebox.showwarning("Warning", "Northbound traffic exceeds thresholds！")
            if iou_tracker.south_count > self.south_threshold:
                messagebox.showwarning("Warning", "Southbound traffic exceeds thresholds！")

            # Update video frame
            img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk
            self.canvas.update()

            frame_count += 1

        self.cap.release()

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def exit(self):
        """ Function of the Exit button """
        self.stop_detection()
        self.root.quit()

    def on_closing(self):
        self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    gui = VideoGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.on_closing)
    root.mainloop()

