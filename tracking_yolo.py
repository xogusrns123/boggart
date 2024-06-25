from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import os
import torch
from tqdm import tqdm

print("start")
output_dir = '/usr/src/app/tracking_results/auburn_first_angle10' 
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
output_path = os.path.join(output_dir, f'yolov8n.csv')
if not os.path.exists(output_path):
    pd.DataFrame(columns=["ObjId", "TS", "x1", "y1", "x2", "y2", "bstate"]).to_csv(output_path, index=False)
# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "/usr/src/app/auburn_first_angle.mp4"
cap = cv2.VideoCapture(video_path)
# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Store the track history
track_history = defaultdict(lambda: [])

print("start process")
# Loop through the video frames
for ts in tqdm(range(total_frames), desc="Processing frames"):
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.25, iou=0.3)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Plot the tracks
        data = {
            'ObjId': [],
            'TS': [],
            'x1': [],
            'y1': [],
            'x2': [],
            'y2': [],
            'bstate': []
        }
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            data['ObjId'].append(track_id)
            data['TS'].append(ts)
            data['x1'].append(int(torch.round(x).item()))
            data['y1'].append(int(torch.round(y).item()))
            data['x2'].append(int(torch.round(x+w).item()))
            data['y2'].append(int(torch.round(y+h).item()))
            data['bstate'].append('ObjectState.OBJECT')
        df = pd.DataFrame(data)
        # df.to_csv(output_path, mode='a', index=False, header=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imwrite(f"/usr/src/app/tracking_results/auburn_first_angle10/img{ts}.png", annotated_frame)
        if ts == 108000:
            break
    else:
        break
    #     # Display the annotated frame
    #     cv2.imshow("YOLOv8 Tracking", annotated_frame)

    #     # Break the loop if 'q' is pressed
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
    # else:
    #     # Break the loop if the end of the video is reached
    #     break

# Release the video capture object and close the display window
cap.release()
# cv2.destroyAllWindows()


