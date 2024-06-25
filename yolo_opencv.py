from itertools import product
from tqdm import tqdm, trange
from VideoData import VideoData, NoMoreVideo
from ut_tracker import Tracker
from tqdm import tqdm, trange
import torch
import numpy as np
import pandas as pd
import os
import cv2
from configs import BOGGART_REPO_PATH
# from load_detections_into_mongodb import load_to_mongodb

# video_name = "auburn_crf47_first_angle"
# ml_model = "yolov3-coco"
video_name = "mfs800k_ultra_max"
ml_model = "yolov5l"
hour = 10
# csv_path = f'{main_dir}/inference_results/yolov3-coco/auburn_first_angle/auburn_crf23_first_angle10.csv'
csv_dir = f"{BOGGART_REPO_PATH}/inference_results/{ml_model}/{video_name}"
csv_path = os.path.join(csv_dir, f"{video_name}{hour}.csv")
# csv_path = f"{BOGGART_REPO_PATH}/inference_results/{ml_model}/{video_name}/{video_name}{hour}.csv"

def run_yolo(video_path):
    try:
        # Create an empty CSV file with headers only if the file does not already exist
        os.makedirs(csv_dir, exist_ok=True)
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"]).to_csv(csv_path, index=False)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        results_df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"])
        
        for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if frame is None:
                print(f"Skipping frame {frame_idx}")
                continue

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = model(frame_gray)

            # Convert results to DataFrame
            frame_results = results.pandas().xyxy[0]
            frame_results = frame_results.rename(columns={
                "xmin": "x1", "ymin": "y1", "xmax": "x2", "ymax": "y2", 
                "confidence": "conf", "name": "label"})
                
            # Change label from 'truck' to 'car'
            frame_results['label'] = frame_results['label'].replace('truck', 'car')

            frame_results['frame'] = frame_idx
            frame_results = frame_results[['frame', 'x1', 'y1', 'x2', 'y2', 'label', 'conf']]

            # Append current frame results to main DataFrame
            results_df = pd.concat([results_df, frame_results], ignore_index=True)

        # Append results of current video to the main CSV file
        results_df.to_csv(csv_path, mode='a', header=False, index=False)   
    except Exception as e:
        print("FAILED AT ", e)



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:2")
        model = torch.hub.load(f'/home/kth/rva/yolov5', 'custom', f'{ml_model}.pt', source='local')
        model.to(device)


    video_path = f'/home/kth/rva/video/{video_name}.mp4'
    run_yolo(video_path)


    # csv -> mongodb
    # load_to_mongodb()