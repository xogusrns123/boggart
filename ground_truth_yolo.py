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
video_name = "auburn_first_angle"
ml_model = "yolov5l"
hour = 10
# csv_path = f'{main_dir}/inference_results/yolov3-coco/auburn_first_angle/auburn_crf23_first_angle10.csv'
csv_dir = f"{BOGGART_REPO_PATH}/inference_results/{ml_model}/{video_name}"
csv_path = os.path.join(csv_dir, f"{video_name}{hour}.csv")
# csv_path = f"{BOGGART_REPO_PATH}/inference_results/{ml_model}/{video_name}/{video_name}{hour}.csv"

def run_yolo(ingest_combos, vd, chunk_size):
    try:
        # Create an empty CSV file with headers only if the file does not already exist
        os.makedirs(csv_dir, exist_ok=True)
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"]).to_csv(csv_path, index=False)

        for k in range(len(ingest_combos)):
            # Initialize DataFrame to store results for the current video
            results_df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"])

            vals = ingest_combos[k]
            chunk_start = vals[1][0]

            # t = Tracker(chunk_start)

            frame_generator = vd.get_frames_by_bounds(chunk_start, chunk_start+chunk_size, int(1))
            for i in trange(chunk_start, chunk_start+chunk_size, int(1), leave=False, desc=f"{chunk_start}_{chunk_size}"):
                f = next(frame_generator)
                if f is None:
                    print(f"skipping frame {i}")
                    # assert i > chunk_start+self.traj_config.chunk_size - 250, i
                    continue

                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                results = model(f)

                # Convert results to DataFrame
                frame_results = results.pandas().xyxy[0]
                frame_results = frame_results.rename(columns={
                    "xmin": "x1", "ymin": "y1", "xmax": "x2", "ymax": "y2", 
                    "confidence": "conf", "name": "label"})
                
                # Change label from 'truck' to 'car'
                frame_results['label'] = frame_results['label'].replace('truck', 'car')

                frame_results['frame'] = i
                frame_results = frame_results[['frame', 'x1', 'y1', 'x2', 'y2', 'label', 'conf']]

                # Append current frame results to main DataFrame
                results_df = pd.concat([results_df, frame_results], ignore_index=True)

            # Append results of current video to the main CSV file
            results_df.to_csv(csv_path, mode='a', header=False, index=False)   
    except Exception as e:
        print("FAILED AT ", e)



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:3")
        model = torch.hub.load(f'/home/kth/rva/yolov5', 'custom', f'{ml_model}.pt', source='local')
        model.to(device)


    chunk_size = 1800
    query_seg_size = 1800

    video_data = VideoData(db_vid = video_name, hour = hour)

    minutes = list(range(0, 60 * 1800, 1800))

    param_sweeps = {
        "diff_thresh" : [16],
        "peak_thresh": [0.1],
        "fps": [30]
    }

    sweep_param_keys = list(param_sweeps.keys())[::-1]
    # print(sweep_param_keys)
    _combos = list(product(*[param_sweeps[k] for k in sweep_param_keys]))
    # print(_combos)
    segment_combos = []
    for minute in minutes:
        chunk_starts = list(range(minute, minute+1800, chunk_size))
        segment_combos.append(chunk_starts)
    # print(segment_combos)
    ingest_combos = list(product(_combos, segment_combos))
    # print(ingest_combos)
    # detect using model and store to csv file
    run_yolo(ingest_combos, video_data, chunk_size)


    # csv -> mongodb
    # load_to_mongodb()