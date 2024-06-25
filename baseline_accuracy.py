import os
import pandas as pd
import json
from mongoengine import connect, disconnect
from ModelProcessor import ModelProcessor
from VideoData import VideoData
from utils import (calculate_bbox_accuracy, calculate_count_accuracy,
                   get_ioda_matrix, calculate_binary_accuracy, parallelize_update_dictionary)
import numpy as np
from visuallize import ResultProcessor

query_class = 2
query_conf = 0.7
fps = 30
video_name = "auburn_first_angle"
    
main_dir = "/home/kth/rva"
video_path = f"{main_dir}/{video_name}.mp4"
video_chunk_path = f"{main_dir}/boggart/data/{video_name}10/video/{video_name}10_0.mp4"
output_video_path = f'{main_dir}/{video_name}_output.mp4'
yolo_path = f"{main_dir}/boggart/inference_results/yolov5/{video_name}/{video_name}10.csv"
trajectory_dir = f"{main_dir}/boggart/data/{video_name}10/trajectories"
query_result_dir = f"{main_dir}/boggart/data/{video_name}10/query_results/bbox"
boggart_result_dir = f"{main_dir}/boggart/data/{video_name}10/boggart_results/bbox"

"""
0: yolo_detection
1: trajectory
2: boggart results
3: query_results
"""
_type = 2
end_frame = 0.0
target_acc = 0.9
vd = VideoData(video_name, 10)
processor = ResultProcessor(vd, video_path, output_video_path, trajectory_dir, boggart_result_dir, query_result_dir, yolo_path, _type, end_frame, target_acc)

boggart_results_files = processor.get_smallest_mfs_files()



def get_gt(video_name, hour, model, query_segment_start, query_segment_size = 1800):
    video_data = VideoData(video_name, hour)
    modelProcessor = ModelProcessor(model, video_data, query_class, query_conf, fps)

    gt_bboxes, gt_counts = modelProcessor.get_ground_truth(query_segment_start, query_segment_start + query_segment_size)

    # disconnect(alias='my')
    # print(gt_bboxes)
    return gt_bboxes

def get_boggart_list(chunk_start):
    for path in boggart_results_files:
        if int(path.split('/')[-1].split('_')[0]) == int(chunk_start):
            # Load bounding boxes from JSON file
            with open(path, 'r') as f:
                bboxes = json.load(f)
                return bboxes

def accuracy(chunk_start):

    scores = []

    gt_bboxes = get_gt("auburn_first_angle_base", 10, "yolov5l", chunk_start)
    # print(gt_bboxes)
    # det_bboxes = get_boggart_list(chunk_start)
    det_bboxes = get_gt("mfs800k_ultra_max", 10, "yolov5l", chunk_start)
    # print(det_bboxes)
    # bounding box accuracy
    for bbox_gt, sr in zip(gt_bboxes, det_bboxes):
        scores.append(calculate_bbox_accuracy(bbox_gt, sr))
    # print(round(np.mean(np.array(scores)),4))
    return {"scores": scores}


# for path in boggart_results_files:
#     if int(path.split('/')[-1].split('_')[0]) == 1800:
#         # Load bounding boxes from JSON file
#         with open(path, 'r') as f:
#             bboxes = json.load(f)
#             print(bboxes)
# print(boggart_results_files)
total_scores = []
scores_dict = parallelize_update_dictionary(accuracy, range(0, 108000, 1800), max_workers=40, total_cpus=40)
for ts, score in scores_dict.items():
    total_scores.extend(score['scores'])

print(round(np.mean(np.array(total_scores)),4))