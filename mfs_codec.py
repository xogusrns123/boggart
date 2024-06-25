import os
import csv
import subprocess
import shutil
import cv2
import torch
from tqdm import tqdm
import pandas as pd


def read_frame_indices(csv_file):
    frame_indices = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:  # Check if row is not empty
                indices_str = row[0].strip('{}')  # 중괄호 제거
                indices = map(int, indices_str.split(', '))
                frame_indices.extend(indices)
    return sorted(frame_indices)

def move_and_rename_images(source_dir, target_dir, desired_frames):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get a list of all image files in the source directory
    all_files = os.listdir(source_dir)
    image_files = [f for f in all_files if f.startswith('image') and f.endswith('.png')]
    print(len(image_files))
    # Filter and sort the desired frame files
    desired_files = [f'image{frame}.png' for frame in desired_frames if f'image{int(frame)}.png' in image_files]
    desired_files.sort(key=lambda x: int(x[5:-4]))  # Sort files by frame number
    print(len(desired_frames))
    print(len(desired_files))
    print(desired_files[-1])
    # Dictionary to keep track of the next index for each directory
    dir_counters = {}

    # Move and rename files
    for filename in desired_files:
        frame_number = int(filename[5:-4])

        chunk_start = (frame_number // 150) * 150
        chunk_end = chunk_start + 149
        chunk_dir = os.path.join(target_dir, f'frames_{chunk_start}_{chunk_end}')
        
        # Initialize the counter for the directory if not already done
        # if chunk_dir not in dir_counters:
        if not os.path.exists(chunk_dir):
            os.makedirs(chunk_dir)

    # assert len(os.listdir(target_dir)) == 720

    for filename in desired_files:
        frame_number = int(filename[5:-4])

        chunk_start = (frame_number // 150) * 150
        chunk_end = chunk_start + 149
        chunk_dir = os.path.join(target_dir, f'frames_{chunk_start}_{chunk_end}')
        
        # Initialize the counter for the directory if not already done
        if chunk_dir not in dir_counters:
            if not os.path.exists(chunk_dir):
                os.makedirs(chunk_dir)
            dir_counters[chunk_dir] = 0
        
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(chunk_dir, f'frame{dir_counters[chunk_dir]}.png')
        shutil.copy(src_path, dest_path)
        print(f'Copied {src_path} to {dest_path}')
        
        dir_counters[chunk_dir] += 1

def process_experiment_files(experiment_dir, output_dir, source_dir):
    all_frame_indices = []
    print(len(os.listdir(experiment_dir)))
    # Read all CSV files in the experiment directory
    for csv_file in os.listdir(experiment_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(experiment_dir, csv_file)
            frame_indices = read_frame_indices(csv_path)
            all_frame_indices.extend(frame_indices)
    
    # Sort all frame indices
    sorted_frame_indices = sorted(all_frame_indices)

    # Extract frames using FFmpeg
    move_and_rename_images(source_dir, output_dir, sorted_frame_indices)
# # # Example usage
# experiment_dir = '/home/kth/rva/boggart/data/experiment/mfs_0.90'
# output_dir = '/home/kth/rva/images/mfs_0.90_0'
# source_dir = '/home/kth/rva/images/original_0'

# process_experiment_files(experiment_dir, output_dir, source_dir)

# print("Frames have been extracted and saved as images.")



def encode_images_in_subdirectories_to_videos(source_base_dir, output_base_dir, frame_rate=30):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # List all subdirectories in the source base directory
    subdirs = [d for d in os.listdir(source_base_dir) if os.path.isdir(os.path.join(source_base_dir, d))]

    for subdir in sorted(subdirs):
        subdir_path = os.path.join(source_base_dir, subdir)
        image_files = sorted([f for f in os.listdir(subdir_path) if f.startswith('frame') and f.endswith('.png')],
                             key=lambda x: int(x[5:-4]))

        if not image_files:
            print(f"No image files found in {subdir_path}")
            continue

        # Initialize the video writer
        first_image_path = os.path.join(subdir_path, image_files[0])
        first_image = cv2.imread(first_image_path)
        height, width, _ = first_image.shape
        output_video_path = os.path.join(output_base_dir, f'{subdir}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

        for image_file in image_files:
            img_path = os.path.join(subdir_path, image_file)
            img = cv2.imread(img_path)
            video_writer.write(img)
            print(f'Added frame {img_path} to video {output_video_path}')

        video_writer.release()
        print(f'Video saved to {output_video_path}')

# # # Example usage
# source_base_dir = '/home/kth/rva/images/mfs_0.90_0'
# output_base_dir = '/home/kth/rva/video/mfs_0.90_0'
# encode_images_in_subdirectories_to_videos(source_base_dir, output_base_dir)

def save_frame_indices_to_csv(frame_indices, csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame Index'])  # Header
        for frame_index in frame_indices:
            writer.writerow([frame_index])

def map_detected_frames_to_original_indices(detected_frame_indices, sorted_frame_indices):
    mapped_indices = [sorted_frame_indices[i] for i in detected_frame_indices]
    return mapped_indices

def detect_objects_in_video(video_path, model, sorted_frame_indices, output_image_dir, start_index):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize DataFrame to store results for the current video
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

        # Map frame_idx to original frame index using sorted_frame_indices and start_index
        if start_index + frame_idx < len(sorted_frame_indices):
            original_frame_index = sorted_frame_indices[start_index + frame_idx]
            frame_results['frame'] = original_frame_index
            frame_results = frame_results[['frame', 'x1', 'y1', 'x2', 'y2', 'label', 'conf']]
            results_df = pd.concat([results_df, frame_results], ignore_index=True)


            # Draw bounding boxes on the frame
            for _, row in frame_results.iterrows():
                x1, y1, x2, y2, label, conf = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2']), row['label'], row['conf']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the frame as an image
            output_image_path = os.path.join(output_image_dir, f"frame_{original_frame_index}.png")
            cv2.imwrite(output_image_path, frame)
        else:
            break

    cap.release()
    return results_df

def object_detect_on_mfs(experiment_dir, video_file, output_csv_file, output_image_dir, model):
    all_frame_indices = []
    
    # Read all CSV files in the experiment directory
    for csv_file in os.listdir(experiment_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(experiment_dir, csv_file)
            frame_indices = read_frame_indices(csv_path)
            all_frame_indices.extend(frame_indices)
    
    # Sort all frame indices
    sorted_frame_indices = sorted(all_frame_indices)

    # Create an empty CSV file with headers only if the file does not already exist
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    if not os.path.exists(output_csv_file):
        pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"]).to_csv(output_csv_file, index=False)

    # Sort video files by their chunk start value
    video_files = sorted(os.listdir(video_dir), key=lambda x: int(x.split('_')[1]))

    # Process each video file in the video directory
    current_start_index = 0
    for video_file in video_files:
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            # print(f'Processing {video_path}...')
            results_df = detect_objects_in_video(video_path, model, sorted_frame_indices, output_image_dir, current_start_index)

            # # Append results of current video to the main CSV file
            results_df.to_csv(output_csv_file, mode='a', header=False, index=False)
            
            
            # Update start index for the next video
            total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
            current_start_index += total_frames



# # Example usage
experiment_dir = '/home/kth/rva/boggart/data/experiment/mfs_0.90'
video_dir = '/home/kth/rva/video/mfs_6000k_0.90_0'
output_csv_file = '/home/kth/rva/my_codec/mfs_results_6000k_0.90_0.csv'
output_image_dir = '/home/kth/rva/images/mfs_6000k_0.90_0'

if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

ml_model = 'yolov5l'

if torch.cuda.is_available():
    device = torch.device("cuda:3")
    model = torch.hub.load(f'/home/kth/rva/yolov5', 'custom', f'{ml_model}.pt', source='local')
    model.to(device)

object_detect_on_mfs(experiment_dir, video_dir, output_csv_file, output_image_dir, model)
print("Mapped frame indices have been saved to CSV file.")

def merge_detect_mfs_to_all_frames(partial_csv_path, full_csv_path, output_csv_path):
    # Read the CSV files
    partial_df = pd.read_csv(partial_csv_path)
    full_df = pd.read_csv(full_csv_path)

    all_frame_indices = []
    
    # 디렉토리에서 csv 파일 목록을 가져옴
    csv_files = [f for f in os.listdir(experiment_dir) if f.endswith('.csv')]

    # 파일 이름에서 숫자를 기준으로 정렬
    sorted_csv_files = sorted(csv_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Read all CSV files in the experiment directory
    for csv_file in sorted_csv_files:
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(experiment_dir, csv_file)
            frame_indices = read_frame_indices(csv_path)
            all_frame_indices.extend(frame_indices)
    
    # Sort all frame indices
    sorted_frame_indices = sorted(all_frame_indices)
    # Ensure the 'frame' column exists in both dataframes
    if 'frame' not in partial_df.columns or 'frame' not in full_df.columns:
        raise ValueError("Both CSV files must contain 'frame' column")

    # Remove the frames in partial_frames from full_df
    full_df = full_df[~full_df['frame'].isin(sorted_frame_indices)]

    # Concatenate the full_df and partial_df
    merged_df = pd.concat([full_df, partial_df])

    # Sort by frame and any other columns if needed, for consistency
    merged_df.sort_values(by=['frame'], inplace=True)

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_csv_path, index=False)

# # Example usage
# experiment_dir = '/home/kth/rva/boggart/data/experiment/mfs_0.90'
# partial_csv_path = '/home/kth/rva/my_codec/mfs_results_300k_0.90_0.csv'
# full_csv_path = '/home/kth/rva/boggart/inference_results/yolov5l/auburn_first_angle/original/auburn_first_angle10.csv'
# # full_csv_path = '/home/kth/rva/boggart/inference_results/yolov5l/auburn_crf23_live/auburn_crf23_live10.csv'
# output_csv_path = '/home/kth/rva/boggart/inference_results/yolov5l/auburn_first_angle/auburn_first_angle10.csv'

# merge_detect_mfs_to_all_frames(partial_csv_path, full_csv_path, output_csv_path)
# print(f"Merged CSV file has been saved to {output_csv_path}.")