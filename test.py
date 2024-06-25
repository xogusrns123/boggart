import os
import csv

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



def read_all(experiment_dir):
    all_frame_indices = []
# Read all CSV files in the experiment directory
    for csv_file in os.listdir(experiment_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(experiment_dir, csv_file)
            frame_indices = read_frame_indices(csv_path)
            all_frame_indices.extend(frame_indices)
    
    # Sort all frame indices
    sorted_frame_indices = sorted(all_frame_indices)
    # print(sorted_frame_indices)


experiment_dir = '/home/kth/rva/boggart/data/experiment/mfs_0.95'
a= read_all(experiment_dir)
another = '/home/kth/rva/boggart/data/experiment/mfs_0.95_test'
b = read_all(another)
print(a==b)



print(len(os.listdir('/home/kth/rva/images/mfs')))