import subprocess
import csv
import numpy as np

def get_i_frame_info(video_path, interval='0:00:00%+0:03:00'):
    # Run ffprobe command to get I-frame information for the first 3 minutes
    cmd = [
        'ffprobe', '-read_intervals', interval, '-select_streams', 'v', '-show_frames', '-show_entries',
        'frame=pkt_size,pict_type,coded_picture_number', '-of', 'csv=print_section=0', video_path
    ]
    # pkt pos 이라는 인수를 이용해서 확인해보자
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    i_frame_info = []

    # Parse the ffprobe output
    for line in result.stdout.split('\n'):
        if line:
            frame_info = line.split(',')
            if len(frame_info) >= 3 and frame_info[1] == 'I':  # Check if the frame is an I-frame
                frame_number = int(frame_info[2])
                frame_size = int(frame_info[0])
                i_frame_info.append((frame_number, frame_size))
    
    return i_frame_info

def calculate_average_i_frame_sizes_and_ratios(i_frame_info, chunk_size, total_frames):
    if not i_frame_info:
        return []

    max_frame_number = max(frame_number for frame_number, _ in i_frame_info)
    num_chunks = (max_frame_number // chunk_size) + 1
    average_sizes = []
    i_frame_ratios = []

    for i in range(num_chunks):
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size
        chunk_frames = [size for frame_number, size in i_frame_info if chunk_start <= frame_number < chunk_end]
        if chunk_frames:
            average_size = np.mean(chunk_frames)
            i_frame_ratio = len(chunk_frames) / chunk_size * 100
        else:
            average_size = 0
            i_frame_ratio = 0
        average_sizes.append((chunk_start, chunk_end, average_size))
        i_frame_ratios.append((chunk_start, chunk_end, i_frame_ratio))
    
    return average_sizes, i_frame_ratios

def plot_i_frame_ratios(i_frame_ratios):
    chunk_starts = [chunk_start for chunk_start, chunk_end, ratio in i_frame_ratios]
    ratios = [ratio for chunk_start, chunk_end, ratio in i_frame_ratios]

    plt.figure(figsize=(10, 6))
    plt.bar(chunk_starts, ratios, width=chunk_size, align='edge')
    plt.xlabel('Chunk Start Frame')
    plt.ylabel('I-Frame Ratio')
    plt.title('I-Frame Ratio per 1800 Frame Chunk')
    plt.savefig('/home/kth/rva/auburn_I_frame_ratio.png')

# if __name__ == "__main__":
#     video_path = '/home/kth/rva/auburn_first_angle.mp4'
#     chunk_size = 1800
#     total_frames = 1800 * 3  # Assume 3 minutes at 30 fps (3 * 60 * 30)

#     i_frame_info = get_i_frame_info(video_path)
    
#     # Calculate average I-frame sizes and ratios for each 1800 frame chunk
#     average_sizes, i_frame_ratios = calculate_average_i_frame_sizes_and_ratios(i_frame_info, chunk_size, total_frames)
    
#     # Print the average sizes
#     for chunk_start, chunk_end, avg_size in average_sizes:
#         print(f"Chunk {chunk_start}-{chunk_end}: Average I-Frame Size = {avg_size} bytes")

#     # Print the I-frame ratios
#     for chunk_start, chunk_end, ratio in i_frame_ratios:
#         print(f"Chunk {chunk_start}-{chunk_end}: I-Frame Ratio = {ratio}")

#     # Plot the I-frame ratios
#     plot_i_frame_ratios(i_frame_ratios)



def get_gop_sizes(video_path):
    # Run ffprobe command to get frame type information
    cmd = [
        'ffprobe', '-read_intervals', '0:00:00%+0:03:00', '-select_streams', 'v', '-show_frames', '-show_entries',
        'frame=pict_type,coded_picture_number', '-of', 'csv=print_section=0', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    frame_info = result.stdout.split('\n')

    gop_sizes = []
    last_i_frame_number = None

    # Parse the ffprobe output
    for line in frame_info:
        if line:
            frame_data = line.split(',')
            frame_type = frame_data[0]
            frame_number = int(frame_data[1])
            
            if frame_type == 'I':
                if last_i_frame_number is not None:
                    gop_size = frame_number - last_i_frame_number
                    gop_sizes.append(gop_size)
                last_i_frame_number = frame_number

    return gop_sizes

import subprocess
import csv

def get_frame_info(video_path, interval):
    # Run ffprobe command to get frame type and size information
    cmd = [
        'ffprobe', '-read_intervals', interval ,'-select_streams', 'v', '-show_frames', '-show_entries',
        'frame=pict_type,pkt_size,coded_picture_number', '-of', 'csv=print_section=0', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    frame_info = []
    gop_sizes = []

    # Parse the ffprobe output
    for line in result.stdout.split('\n'):
        if line:
            # print(line)
            frame_data = line.split(',')
            frame_type = frame_data[1]
            frame_size = int(frame_data[0])
            frame_number = int(frame_data[2])
            frame_info.append((frame_number, frame_type, frame_size))

    return frame_info

def analyze_gop(frame_info):
    gop_info = []
    current_gop = []
    for frame in frame_info:
        frame_number, frame_type, frame_size = frame
        if frame_type == 'I':
            if current_gop:
                gop_info.append(current_gop)
            current_gop = []
        current_gop.append((frame_number, frame_type, frame_size))
    if current_gop:
        gop_info.append(current_gop)
    return gop_info

def save_gop_info(gop_info, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['GOP Index', 'Frame Number', 'Frame Type', 'Frame Size (bytes)'])
        for gop_index, gop in enumerate(gop_info):
            for frame_number, frame_type, frame_size in gop:
                csvwriter.writerow([gop_index, frame_number, frame_type, frame_size])

if __name__ == "__main__":
    video_path = '/home/kth/rva/video/auburn_segment_0.mp4'
    output_csv = '/home/kth/rva/gop_frame_info.csv'
    interval = '0:00:00%+0:01:00'
    # Get frame information
    frame_info = get_frame_info(video_path, interval)
    
    # Analyze GOPs
    gop_info = analyze_gop(frame_info)
    
    # Save GOP information to CSV
    save_gop_info(gop_info, output_csv)

    print(f"GOP information saved to {output_csv}")

# if __name__ == "__main__":
#     video_path = '/home/kth/rva/auburn_first_angle.mp4'
#     gop_sizes = get_gop_sizes(video_path)

#     # Print GOP sizes
#     for idx, gop_size in enumerate(gop_sizes):
#         print(f"GOP {idx + 1}: {gop_size} frames")

#     # Calculate and print average GOP size
#     if gop_sizes:
#         average_gop_size = sum(gop_sizes) / len(gop_sizes)
#         print(f"Average GOP size: {average_gop_size:.2f} frames")
#     else:
#         print("No I-frames found.")


# import subprocess
# import pandas as pd
# from io import StringIO

# # 비디오 파일 경로
# video_path = '/home/kth/rva/auburn_first_angle.mp4'

# # ffprobe 명령어
# cmd = [
#     'ffprobe', '-read_intervals', '0:00:00%+0:03:00', '-select_streams', 'v', '-show_frames', '-show_entries',
#     'frame=pkt_pos', '-of', 'csv=print_section=0', video_path
# ]

# # ffprobe 명령어 실행
# result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# # 결과 출력
# if result.returncode == 0:
#     # CSV 형식의 결과를 pandas DataFrame으로 변환
#     csv_data = StringIO(result.stdout)
#     df = pd.read_csv(csv_data, header=None, names=['pkt_pos'])
    
#     # 프레임 사이즈 계산
#     df['pkt_pos'] = df['pkt_pos'].astype(int)
#     total_size = df['pkt_pos'].diff().fillna(0)

#     # total_size 데이터를 CSV 파일로 저장
#     total_size.to_csv('/home/kth/rva/total_size.csv', header=['size'], index_label='index')

#     # 각 index의 값이 50000 이상인 값들 추출
#     large_sizes = total_size[total_size >= 50000]

#     # 평균 계산
#     if not large_sizes.empty:
#         avg_large_size = large_sizes.mean()
#         print(f"Average size of frames with size >= 100000: {avg_large_size} bytes")
#     else:
#         print("No frames with size >= 50000 found.")
# else:
#     # 오류 메시지 출력
#     print("Error:", result.stderr)