import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
from visuallize import ResultProcessor


video_name = "lausanne_pont_bassieres"
    
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
end_frame = 0
processor = ResultProcessor(video_path, output_video_path, trajectory_dir, boggart_result_dir, query_result_dir, yolo_path, _type, end_frame)


def draw_line_graph():
    # 최고 score를 가진 파일의 min_frames 데이터를 저장할 딕셔너리
    best_score_files = {}

    # 각 CSV 파일을 반복하여 처리
    for file in csv_files:
        # 파일명 분석
        filename = os.path.basename(file)
        parts = filename.split('_')
        chunk_start = int(parts[0])

        # 파일 읽기
        df = pd.read_csv(file)

        # 파일에서 첫 번째 행의 score와 min_frames 값 가져오기
        score = df.at[0, 'score']
        total_min_frames = df.at[0, 'min_frames']

        # 동일한 chunk_start에 대해 가장 높은 score를 가진 파일만 고려
        if chunk_start not in best_score_files or best_score_files[chunk_start]['score'] < score:
            best_score_files[chunk_start] = {'score': score, 'min_frames': total_min_frames}

    # 그래프 데이터 준비 (chunk_start를 기준으로 정렬)
    sorted_items = sorted(best_score_files.items())
    x_values = [item[0] for item in sorted_items]
    percent_min_frames = [(item[1]['min_frames'] / 1800 * 100) for item in sorted_items]
    total_min_frames_sum = sum(item[1]['min_frames'] for item in sorted_items)

    # 데이터를 기반으로 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, percent_min_frames, color='b', label='Percent of Min Frames')
    plt.plot(x_values, percent_min_frames, 'r--', alpha=0.5)  # 선으로 연결하기
    plt.title('Percentage of Minimum Frames by Chunk Start')
    plt.xlabel('Chunk Start')
    plt.ylabel('Percentage of Min Frames (%)')
    plt.legend()
    plt.grid(True)

    # 최종 min_frames 합계를 그래프에 표시
    plt.text(x_values[-1], percent_min_frames[-1], f'Total min_frames(%): {round(total_min_frames_sum / 108000,3)}', 
            horizontalalignment='right', verticalalignment='top', fontsize=12, color='green')

    # 그래프를 파일로 저장
    plt.savefig('min_frames_percentage.png')


def draw_bar_graph():
    df = processor.get_smallest_mfs_info()
    x_values = [i * 20 for i in range(len(df))]  # x값에 인덱스를 10배 증가시켜 넓은 간격 부여
    df = df.sort_values(by='chunk_start').reset_index(drop=True)
    print(df)
    # percent_min_frames 열 추가
    df['percent_min_frames'] = (df['min_frames'] / 1800) * 100
    # min_frames의 총합 계산
    total_min_frames = df['min_frames'].sum()
    
    # 데이터를 기반으로 막대 그래프 그리기
    plt.figure(figsize=(20, 7))
    bars = plt.bar(x_values, df['percent_min_frames'], color='skyblue', label='Percent of Min Frames', width=19)

    plt.title('Percentage of Minimum Frames by Chunk Start')
    plt.xlabel('Chunk Start Index')
    plt.ylabel('Percentage of Min Frames (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # x축 레이블에 실제 chunk_start 값을 표시
    plt.xticks(x_values[::5], [item for item in df['chunk_start']][::5], rotation=45)  # 실제 chunk_start 값을 x축 레이블로 사용
    # plt.yticks(np.arange(0, 11, 1))

    # 최종 min_frames 합계를 그래프에 표시
    plt.gcf().text(0.92, 0.95, f'Total min_frames(%): {round(total_min_frames / 108000,3)}', 
                fontsize=12, color='green', horizontalalignment='right', verticalalignment='top')
    

    # 그래프를 파일로 저장
    plt.savefig('min_frames_percentage.png')


def draw_bar_group():
    # 최고 score를 가진 파일의 min_frames 데이터를 저장할 딕셔너리
    best_score_files = {}

    # 각 CSV 파일을 반복하여 처리
    for file in csv_files:
        # 파일명 분석
        filename = os.path.basename(file)
        parts = filename.split('_')
        chunk_start = int(parts[0])

        # 파일 읽기
        df = pd.read_csv(file)

        # 파일에서 첫 번째 행의 score와 min_frames 값 가져오기
        score = df.at[0, 'score']
        total_min_frames = df.at[0, 'min_frames']

        # 동일한 chunk_start에 대해 가장 높은 score를 가진 파일만 고려
        if chunk_start not in best_score_files or best_score_files[chunk_start]['score'] < score:
            best_score_files[chunk_start] = {'score': score, 'min_frames': total_min_frames}

    # 그룹 데이터 준비
    group_size = 10
    grouped_min_frames = {}
    for chunk_start, data in best_score_files.items():
        group_index = chunk_start // (1800 * group_size)
        if group_index not in grouped_min_frames:
            grouped_min_frames[group_index] = []
        grouped_min_frames[group_index].append(data['min_frames'])

    # 각 그룹별 총 min_frames 합계를 계산하고, 그룹의 크기인 18000으로 나누어 백분율을 계산
    total_percent_min_frames = {k: sum(v) / 18000 * 100 for k, v in grouped_min_frames.items()}
    sorted_keys = sorted(total_percent_min_frames.keys())
    x_values = [k for k in sorted_keys]
    y_values = [total_percent_min_frames[k] for k in sorted_keys]

    # 데이터를 기반으로 막대 그래프 그리기
    plt.figure(figsize=(12, 7))
    bars = plt.bar(x_values, y_values, color='skyblue', label='Total Percent of Min Frames', width=0.5)

    # 각 막대에 값 표시
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{round(yval, 2)}%', ha='center', va='bottom', color='dimgrey')

    plt.title('Total Percentage of Minimum Frames by Chunk Start')
    plt.xlabel('Chunk Start (minutes)')
    plt.ylabel('Total Percentage of Min Frames (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # x축 레이블 설정
    plt.xticks(x_values, [f'{k*10}' for k in x_values])  # x축 레이블을 0, 10, 20, ... (각 레이블은 10분 간격을 나타냄)
    # plt.yticks(np.arange(0, 11, 1))

    # 그래프를 파일로 저장
    plt.savefig('total_min_frames_percentage_by_chunk_start.png')


draw_bar_graph()
# draw_bar_group()
# draw_line_graph()
