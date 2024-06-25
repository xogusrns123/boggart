import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
from visuallize import ResultProcessor
from VideoData import VideoData


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
end_frame = 0
vd = VideoData(video_name, 10)
processor = ResultProcessor(vd, video_path, output_video_path, trajectory_dir, boggart_result_dir, query_result_dir, yolo_path, _type, end_frame)


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
    # print(df)
    # percent_min_frames 열 추가
    df['percent_min_frames'] = (df['min_frames'] / 1800) * 100
    df['bandwidth'] = df['min_frames'] *6000000
    print(df['bandwidth'])
    # min_frames의 총합 계산
    total_min_frames = df['min_frames'].sum()
    print(total_min_frames*6000000/108000*30)
    
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
    plt.gcf().text(0.92, 0.95, f'Total min_frames(%): {round(total_min_frames / 108000 * 100,3)}', 
                fontsize=12, color='green', horizontalalignment='right', verticalalignment='top')


    # 그래프를 파일로 저장
    plt.savefig('min_frames_percentage.png')

draw_bar_graph()
# import pandas as pd
# import json
# import glob

# # CSV 파일들이 있는 디렉토리 경로
# csv_directory = '/home/kth/rva/boggart/data/auburn_first_angle10/trajectories/'
# # 모든 CSV 파일 경로 가져오기
# csv_files = glob.glob(csv_directory + '*.csv')
# # print(csv_files)
# # 모든 CSV 파일을 읽어들여 하나의 DataFrame으로 합치기
# combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# # DataFrame을 딕셔너리 리스트로 변환
# detection_infos = combined_df.to_dict(orient='records')

# # 딕셔너리 리스트를 JSON 형식으로 변환
# detection_infos_json = json.dumps(detection_infos)

# # JSON 문자열의 크기 계산 (바이트 단위)
# infos_size = len(detection_infos_json.encode('utf-8'))

# # 결과 출력
# # print(f"Detection infos JSON: {detection_infos_json}")
# print(f"Size of detection infos: {infos_size} bytes")
