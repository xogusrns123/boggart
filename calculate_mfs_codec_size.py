import os
import subprocess
import json

def get_video_info(video_path):
    command = [
        'ffprobe', '-v', 'error', '-print_format', 'json',
        '-show_format', '-show_streams', video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(result.stdout)

def get_bitrate_duration(video_info):
    format_info = video_info['format']
    bitrate = int(format_info['bit_rate'])
    duration = float(format_info['duration'])
    return bitrate, duration

def calculate_video_size_from_bitrate(video_path):
    video_info = get_video_info(video_path)
    bitrate, duration = get_bitrate_duration(video_info)
    video_size = bitrate * duration  # Keep in bits
    return video_size, bitrate, duration

def process_videos_in_directory(directory_path):
    video_files = [f for f in os.listdir(directory_path) if f.endswith(('.mp4', '.mkv', '.avi', '.mov'))]
    results = {}

    for video_file in video_files:
        video_path = os.path.join(directory_path, video_file)
        try:
            video_size, bitrate, duration = calculate_video_size_from_bitrate(video_path)
            new_duration = 5
            results[video_file] = (video_size, bitrate, duration, new_duration, video_size / new_duration)
            # print(f'Total calculated size for {video_file}: {video_size} bits')
            # print(f'Bitrate: {bitrate} bps, Duration: {duration} seconds')
            # print(f'new Bitrate: {video_size / new_duration} bps')
        except Exception as e:
            print(f'Error processing {video_file}: {e}')

    return results

def calculate_average_bitrate(results):
    total_bitrate = sum(bitrate for _, bitrate, _, _, _ in results.values())
    total_new_bitrate = sum(new_bitrate for _, _, _, _, new_bitrate in results.values())
    average_bitrate = total_bitrate / len(results) if results else 0
    average_bitrate_new = total_new_bitrate / len(results) if results else 0
    return average_bitrate, average_bitrate_new

# Example usage
directory_path = '/home/kth/rva/video/mfs_4000k_0.90_0'
results = process_videos_in_directory(directory_path)
average_bitrate, average_bitrate_new = calculate_average_bitrate(results)
print(average_bitrate, average_bitrate_new)
# print(f'Results: {results}')
