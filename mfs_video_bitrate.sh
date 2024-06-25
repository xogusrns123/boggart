#!/bin/bash

# 비디오 파일이 저장된 디렉토리
# crf조절
# crf_value=$1
# video_input_dir="/home/kth/rva/video/mfs_0.95_0"
# video_output_dir="/home/kth/rva/video/mfs_crf${crf_value}_0.95_0"

# # 출력 디렉토리가 없으면 생성
# mkdir -p "$video_output_dir"

# # 모든 비디오 파일에 대해 FFmpeg를 사용하여 비트레이트 조정
# for video_file in "$video_input_dir"/*.mp4; do
#     filename=$(basename -- "$video_file")
#     output_file="$video_output_dir/$filename"
    
#     ffmpeg -i "$video_file" -preset veryfast -crf "$crf_value" "$output_file"
    
#     echo "Processed $video_file to $output_file with crf $crf_value"
# done

# bitrate 조절
bitrate=$1
video_input_dir="/home/kth/rva/video/mfs_0.90_0"
video_output_dir="/home/kth/rva/video/mfs_${bitrate}_0.90_0"

# 출력 디렉토리가 없으면 생성
mkdir -p "$video_output_dir"

# 모든 비디오 파일에 대해 FFmpeg를 사용하여 비트레이트 조정
for video_file in "$video_input_dir"/*.mp4; do
    filename=$(basename -- "$video_file")
    output_file="$video_output_dir/$filename"
    
    ffmpeg -i "$video_file" -b:v "$bitrate" "$output_file"
    
    echo "Processed $video_file to $output_file with bitrate $bitrate"
done
