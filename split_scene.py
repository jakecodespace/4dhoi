from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import os
import argparse
import cv2
from moviepy import VideoFileClip
import re

def split_video_into_scenes(video_path, output_dir, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    split_video_ffmpeg(video_path, scene_list, output_dir, show_progress=True)
    return scene_list.__len__()

def clean_filename(filename):
    cleaned_filename = re.sub(r'[ \(\)]', '_', filename)
    return cleaned_filename
def split_video(input_file, max_duration=25):
    # 读取视频文件
    video_dir  = os.path.dirname(input_file)
    video_name = os.path.splitext(os.path.basename(input_file))[0]
    cleaned_filename = clean_filename(video_name)
    video = VideoFileClip(input_file)
    
    # 获取视频总时长
    total_duration = video.duration
    print(f"Video duration: {total_duration:.2f} seconds")
    
    # 如果视频长度大于 max_duration, 切割成多个片段
    if total_duration <= max_duration:
        return
    else:
        num_parts = int(total_duration // max_duration)  # 计算分割的片段数
        for i in range(num_parts):
            start_time = i * max_duration
            end_time = (i + 1) * max_duration
            # 裁剪视频片段
            clip = video.subclipped(start_time, end_time)
            output_filename = f"{video_dir}/{cleaned_filename}_{i+1}.mp4"
            clip.write_videofile(output_filename, codec="mpeg4", fps=clip.fps)
            print(f"Saved {output_filename}")
        
        # 处理视频最后剩下的部分
        if total_duration % max_duration != 0:
            start_time = num_parts * max_duration
            if(total_duration -  start_time > 2):
                clip = video.subclipped(start_time, total_duration)
                output_filename = f"{video_dir}/{cleaned_filename}_{num_parts+1}.mp4"
                clip.write_videofile(output_filename, codec="mpeg4", fps=clip.fps)
                print(f"Saved {output_filename}")
        os.remove(input_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video frames")
    video_dir = parser.parse_args().video_dir

    for video in os.listdir(video_dir):
        if video.endswith(".mp4"):
            print(video)
            full_path = video_dir + "/" + video
            scene = split_video_into_scenes(full_path, video_dir)
            if scene != 0:
                os.remove(video_dir + "/" + video)
                
    for video in os.listdir(video_dir):
        print(video)
        if video.endswith(".mp4"):
            split_video(video_dir + "/" + video)

# conda activate sam2
# python split_scene.py --video_dir /data/boran/4dhoi/video_cut/video/motorcycle
    
        
    
