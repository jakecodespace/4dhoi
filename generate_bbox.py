import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append(os.path.join(os.path.dirname(__file__), '../4dhoi_object_tracking/GroundingDINO/'))
from groundingdino_api import get_object_bbox

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and object type.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video frames")
    parser.add_argument("--object_name", type=str, required=True, help="Type of object to detect")
    parser.add_argument("--start_frame", type=int, required=True, help="Starting frame index")
    args = parser.parse_args()

    video_list = os.listdir(args.video_dir)
    video_list = [video for video in video_list if video.endswith('.mp4')]
    print(video_list)
    for video_name in tqdm.tqdm(video_list):

        video_name = os.path.splitext(video_name)[0]
        print(video_name)
        end_frame = get_video_frame_count(args.video_dir+f'{video_name}.mp4')-1
        print(end_frame)
            
        video_dir = args.video_dir
        object_type = args.object_name

        ann_frame_idx = 0
        ann_obj_id = 0

        ## process video:p
        if not os.path.exists(video_dir+f'/frame_list_{video_name}/'):
            os.makedirs(video_dir+f'/frame_list_{video_name}/')

        cap = cv2.VideoCapture(video_dir+f'{video_name}.mp4')
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number < args.start_frame:
                continue
            output_path = os.path.join(video_dir+f'/frame_list_{video_name}/', f"{frame_number:06d}.jpg")
            cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frame_number += 1

            if frame_number > end_frame:
                break

        cap.release()
        try:
            bbox = get_object_bbox(video_dir,os.path.join(video_dir, f'frame_list_{video_name}/000000.jpg'), object_type)
            human_bbox=get_object_bbox(video_dir,os.path.join(video_dir, f'frame_list_{video_name}/000000.jpg'), 'person')
        except Exception as e:
            os.remove(os.path.join(video_dir, f'{video_name}.mp4'))
            continue
        box = np.asarray(bbox).astype(np.float32).tolist()
        human_box = np.asarray(human_bbox).astype(np.float32).tolist()
        with open(os.path.join(video_dir, f'bbox_{video_name}.json'), 'w') as f:
            json.dump(box, f)
        with open(os.path.join(video_dir, f'human_bbox_{video_name}.json'), 'w') as f:
            json.dump(human_box, f)
 # python 4dhoi_object_tracking/generate_bbox.py --video_dir "/data/boran/4dhoi/Dataset/fan" --object_name "fab" --start_frame 0 --end_frame 30