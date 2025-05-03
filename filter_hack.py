from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn as nn
import cv2
import json
import tqdm
import sys
sys.path.insert(0, '/data/boran/4dhoi/human_motion/GVHMR/hamer/third-party/ViTPose')

from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result

os.environ["PYOPENGL_PLATFORM"] = "egl"

# project root directory
ROOT_DIR = "/data/boran/4dhoi/human_motion/GVHMR/hamer"
VIT_DIR = os.path.join(ROOT_DIR, "third-party/ViTPose")

class ViTPoseModel(object):
    MODEL_DICT = {
        'ViTPose+-G (multi-task train, COCO)': {
            'config': f'{VIT_DIR}/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_coco_256x192.py',
            'model': f'{ROOT_DIR}/_DATA/vitpose_ckpts/vitpose+_huge/vitpose-h.pth',
        },
    }

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.model_name = 'ViTPose+-G (multi-task train, COCO)'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = dic['model']
        model = init_pose_model(dic['config'], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        image = image[:, :, ::-1]  # RGB -> BGR
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1]  # BGR -> RGB

def segment_video(video_name, input_video, output_dir, mask):
    # 打开视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {input_video}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    segments = []      # 存储所有连续片段（每个片段为一帧列表）
    current_segment = []  # 当前片段帧列表

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 判断是否保留当前帧（这里基于亮度条件，也可以用帧索引等其它逻辑）
        if mask[frame_index]:
            current_segment.append(frame)
        else:
            # 当前帧不满足条件，如果当前片段长度满足要求，则保存为一个独立片段
            print()
            if len(current_segment) >= 50:
                segments.append(current_segment)
            # 重置当前片段
            current_segment = []
        
        frame_index += 1
    
    # 处理视频结束后还未结束的最后一个片段
    if len(current_segment) >= 50:
        segments.append(current_segment)
    
    cap.release()
    
    # 保存片段为新视频
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    segment_files = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i, segment in enumerate(segments):
        output_file = os.path.join(output_dir, f"{video_name}_{i:03d}.mp4")
        writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        for frame in segment:
            writer.write(frame)
        writer.release()
        segment_files.append(output_file)
        print(f"片段 {i} 已保存到: {output_file}")
    # os.remove(os.path.join(video_dir, f'{video_name}.mp4'))
    
    return segment_files

def generate_video(frames):
    fps = 30
    height, width = frames[0].shape[:2]
    print(fps, height, width)
    print(len(frames))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    for idx, frame in enumerate(frames):
        out.write(frame)

    out.release()
    print('视频写入完成：output.mp4')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video frames")
    parser.add_argument("--obj_type", type=str, required=True, help="Object type to detect and track")
    video_dir = parser.parse_args().video_dir
    obj_type = parser.parse_args().obj_type
    device = 'cuda:0'
    model = ViTPoseModel(device)
    result = []

    for video_name in os.listdir(video_dir):
        video_name = "motorcycle_1__6_-Scene-002_2.mp4"
        if video_name.endswith(".mp4"):
            name = video_name.split('.')[0]
            frame_path = f'{video_dir}/frame_list_{name}'
            img_path = os.listdir(frame_path)
            img_path.sort()
            
            json_path = f'{video_dir}/{name}_result.json'
            with open(json_path, 'r') as f:
                results = json.load(f)
                obj_mask = results['mask']
                det_results = [np.array(results['bboxes']).reshape(-1, 5)]

            ans = 0
            mask = []
            viss = []
            for i, img_name in enumerate(tqdm.tqdm(img_path, desc="Processing images")):
                # img = cv2.imread('/data/boran/4dhoi/video_cut/video/barbell/frame_list_barbell (4)/000001.jpg')
                img = cv2.imread(os.path.join(frame_path, img_name))
                
                    # print(det_results.shape)
                # det_results = [np.array([[0., 520., 250., 875., 0.99]])]

                box_score_threshold = 0.5
                kpt_score_threshold = 0.3
                vis_dot_radius = 4
                vis_line_thickness = 2
                single_bbox = det_results[0][i].reshape(1, 5)
                result = model.predict_pose(img, [single_bbox])
                # print(result)
                confidences = result[0]['keypoints'][:, 2]
                count = np.sum(confidences > 0.6)
                # print(count)
                if(count >= 7):
                    mask.append(True)
                else:
                    mask.append(False)

                print(img_name)
                img, vis = model.predict_pose_and_visualize(img, [single_bbox], box_score_threshold,
                                                            kpt_score_threshold, vis_dot_radius,
                                                            vis_line_thickness)
                viss.append(vis)
                ans += 1
            generate_video(viss)
            # combined_mask = [human and obj for human, obj in zip(mask, obj_mask)] 
            # print(mask)
            # segment_video(name, os.path.join(video_dir, f'{video_name}'), f"/data/boran/4dhoi/Dataset/refined_video/{obj_type}/", combined_mask)