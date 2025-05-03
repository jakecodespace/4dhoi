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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(os.path.join(os.path.dirname(__file__), '../4dhoi_object_tracking/sam2_api/'))
from sam2.build_sam import build_sam2_video_predictor

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "/data/boran/4dhoi/4dhoi_object_tracking/sam2_api/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def save_masks_as_images(frame_names, out_mask_logits, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ann_frame_idx, frame_name in enumerate(frame_names):
        mask = out_mask_logits[ann_frame_idx].astype(np.float32)
        mask_image = (mask > 0.0).astype(np.uint8) * 255
        mask_image = mask_image.squeeze(0)
        output_path = os.path.join(output_dir, f"{os.path.splitext(frame_name)[0]}_mask.png")
        Image.fromarray(mask_image).save(output_path)

def save_mask_video(video_dir, frame_names, out_mask_logits, output_video_path, fps=30, frame_size=(1280, 720)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for ann_frame_idx, frame_name in enumerate(frame_names):
        frame_path = os.path.join(f'{video_dir}/frame_list', frame_name)
        frame = np.array(Image.open(frame_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mask = (out_mask_logits[ann_frame_idx] > 0.0).astype(np.uint8)
        cmap = plt.get_cmap("tab10")
        color = np.array([*cmap(0)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        overlay = frame_rgb * (1 - mask_image[:, :, 3:]) + mask_image[:, :, :3] * mask_image[:, :, 3:]

        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        video_writer.write(overlay_bgr)

    video_writer.release()

def segment_video(video_name, input_video, output_dir, mask):
    # 打开视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {input_video}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
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
            if len(current_segment) >= fps:
                segments.append(current_segment)
            # 重置当前片段
            current_segment = []
        
        frame_index += 1
    
    # 处理视频结束后还未结束的最后一个片段
    if len(current_segment) >= fps:
        segments.append(current_segment)
    
    cap.release()
    
    # 保存片段为新视频
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    segment_files = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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

def get_white_bbox(mask):
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("输入必须为单通道二值化矩阵 (H x W)")
    
    # 提取白色像素坐标
    white_pixels = np.argwhere(mask == 255)
    if len(white_pixels) == 0:
        return None
    
    # 计算极值坐标（注意行列与xy的对应关系）
    y1, x1 = np.min(white_pixels, axis=0)  # 左上角坐标
    y2, x2 = np.max(white_pixels, axis=0)  # 右下角坐标
    
    return x1, y1, x2, y2  # OpenCV坐标系格式


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and object type.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video frames")

    # parser.add_argument("--object_name", type=str, required=True, help="Type of object to detect")
    args = parser.parse_args()

    video_dir = args.video_dir
    # object_type = args.object_name
    video_list = os.listdir(video_dir)
    video_list = [video for video in video_list if video.endswith('.mp4')]
    # video_list = ["motorcycle_1__6_-Scene-002_1.mp4"]
    
    obj_type = video_dir.split('/')[-1]
    output_dir = f"/data/boran/4dhoi/Dataset/refined_video/{obj_type}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_list = os.listdir(output_dir)

    for video_name in tqdm.tqdm(video_list):
        print(video_name)
        
        video_name = os.path.splitext(video_name)[0]

        if any(file.startswith(video_name) for file in output_list):
            continue
        ann_frame_idx = 0
        ann_obj_id = 0

        ## process video:
        # if not os.path.exists(video_dir+'/frame_list/'):
        #     os.makedirs(video_dir+'/frame_list/')
        
        #     cap = cv2.VideoCapture(video_dir+'/output.mp4/')
        #     frame_number = 0
        
        #     while cap.isOpened():
        #         ret, frame = cap.read()
        #         if not ret:
        #             break
        #         output_path = os.path.join(video_dir+'/frame_list/', f"{frame_number:05d}.jpg")
        #         cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        #         frame_number += 1
        
        #     cap.release()


        # bbox = get_object_bbox(os.path.join(video_dir, 'frame_list/00000.jpg'), object_type)
        bbox=json.load(open(os.path.join(video_dir, f'bbox_{video_name}.json'), 'r'))
        box = np.asarray(bbox).astype(np.float32)
        # print(box)

        # human_bbox=get_object_bbox(os.path.join(video_dir, 'frame_list/00000.jpg'), 'person')
        human_bbox=json.load(open(os.path.join(video_dir, f'human_bbox_{video_name}.json'), 'r'))
        human_box = np.asarray(human_bbox).astype(np.float32)

        frame_names = [p for p in os.listdir(video_dir+f'/frame_list_{video_name}/') if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir+f'/frame_list_{video_name}/', offload_state_to_cpu = True)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
        )

        mask_obj = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # print(out_mask_logits)
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            if(mask.sum() == 0):
                mask_obj.append(False)
            else:
                mask_obj.append(True)
        print('done')

        # if not os.path.exists(os.path.join(video_dir, "./mask_dir")):
        #     os.makedirs(os.path.join(video_dir, "./mask_dir"))
        # save_masks_as_images(frame_names, video_segments, os.path.join(video_dir, "./mask_dir"))



        inference_state = predictor.init_state(video_path=video_dir+f'/frame_list_{video_name}/', offload_state_to_cpu = True)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=human_box,
        )
        mask_human = []
        tracking_bboxes = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            mask = mask.squeeze()
            mask_uint8 = (mask.astype(np.uint8) * 255)

            if(mask.sum() == 0):
                mask_human.append(False)
                print("No mask found")
                tracking_bboxes.append((0,0,0,0, 0.99))
            else:
                x1, y1, x2, y2 = get_white_bbox(mask_uint8)
                tracking_bboxes.append((
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                    float(0.99)
                ))
            
        print('done')
        result = {
            "mask": mask_obj,
            "bboxes": tracking_bboxes,
        }

        json_dir = video_dir + f"/{video_name}_result.json"
        with open(json_dir, 'w') as f:
            json.dump(result, f)

        # combined_mask = [human and obj for human, obj in zip(mask_human, mask_obj)]
        # print(combined_mask)

        # segment_video(video_name, os.path.join(video_dir, f'{video_name}.mp4'), f"/data/boran/4dhoi/Dataset/refined_video/{obj_type}/", combined_mask)
    # if not os.path.exists(os.path.join(video_dir, "./human_mask_dir")):
    #     os.makedirs(os.path.join(video_dir, "./human_mask_dir"))
    # save_masks_as_images(frame_names, video_segments, os.path.join(video_dir, "./human_mask_dir"))
    # save_mask_video(video_dir, frame_names, video_segments, os.path.join(video_dir, "output.mp4"), fps=30, frame_size=(1280, 720))