import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import json


def load_cur_rgb(idx, video_folder):
    frame_filename = f"{idx + 1:03d}.jpg"
    left_image_path = os.path.join(video_folder, 'rgb_left', frame_filename)
    right_image_path = os.path.join(video_folder, 'rgb_right', frame_filename)
    left_image = Image.open(left_image_path).convert('RGB')
    right_image = Image.open(right_image_path).convert('RGB')
    transform = transforms.ToTensor()
    left_tensor = transform(left_image) * 255.0
    right_tensor = transform(right_image) * 255.0
    left_current_frame = left_tensor.unsqueeze(0)
    right_current_frame = right_tensor.unsqueeze(0)
    return left_current_frame, right_current_frame


def load_history_rgb_right(init_frame_idx, history_num, video_folder):
    # 1. 确定可用的历史帧索引范围 [0, init_frame_idx-1]
    available_indices = list(range(0, init_frame_idx)) if init_frame_idx > 0 else []
    # 2. 均匀采样历史帧索引
    if len(available_indices) == 0:
        sampled_indices = []
    elif len(available_indices) <= history_num:
        sampled_indices = available_indices
    else:
        step = len(available_indices) / history_num
        sampled_indices = [int(i * step) for i in range(history_num)]
    # 3. 加载采样的历史帧
    transform = transforms.ToTensor()
    history_frames = []
    for idx in sampled_indices:
        frame_filename = f"{idx + 1:03d}.jpg"
        right_image_path = os.path.join(video_folder, 'rgb_right', frame_filename)
        right_image = Image.open(right_image_path).convert('RGB')
        right_tensor = transform(right_image) * 255.0
        history_frames.append(right_tensor)
    # 4. 如果帧数不足 history_num，在前面补零
    num_actual_frames = len(history_frames)
    if num_actual_frames < history_num:
        if num_actual_frames > 0:
            img_shape = history_frames[0].shape 
        else:
            img_shape = (3, 448, 448) 
        num_padding = history_num - num_actual_frames
        padding_frames = [torch.zeros(img_shape) for _ in range(num_padding)]
        history_frames = padding_frames + history_frames
    # 5. 堆叠成 [history_num, 3, 448, 448]
    history_frames_tensor = torch.stack(history_frames, dim=0)
    return history_frames_tensor


def load_depth_left(idx, video_folder, valid_depth):
    # 1. 构造深度图像路径（.png 格式）
    frame_filename = f"{idx + 1:03d}.png"
    depth_image_path = os.path.join(video_folder, 'depth_left', frame_filename)
    # 2. 加载深度图像（uint16, 单位：毫米）
    depth_image = Image.open(depth_image_path)
    depth_array = np.array(depth_image, dtype=np.float32)
    # 3. 转换单位：毫米 -> 米
    depth_array = depth_array / 1000.0
    # 4. 将大于 3m 的像素设为 inf（无效值，会被 valid mask 过滤）
    depth_array[depth_array > valid_depth] = np.inf
    # 5. 转换为 tensor 并添加 channel 维度 [448, 448] -> [1, 448, 448]
    depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)
    return depth_tensor


def load_label_points(idx, video_folder):
    # 1. 加载 label_points.json 文件
    label_points_path = os.path.join(video_folder, 'label_points.json')
    with open(label_points_path, 'r') as f:
        label_points_data = json.load(f)
    # 2. 找到对应的帧（frame_id 从 1 开始，idx 从 0 开始）
    target_frame_id = idx + 1
    for frame_data in label_points_data:
        if frame_data['frame_id'] == target_frame_id:
            # 3. 提取点坐标并转换为 tensor
            label_left_point = torch.tensor(frame_data['label_left_point'], dtype=torch.float32)
            label_right_point = torch.tensor(frame_data['label_right_point'], dtype=torch.float32)
            return label_left_point, label_right_point