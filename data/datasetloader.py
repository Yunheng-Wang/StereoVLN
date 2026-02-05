import torch.utils.data as data
import torch
import os
import random
import json
import logging
from data.utils.load import load_cur_rgb, load_history_rgb_right, load_depth_left, load_label_points

logger = logging.getLogger(__name__)

class Dataset_Normal(data.Dataset):
    def __init__(self, config):
        self.valid_depth = config.main.valid_depth
        self.dataset_root = os.path.join(config.main.data_root, "train")
        self.predict_num = config.main.prediction_steps
        self.history_num = config.main.history_steps
        self.image_size = (448, 448)
        self.actionsmapping = {
            '0': '<action>Stop</action>',
            '1': "<action>Move Forward</action>",
            '2': "<action>Turn Left</action>",
            '3': "<action>Turn Right</action>",
        }
        self.num_episodes = None
        self.all_episodes = self._load_episodes()


    def _load_episodes(self):
        episodes = []
        # 1. 遍历 dataset_root 下的所有文件夹
        for folder_name in os.listdir(self.dataset_root):
            folder_path = os.path.join(self.dataset_root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            # 2. 读取 summary.json 文件
            summary_path = os.path.join(folder_path, "summary.json")
            if not os.path.exists(summary_path):
                continue
            # 3. 读取每一行 trajectory
            with open(summary_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:  
                        continue
                    trajectory = json.loads(line)
                    # 4. 提取基本信息
                    video_path = os.path.join(folder_path, trajectory['video'])
                    instruction = trajectory['instructions'][0]  # 取第一条指令
                    actions = trajectory['actions']
                    actions.append(0)  
                    # 5. 根据 predict_num 切分 trajectory
                    num_frames = len(actions)  
                    current_frame_idx = 0
                    # 6. 取出 episode
                    while current_frame_idx < num_frames - 1:  
                        action_start_idx = current_frame_idx + 1
                        episode = {
                            "instruction": instruction,
                            "actions": actions,
                            "video_path": video_path,
                            "init_obers_idx": current_frame_idx,
                            "init_action_idx": action_start_idx,
                        }
                        episodes.append(episode)
                        current_frame_idx += self.predict_num
        self.num_episodes = len(episodes)
        logger.info(f"Total amount of data: {len(episodes)}")
        return episodes
    

    def __len__(self):
        return len(self.all_episodes)


    def __getitem__(self, idx):
        # 1. 选择 episode
        episode = self.all_episodes[idx]
        init_frame_idx = episode['init_obers_idx']
        video_folder = episode['video_path']  
        # 2. 获取指令
        instruction = episode['instruction']
        # 3. 获取左右视角的初始帧
        left_current_frame, right_current_frame = load_cur_rgb(init_frame_idx, video_folder)
        # 4. 获取右视角的历史帧
        right_history_video = load_history_rgb_right(init_frame_idx, self.history_num, video_folder)
        # 5. 获取深度图像 [1, 448, 448], 单位：毫米
        label_depth = load_depth_left(init_frame_idx, video_folder, self.valid_depth)
        # 6. 获取左右点标签
        label_left_point, label_right_point = load_label_points(init_frame_idx, video_folder)
        # 7. 获取历史动作
        init_action_idx = episode['init_action_idx']
        actions = episode['actions']
        if init_frame_idx == 0:
            # 没有历史动作
            history_action = "This is the initial timestep, so no previous action sequence is available."
        else:
            history_action_indices = actions[1:init_action_idx]
            history_action_strs = [self.actionsmapping[str(action)] for action in history_action_indices]
            history_action = "".join(history_action_strs)
        # 8. 获取动作标签
        action_end_idx = min(init_action_idx + self.predict_num, len(actions))
        label_action_indices = actions[init_action_idx:action_end_idx]
        label_action_strs = [self.actionsmapping[str(action)] for action in label_action_indices]
        label_answer = "".join(label_action_strs)
        # 9. 返回数据
        return {
            "instruction": instruction,
            "history_action": history_action,
            "left_current_frame": left_current_frame,
            "right_current_frame": right_current_frame,
            "right_history_video": right_history_video,
            "label_left_point": label_left_point,
            "label_right_point": label_right_point,
            "label_depth": label_depth,
            "label_answer": label_answer,
        }
        