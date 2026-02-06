import sys
import os
import re
import tqdm
import torch
import copy
import json
import random
import argparse
import itertools
import quaternion
import numpy as np

from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict
import torchvision.transforms as transforms

import habitat
from habitat import logger, Env
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from utils.dist import *
from model_client import StereoVLNClient

# 导入自定义测量器以注册到 Habitat registry
from data.utils import measures  # noqa: F401


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)

        # 配置双目摄像机（参考 gen_trajectory.py）
        with habitat.config.read_write(self.config):
            OmegaConf.set_struct(self.config, False)
            self.config.habitat.dataset.split = self.split

            # 左摄像机 RGB 配置
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_left = OmegaConf.create({
                "type": "HabitatSimRGBSensor",
                "uuid": "rgb_left",
                "width": 448,
                "height": 448,
                "hfov": 79,
                "position": [-0.05, 1.25, 0.0],  # 左眼，向左偏移5cm
                "orientation": [0.0, 0.0, 0.0]
            })
            # 右摄像机 RGB 配置
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_right = OmegaConf.create({
                "type": "HabitatSimRGBSensor",
                "uuid": "rgb_right",
                "width": 448,
                "height": 448,
                "hfov": 79,
                "position": [0.05, 1.25, 0.0],  # 右眼，向右偏移5cm
                "orientation": [0.0, 0.0, 0.0]
            })

            # 添加测量配置
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
            OmegaConf.set_struct(self.config, True)

        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

        self.model = model
        self.history_num = args.history_num
        self.prediction_steps = args.prediction_steps
        self.image_size = (448, 448)

        # StereoVLN action mapping
        self.actions2idx = OrderedDict({
            'stop': 0,
            'move forward': 1,
            'turn left': 2,
            'turn right': 3,
            'move backward': 4,
        })
        self.actionsmapping = {
            0: '<action>Stop</action>',
            1: "<action>Move Forward</action>",
            2: "<action>Turn Left</action>",
            3: "<action>Turn Right</action>",
            4: "<action>Move Backward</action>",
        }

        self.transform = transforms.ToTensor()

    def preprocess_rgb(self, rgb_image):
        """Preprocess RGB image to [1, 3, 448, 448] tensor with values 0~255"""
        image = Image.fromarray(rgb_image).convert('RGB')
        image = image.resize(self.image_size, Image.BILINEAR)
        tensor = self.transform(image) * 255.0
        return tensor.unsqueeze(0)  # [1, 3, 448, 448]

    def get_stereo_images(self, observations):
        """
        Get stereo images from habitat observations.
        Uses dual RGB sensors configured for left and right eyes.
        """
        rgb_left = observations["rgb_left"]
        rgb_right = observations["rgb_right"]
        left_frame = self.preprocess_rgb(rgb_left)
        right_frame = self.preprocess_rgb(rgb_right)
        return left_frame, right_frame

    def build_history_video(self, history_frames):
        """
        Build history video tensor from a list of RGB tensors.
        Returns [history_num, 3, 448, 448] tensor with zero padding if needed.
        """
        num_actual = len(history_frames)
        if num_actual < self.history_num:
            img_shape = (3, self.image_size[0], self.image_size[1])
            num_padding = self.history_num - num_actual
            padding_frames = [torch.zeros(img_shape) for _ in range(num_padding)]
            history_frames = padding_frames + history_frames
        elif num_actual > self.history_num:
            # Uniform sampling
            step = num_actual / self.history_num
            indices = [int(i * step) for i in range(self.history_num)]
            history_frames = [history_frames[i] for i in indices]

        return torch.stack(history_frames, dim=0)  # [history_num, 3, 448, 448]

    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env

    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'), 'r') as f:
                for line in f.readlines():
                    try:
                        res = json.loads(line)
                        if "scene_id" in res:  # Skip summary lines
                            done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                            if get_rank() == 0:
                                sucs.append(res['success'])
                                spls.append(res['spl'])
                                oss.append(res['os'])
                                ones.append(res['ne'])
                    except:
                        continue

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")

            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start", episode_instruction)
                episode_id = episode.episode_id

                if [scene_id, episode_id, episode_instruction] in done_res:
                    process_bar.update(1)
                    continue

                env.current_episode = episode
                observations = env.reset()
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb_left']).save(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg'))

                vis_frames = []
                step_id = 0

                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}'), exist_ok=True)

                initial_height = env.sim.get_agent_state().position[1]

                # History tracking
                right_history_frames = []  # List of [3, 448, 448] tensors
                executed_actions = []  # List of action indices
                action_seq = []  # Pending actions to execute

                while not env.episode_over:
                    self.model.eval()

                    # Get stereo images
                    left_frame, right_frame = self.get_stereo_images(observations)

                    # Visualization
                    info = env.get_metrics()
                    if info['top_down_map'] is not None:
                        frame = observations_to_image({'rgb': observations['rgb_left']}, info)
                        vis_frames.append(frame)

                    # Generate new actions if queue is empty
                    if len(action_seq) == 0:
                        # Build history video
                        right_history_video = self.build_history_video(right_history_frames)

                        # Build history action string
                        if step_id == 0:
                            history_action = "This is the initial timestep, so no previous action sequence is available."
                        else:
                            history_action_strs = [self.actionsmapping[a] for a in executed_actions]
                            history_action = "".join(history_action_strs)

                        # Prepare inputs for model
                        left_current_frame = left_frame.unsqueeze(0).to(self.device, dtype=torch.bfloat16)  # [1, 1, 3, 448, 448]
                        right_current_frame = right_frame.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
                        right_history_video_tensor = right_history_video.unsqueeze(0).to(self.device, dtype=torch.bfloat16)  # [1, history_num, 3, 448, 448]

                        # Run inference
                        outputs = self.model.inference(
                            instruction=[episode_instruction],
                            history_action=[history_action],
                            left_current_frame=left_current_frame,
                            right_current_frame=right_current_frame,
                            right_history_video=right_history_video_tensor,
                            max_new_tokens=128,
                        )

                        generated_text = outputs['generated_text'][0]
                        print(f"Step {step_id}, Generated: {generated_text}", flush=True)

                        # Parse actions from generated text
                        action_seq = self.parse_actions(generated_text)
                        print(f"Parsed actions: {action_seq}", flush=True)

                        if len(action_seq) == 0:
                            # If no valid actions parsed, default to stop
                            action_seq = [0]

                    # Execute action
                    action = action_seq.pop(0)
                    executed_actions.append(action)

                    # Update history (add current right frame before action)
                    right_history_frames.append(right_frame.squeeze(0))  # [3, 448, 448]

                    # Step environment
                    observations = env.step(action)
                    step_id += 1

                process_bar.update(1)
                metrics = env.get_metrics()

                if self.save_video:
                    images_to_video(
                        vis_frames, os.path.join(self.output_path, f'vis_{self.epoch}'), f'{scene_id}_{episode_id}', fps=6, quality=9
                    )
                vis_frames.clear()

                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])

                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }

                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)

    def parse_actions(self, output):
        """
        Parse action sequence from model output.
        Expected format: <action>Move Forward</action><action>Turn Left</action>...
        """
        # Find all action tags
        pattern = r'<action>(.*?)</action>'
        matches = re.findall(pattern, output, re.IGNORECASE)

        actions = []
        for match in matches:
            action_text = match.strip().lower()
            if action_text in self.actions2idx:
                actions.append(self.actions2idx[action_text])

        return actions


def load_model(args):
    """Connect to remote StereoVLN server."""
    print(f"Connecting to remote StereoVLN server at {args.server_url}")
    return StereoVLNClient(server_url=args.server_url)


def evaluate(model, args):
    model.eval()

    world_size = get_world_size()
    os.makedirs(args.output_path, exist_ok=True)

    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args
    )

    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank())

    # Gather results from all processes
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)

    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]

    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()

    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)

    result_all = {
        "success_rate": (sum(sucs_all) / len(sucs_all)).item(),
        "spl": (sum(spls_all) / len(spls_all)).item(),
        "oracle_success": (sum(oss_all) / len(oss_all)).item(),
        "navigation_error": (sum(ones_all) / len(ones_all)).item(),
        'num_episodes': len(sucs_all)
    }

    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))




def eval():
    global local_rank
    parser = argparse.ArgumentParser()

    parser.add_argument("--server_url", type=str, default="http://localhost:5000", help="URL of remote StereoVLN server")
    parser.add_argument("--habitat_config_path", type=str, default='/home/CONNECT/yfang870/yunhengwang/StereoVLN/config/gen_data_r2r.yaml')
    parser.add_argument("--output_path", type=str, default='./results')
    parser.add_argument("--eval_split", type=str, default='val_unseen')


    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--history_num", type=int, default=8)
    parser.add_argument("--prediction_steps", type=int, default=4)

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--gpu', default=0, type=int, help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank

    # Load model client
    model = load_model(args)

    # Run evaluation
    evaluate(model, args)


if __name__ == "__main__":
    eval()
