import habitat
import logging
import random
import json
import numpy as np
import argparse
import sys
import os
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from utils import measures
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config import read_write
import quaternion
from typing import Tuple, List
# Import custom RxR dataset to register it with habitat
from utils import rxr_dataset  # noqa: F401


def get_camera_intrinsics(width: int, height: int, hfov: float) -> np.ndarray:
    """
    Calculate camera intrinsic matrix from image dimensions and horizontal FOV.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        hfov: Horizontal field of view in degrees

    Returns:
        3x3 camera intrinsic matrix
    """
    hfov_rad = np.deg2rad(hfov)
    fx = width / (2.0 * np.tan(hfov_rad / 2.0))
    fy = fx  # Assuming square pixels
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K


def quaternion_to_rotation_matrix(q: np.quaternion) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion

    Returns:
        3x3 rotation matrix
    """
    return quaternion.as_rotation_matrix(q)


def project_3d_to_2d(
    world_point: np.ndarray,
    agent_position: np.ndarray,
    agent_rotation: np.quaternion,
    camera_offset: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    debug: bool = False
) -> Tuple[float, float, float]:
    """
    Project a 3D world point to 2D pixel coordinates in camera view.
    Uses bottom-left as origin (0,0) as specified.

    Args:
        world_point: 3D point in world coordinates [x, y, z]
        agent_position: Agent position in world coordinates [x, y, z]
        agent_rotation: Agent rotation as quaternion
        camera_offset: Camera offset from agent center [x, y, z]
        K: Camera intrinsic matrix (3x3)
        width: Image width
        height: Image height
        debug: Print debug information

    Returns:
        Tuple of (u, v, cam_x) where:
        - u, v: pixel coordinates with origin at bottom-left (None if behind camera)
        - cam_x: X coordinate in camera frame (negative=left, positive=right)
    """
    # Get rotation matrix from agent's quaternion
    R_agent = quaternion_to_rotation_matrix(agent_rotation)

    # Calculate camera position in world coordinates
    # In Habitat, camera offset is in agent's local frame
    camera_position = agent_position + R_agent @ camera_offset

    # Transform world point to camera coordinates
    point_rel = world_point - camera_position

    # Rotate point into camera frame
    # IMPORTANT: Habitat uses OpenGL convention where camera looks toward -Z
    # We need to convert to standard CV convention (camera looks toward +Z)
    point_cam_opengl = R_agent.T @ point_rel

    # Convert from OpenGL (camera looks at -Z) to CV convention (camera looks at +Z)
    # by negating the Z axis
    point_cam = point_cam_opengl.copy()
    point_cam[2] = -point_cam_opengl[2]  # Negate Z to convert from OpenGL to CV

    # cam_x is used to determine if target is on left (negative) or right (positive)
    cam_x = point_cam[0]

    if debug:
        print(f"\n=== Debug Projection ===")
        print(f"World point: {world_point}")
        print(f"Agent position: {agent_position}")
        print(f"Camera position: {camera_position}")
        print(f"Point relative: {point_rel}")
        print(f"Point in camera (OpenGL): {point_cam_opengl}")
        print(f"Point in camera (CV): {point_cam}")
        print(f"Camera X (left-/right+): {cam_x:.2f}")
        print(f"Distance: {np.linalg.norm(point_rel):.2f}m")

    # Check if point is behind camera (in CV convention, Z should be positive)
    if point_cam[2] <= 0:
        if debug:
            print(f"Point is behind camera (Z={point_cam[2]:.2f})")
        return None, None, cam_x

    # Project to image plane
    point_2d_homogeneous = K @ point_cam
    u = point_2d_homogeneous[0] / point_2d_homogeneous[2]
    v = point_2d_homogeneous[1] / point_2d_homogeneous[2]

    # Convert from top-left origin to bottom-left origin
    v_bottom_left = height - v

    if debug:
        print(f"Projected u, v (top-left): ({u:.2f}, {v:.2f})")
        print(f"Projected u, v (bottom-left): ({u:.2f}, {v_bottom_left:.2f})")

    return u, v_bottom_left, cam_x


def clip_to_image_bounds(
    u: float,
    v: float,
    width: int,
    height: int,
    cam_x: float
) -> Tuple[float, float]:
    """
    Clip pixel coordinates to image bounds.
    When target is out of view, return left/right boundary based on target's relative position.

    Args:
        u: Horizontal pixel coordinate (0 = left edge)
        v: Vertical pixel coordinate (0 = bottom edge)
        width: Image width
        height: Image height
        cam_x: X coordinate in camera frame (negative=target on left, positive=target on right)

    Returns:
        Clipped (u, v) coordinates
    """
    v_center = height / 2.0

    # 情况1：点在相机后面（投影失败，u或v为None）
    # 根据目标点在相机坐标系中的X坐标判断方位
    if u is None or v is None:
        if cam_x < 0:  # 目标在左边
            return 0.0, v_center
        else:  # 目标在右边
            return width - 1.0, v_center

    # 情况2：点在相机前方但严重超出视野范围
    # 如果投影坐标超出图像边界太多，说明点在视野外
    out_of_view_threshold = 0.2
    u_margin = width * out_of_view_threshold
    v_margin = height * out_of_view_threshold

    # 检查是否严重超出视野
    is_far_left = u < -u_margin
    is_far_right = u > width - 1 + u_margin
    is_far_out_vertically = v < -v_margin or v > height - 1 + v_margin

    # 如果点在视野外，根据目标点的相对方位返回对应边界
    if is_far_left or is_far_right or is_far_out_vertically:
        if cam_x < 0:  # 目标在左边
            return 0.0, v_center
        else:  # 目标在右边
            return width - 1.0, v_center

    # 情况3：点在视野内或接近视野边界，正常clip到图像边界
    u_clipped = np.clip(u, 0, width - 1)
    v_clipped = np.clip(v, 0, height - 1)

    return u_clipped, v_clipped


class StreamVLNHabitatRunner:
    def __init__(self, dataset: str, config_path: str, output_path: str, data_path: str = None):
        # 1. 基本配置
        self.device = torch.device("cuda")
        self.dataset = dataset.lower()
        self.config_path = config_path
        self.output_path = output_path
        self.data_path = data_path
        self.config = get_habitat_config(self.config_path)
        # 2. 双目摄像机配置
        with read_write(self.config):
            OmegaConf.set_struct(self.config, False)
            # 2.1 左摄像机RGB配置
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_left = OmegaConf.create({
                "type": "HabitatSimRGBSensor",
                "uuid": "rgb_left",
                "width": 448,
                "height": 448,
                "hfov": 79,
                "position": [-0.05, 1.25, 0.0],  # 左眼，向左偏移5cm
                "orientation": [0.0, 0.0, 0.0]
            })
            # 2.2 右摄像机RGB配置
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_right = OmegaConf.create({
                "type": "HabitatSimRGBSensor",
                "uuid": "rgb_right",
                "width": 448,
                "height": 448,
                "hfov": 79,
                "position": [0.05, 1.25, 0.0],  # 右眼，向右偏移5cm
                "orientation": [0.0, 0.0, 0.0]
            })
            # 2.3 左摄像机深度配置
            self.config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor_left = OmegaConf.create({
                "type": "HabitatSimDepthSensor",
                "uuid": "depth_left",
                "width": 448,
                "height": 448,
                "hfov": 79,
                "min_depth": 0.0,
                "max_depth": 3.0,
                "normalize_depth": True,
                "position": [-0.05, 1.25, 0.0],  # 左眼，向左偏移5cm
                "orientation": [0.0, 0.0, 0.0]
            })
            OmegaConf.set_struct(self.config, True)


    def config_env(self, scene: str = None) -> habitat.Env:
        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.dataset.update(
                    {
                        "data_path": self.data_path,
                    }
                )
        print(OmegaConf.to_yaml(self.config))
        return habitat.Env(config=self.config)


    def generate(self, rank: int = 0, world_size: int = 1, render_points: bool = False) -> None:
        # 1. 设置保存路径
        os.makedirs(os.path.join(self.output_path), exist_ok=True)

        # 1.1 计算并保存相机内参（所有相机共享：左RGB、右RGB、左深度）
        img_width, img_height = 448, 448
        hfov = 79  # degrees
        K = get_camera_intrinsics(img_width, img_height, hfov)

        # 保存内参矩阵到 intrinsics 文件（3x3矩阵，每行一行）
        intrinsics_path = os.path.join(self.output_path, "intrinsics.txt")
        with open(intrinsics_path, 'w') as f:
            for row in K:
                f.write(' '.join([f"{val:.6f}" for val in row]) + '\n')
        print(f"Camera intrinsics saved to {intrinsics_path}")

        # 1.2 计算并保存双目相机基线距离
        # 左相机位置：[-0.05, 1.25, 0.0]，右相机位置：[0.05, 1.25, 0.0]
        baseline = 0.05 - (-0.05)  # 基线距离 = 0.1米
        baseline_path = os.path.join(self.output_path, "baseline.txt")
        with open(baseline_path, 'w') as f:
            f.write(f"{baseline:.6f}\n")
        print(f"Stereo baseline saved to {baseline_path}")

        # 2. 创建环境 & 任务 （habitat-lab）
        env = self.config_env()
        # 3. 提取每个环境对应的任务
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)
        # 4.
        annotations = []
        for scene_id in sorted(scene_episode_dict.keys()):
            # 4.1 提取当前场景
            scan = scene_id.split("/")[-2]
            # 4.2 提取当前场景对应任务
            episodes = scene_episode_dict[scene_id]
            print(f"scene_id: {scene_id}, scan: {scan}")
            # 4.3 核心任务收集 & 执行
            for episode in episodes[rank::world_size]:
                # 4.3.1 准备当前要执行的任务
                env.current_episode = episode
                # 4.3.2
                agent = ShortestPathFollower(
                    sim=env.sim, goal_radius=0.5, return_one_hot=False)
                # 4.3.3 提取任务基本信息
                # 兼容 R2R 和 RXR 数据格式
                if self.dataset == 'rxr':
                    # RXR: instruction 是对象，有 language 属性
                    instruction_obj = episode.instruction
                    # 检查是否有 language 属性（RXR特有）
                    if hasattr(instruction_obj, 'language'):
                        language = instruction_obj.language
                        # 只处理英文任务
                        if not language.startswith('en'):
                            continue
                    instructions = instruction_obj.instruction_text
                else:
                    # R2R: instruction 是对象
                    instructions = episode.instruction.instruction_text

                trajectory_id = episode.trajectory_id
                scene_id = episode.scene_id.split('/')[-2]
                episode_id = int(episode.episode_id)
                ref_path = episode.reference_path
                # 4.3.4 获取起始位置的观测信息
                observation = env.reset()
                # 4.3.5 初始化相机偏移量（复用之前计算的内参K）
                camera_offset_left = np.array([-0.05, 1.25, 0.0])
                camera_offset_right = np.array([0.05, 1.25, 0.0])
                camera_offset_center = np.array([0.0, 1.25, 0.0])  # 两摄像头中间位置
                # 4.3.6 物理执行
                rgb_left_list = []
                rgb_right_list = []
                depth_left_list = []
                actions = [-1]
                next_waypoint_id = 1
                scene_dir = os.path.join(
                    self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}")
                rgb_left_dir = os.path.join(scene_dir, "rgb_left")
                rgb_right_dir = os.path.join(scene_dir, "rgb_right")
                depth_left_dir = os.path.join(scene_dir, "depth_left")
                label_points_file = os.path.join(scene_dir, "label_points.json")
                os.makedirs(rgb_left_dir, exist_ok=True)
                os.makedirs(rgb_right_dir, exist_ok=True)
                os.makedirs(depth_left_dir, exist_ok=True)
                # 初始化点标签数据结构
                label_points_data = []
                while not env.episode_over:
                    # 4.3.5.1 获取双目rgb观测和左视角深度观测
                    rgb_left = observation["rgb_left"]
                    rgb_right = observation["rgb_right"]
                    depth_left = observation["depth_left"]
                    rgb_left_list.append(rgb_left)
                    rgb_right_list.append(rgb_right)
                    depth_left_list.append(depth_left)

                    # 先计算下一步动作（用于目标点在后方时决定标注位置）
                    next_action = agent.get_next_action(
                        ref_path[next_waypoint_id])

                    # 计算 label_left_point 和 label_right_point
                    # 获取当前agent的位置和旋转
                    agent_state = env.sim.get_agent_state()
                    agent_position = agent_state.position  # [x, y, z]
                    agent_rotation = agent_state.rotation  # quaternion

                    # 获取最终目标位置（ref_path的最后一个点）
                    # 注意：ref_path中的点是agent在地面的位置，需要加上相机高度偏移
                    goal_agent_position = np.array(ref_path[-1])
                    goal_camera_center = goal_agent_position + np.array([0.0, 1.25, 0.0])  # 相机中心高度

                    # Debug: 只在第一帧打印调试信息
                    debug_mode = (len(rgb_left_list) == 1)
                    if debug_mode:
                        print(f"\n=== Frame {len(rgb_left_list)} Debug Info ===")
                        print(f"Current waypoint_id: {next_waypoint_id}")
                        print(f"Total waypoints: {len(ref_path)}")
                        print(f"Current agent position: {agent_position}")
                        print(f"Goal agent position: {goal_agent_position}")
                        print(f"Goal camera center: {goal_camera_center}")
                        print(f"Distance to goal: {np.linalg.norm(goal_agent_position - agent_position):.2f}m")
                        print(f"Next action: {next_action}")

                    # 投影最终目标相机中心到当前左视角
                    u_left, v_left, cam_x_left = project_3d_to_2d(
                        world_point=goal_camera_center,
                        agent_position=agent_position,
                        agent_rotation=agent_rotation,
                        camera_offset=camera_offset_left,
                        K=K,
                        width=img_width,
                        height=img_height,
                        debug=debug_mode
                    )
                    # 裁剪到图像边界（根据目标点的相对方位）
                    u_left, v_left = clip_to_image_bounds(u_left, v_left, img_width, img_height, cam_x_left)

                    # 投影最终目标相机中心到当前右视角
                    u_right, v_right, cam_x_right = project_3d_to_2d(
                        world_point=goal_camera_center,
                        agent_position=agent_position,
                        agent_rotation=agent_rotation,
                        camera_offset=camera_offset_right,
                        K=K,
                        width=img_width,
                        height=img_height,
                        debug=False
                    )
                    # 裁剪到图像边界（根据目标点的相对方位）
                    u_right, v_right = clip_to_image_bounds(u_right, v_right, img_width, img_height, cam_x_right)

                    # 保存当前帧的点标签数据
                    label_points_data.append({
                        "frame_id": len(rgb_left_list),
                        "label_left_point": [float(u_left), float(v_left)],
                        "label_right_point": [float(u_right), float(v_right)]
                    })

                    # 保存RGB图像（可选渲染目标点）
                    if render_points:
                        # 转换为PIL图像并绘制目标点
                        # 注意：v坐标使用bottom-left原点，需要转换回top-left用于PIL绘制
                        v_left_topleft = img_height - v_left
                        v_right_topleft = img_height - v_right

                        # 左图像
                        img_left = Image.fromarray(rgb_left).convert("RGB")
                        draw_left = ImageDraw.Draw(img_left)
                        radius = 5
                        draw_left.ellipse(
                            [(u_left - radius, v_left_topleft - radius),
                             (u_left + radius, v_left_topleft + radius)],
                            fill='red', outline='red'
                        )
                        img_left.save(os.path.join(rgb_left_dir, f"{len(rgb_left_list):03d}.jpg"))

                        # 右图像
                        img_right = Image.fromarray(rgb_right).convert("RGB")
                        draw_right = ImageDraw.Draw(img_right)
                        draw_right.ellipse(
                            [(u_right - radius, v_right_topleft - radius),
                             (u_right + radius, v_right_topleft + radius)],
                            fill='red', outline='red'
                        )
                        img_right.save(os.path.join(rgb_right_dir, f"{len(rgb_right_list):03d}.jpg"))
                    else:
                        # 不渲染目标点，直接保存原始图像
                        Image.fromarray(rgb_left).convert("RGB").save(
                            os.path.join(rgb_left_dir, f"{len(rgb_left_list):03d}.jpg"))
                        Image.fromarray(rgb_right).convert("RGB").save(
                            os.path.join(rgb_right_dir, f"{len(rgb_right_list):03d}.jpg"))
                    # 保存深度图像（单位：毫米，uint16格式）
                    # Habitat深度是归一化的(0-1)，需要反归一化到真实深度
                    depth_m = depth_left * 10.0  # 反归一化：深度范围 0-10米
                    depth_mm = (depth_m * 1000.0).astype(np.uint16)  # 转换为毫米
                    # 移除单维度：从 (H, W, 1) 变为 (H, W)
                    depth_mm = np.squeeze(depth_mm)
                    Image.fromarray(depth_mm).save(
                        os.path.join(depth_left_dir, f"{len(depth_left_list):03d}.png"))

                    # 走完第一个目标点后切换下一个目标点
                    force_episode_over = False
                    while next_action == 0:
                        next_waypoint_id += 1
                        if next_waypoint_id == len(ref_path) - 1:
                            agent = ShortestPathFollower(
                                sim=env.sim, goal_radius=0.25, return_one_hot=False)
                        if next_waypoint_id >= len(ref_path):
                            force_episode_over = True
                            break
                        next_action = agent.get_next_action(
                            ref_path[next_waypoint_id])
                    # 如果所有参考点都执行完 就跳出
                    if force_episode_over:
                        break
                    # 执行动作
                    observation = env.step(next_action)
                    actions.append(next_action)
                # 4.3.6 动作太多就跳过
                if len(actions) > 498:
                    continue
                # 4.3.7 保存点标签数据到 JSON 文件
                with open(label_points_file, 'w') as f:
                    json.dump(label_points_data, f, indent=4)
                # 4.3.8 收集 & 保存 内容
                assert len(actions) == len(rgb_left_list) == len(rgb_right_list) == len(depth_left_list) == len(label_points_data), \
                    f"Data length mismatch - actions: {len(actions)}, rgb_left: {len(rgb_left_list)}, rgb_right: {len(rgb_right_list)}, depth_left: {len(depth_left_list)}, label_points: {len(label_points_data)}"
                annotations.append({
                    "id": episode_id,
                    "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
                    "instructions": instructions if isinstance(instructions, list) else [instructions],
                    "actions": actions,
                })

                with open(os.path.join(self.output_path, "summary.json"), "a") as f:
                    result = {
                        "id": episode_id,
                        "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
                        "instructions": instructions if isinstance(instructions, list) else [instructions],
                        "actions": actions,
                        "trajectory_id": trajectory_id,
                        "scene_id": scene_id,
                    }
                    f.write(json.dumps(result) + "\n")

            with open(os.path.join(self.output_path, f"annotations_{rank}.json"), "w") as f:
                json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="R2R")
    parser.add_argument("--config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--output_path", type=str, default="cache/R2R")
    parser.add_argument("--data_path", type=str, default="task/r2r/train/train.json.gz")
    parser.add_argument("--render_points", action="store_true",
                        help="If set, render target points on RGB images (default: False)")
    args = parser.parse_args()

    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', 1)))

    runner = StreamVLNHabitatRunner(
        dataset=args.dataset,
        config_path=args.config_path,
        output_path=args.output_path,
        data_path=args.data_path
    )
    runner.generate(rank, world_size, render_points=args.render_points)

