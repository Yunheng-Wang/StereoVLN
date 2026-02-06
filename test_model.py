"""Test script for StereoVLN model

Run this in the `stereovln` environment (Python 3.12):
    python test_model.py --checkpoint_path /path/to/checkpoint
"""
import os
import json
import random
import torch
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from model.StereoVLNConfig import StereoVLNConfig
from model.StereoVLN import StereoVLN


def load_model(args):
    """Load StereoVLN model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load(os.path.join(args.checkpoint_path, "config.json"))
    camera_k = torch.tensor([
        [271.733729, 0.000000, 224.000000],
        [0.000000, 271.733729, 224.000000],
        [0.000000, 0.000000, 1.000000]
    ], dtype=torch.float32)
    camera_baseline = 0.1
    project_root = os.path.dirname(os.path.abspath(__file__))

    model_config = StereoVLNConfig(
        image_size=(config.main.image_size, config.main.image_size),
        dtype=torch.bfloat16 if config.main.dtype == "bf16" else torch.float16,
        dim=config.model.dim,
        max_tokens=config.model.vlm.max_tokens,
        vlm_checkpoints_path=os.path.join(project_root, "model/base/InternVL3_5-2B"),
        camera_k=camera_k,
        camera_baseline=camera_baseline,
        foundationstereo_checkpoints_path=os.path.join(project_root, "model/base/FoundationStereo/checkpoints/23-51-11"),
        mlp_ratio=config.model.pointhead.mlp_ratio,
        dropout=config.model.pointhead.dropout
    )
    model = StereoVLN(model_config)

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint_dir = args.checkpoint_path
        bin_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.bin')]
        if len(bin_files) > 0:
            model_files = [f for f in bin_files if f.startswith('pytorch_model')]
            if model_files:
                checkpoint_file = os.path.join(checkpoint_dir, model_files[0])
            else:
                checkpoint_file = os.path.join(checkpoint_dir, bin_files[0])
            print(f"Loading checkpoint from {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"[WARNING] Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"[WARNING] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
            if not missing_keys and not unexpected_keys:
                print("[INFO] All keys matched successfully!")
        else:
            print(f"Warning: No .bin file found in {checkpoint_dir}")

    model.requires_grad_(False)
    model.eval()
    model.to(device)
    return model, device


def load_image(image_path, image_size=448):
    """Load and preprocess image to tensor [3, H, W] in range 0-255."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)  # [H, W, 3]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [3, H, W]
    return img_tensor


def main(args):
    data_root = "/home/CONNECT/yfang870/yunhengwang/StereoVLN/data/cache/train/R2R"
    images_root = os.path.join(data_root, "images")

    # Load random annotation file
    annotation_files = [f for f in os.listdir(data_root) if f.startswith('annotations_') and f.endswith('.json')]
    annotation_file = random.choice(annotation_files)
    print(f"Loading annotations from: {annotation_file}")

    with open(os.path.join(data_root, annotation_file), 'r') as f:
        annotations = json.load(f)

    # Pick a random episode
    episode = random.choice(annotations)
    video_path = os.path.join(data_root, episode['video'])
    instruction = episode['instructions'][0]
    actions = episode['actions']

    print(f"\n=== Selected Episode ===")
    print(f"Video: {episode['video']}")
    print(f"Instruction: {instruction}")
    print(f"Total frames: {len(actions)}")

    # List available frames
    rgb_left_dir = os.path.join(video_path, "rgb_left")
    rgb_right_dir = os.path.join(video_path, "rgb_right")
    frames = sorted([f for f in os.listdir(rgb_left_dir) if f.endswith('.jpg')])
    num_frames = len(frames)
    print(f"Available frames: {num_frames}")

    # Pick a random current frame (need at least some history)
    history_length = 8
    if num_frames <= history_length:
        current_frame_idx = num_frames - 1
    else:
        current_frame_idx = random.randint(history_length, num_frames - 1)

    print(f"\nSelected current frame index: {current_frame_idx}")

    # Load model
    print(f"\n=== Loading Model ===")
    model, device = load_model(args)
    print(f"Model loaded on {device}")

    # Load current frame (left and right)
    current_frame_name = frames[current_frame_idx]
    left_current = load_image(os.path.join(rgb_left_dir, current_frame_name))
    right_current = load_image(os.path.join(rgb_right_dir, current_frame_name))

    # Load history frames (right camera only)
    history_frames = []
    for i in range(history_length):
        hist_idx = max(0, current_frame_idx - history_length + i)
        hist_frame_name = frames[hist_idx]
        hist_img = load_image(os.path.join(rgb_right_dir, hist_frame_name))
        history_frames.append(hist_img)

    # Stack to batch tensors
    left_current_frame = left_current.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1, 1, 3, 448, 448]
    right_current_frame = right_current.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1, 1, 3, 448, 448]
    right_history_video = torch.stack(history_frames, dim=0).unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1, 8, 3, 448, 448]

    print(f"\n=== Input Shapes ===")
    print(f"left_current_frame: {left_current_frame.shape}")
    print(f"right_current_frame: {right_current_frame.shape}")
    print(f"right_history_video: {right_history_video.shape}")

    # Build history action string
    history_actions = actions[1:current_frame_idx + 1]  # actions[0] is -1 (start)
    action_map = {1: "forward", 2: "turn_left", 3: "turn_right", 4: "stop"}
    history_action_str = ", ".join([action_map.get(a, str(a)) for a in history_actions[-history_length:]])
    print(f"\nHistory actions: {history_action_str}")

    # Run inference
    print(f"\n=== Running Inference ===")
    with torch.no_grad():
        outputs = model.inference(
            instruction=[instruction],
            history_action=[history_action_str],
            left_current_frame=left_current_frame,
            right_current_frame=right_current_frame,
            right_history_video=right_history_video,
            max_new_tokens=4096,
        )

    print(f"\n=== Model Output ===")
    print(f"Generated text: {outputs['generated_text']}")
    if 'left_point' in outputs and outputs['left_point'] is not None:
        print(f"Left point: {outputs['left_point'].cpu().tolist()}")
    if 'right_point' in outputs and outputs['right_point'] is not None:
        print(f"Right point: {outputs['right_point'].cpu().tolist()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default="/home/CONNECT/yfang870/yunhengwang/StereoVLN/results/2026-02-05_22/checkpoint_1")
    args = parser.parse_args()

    main(args)

    # Example usage:
    # python test_model.py --checkpoint_path /home/CONNECT/yfang870/yunhengwang/StereoVLN/results/2026-02-05_22/checkpoint_1
