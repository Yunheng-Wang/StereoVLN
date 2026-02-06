"""StereoVLN Model Server

Run this in the `stereovln` environment (Python 3.12):
    python model_server.py --checkpoint_path /path/to/checkpoint --port 5000
"""
import os
import io
import json
import torch
import argparse
import numpy as np
from flask import Flask, request, jsonify
from omegaconf import OmegaConf

from model.StereoVLNConfig import StereoVLNConfig
from model.StereoVLN import StereoVLN

app = Flask(__name__)
model = None
device = None


def load_model(args):
    """Load StereoVLN model from checkpoint."""
    # 1. 配置基本项
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load(os.path.join(args.checkpoint_path, "config.json"))
    camera_k = torch.tensor([
        [271.733729, 0.000000, 224.000000],
        [0.000000, 271.733729, 224.000000],
        [0.000000, 0.000000, 1.000000]
    ], dtype=torch.float32)
    camera_baseline = 0.1
    project_root = os.path.dirname(os.path.abspath(__file__))
    # 2. 创建模型
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
    # 3. 加载权重
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint_dir = args.checkpoint_path
        bin_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.bin')]
        if len(bin_files) > 0:
            # 优先找 pytorch_model 开头的文件（accelerator 保存的格式）
            model_files = [f for f in bin_files if f.startswith('pytorch_model')]
            if model_files:
                checkpoint_file = os.path.join(checkpoint_dir, model_files[0])
            else:
                checkpoint_file = os.path.join(checkpoint_dir, bin_files[0])
            print(f"Loading checkpoint from {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')

            # DEBUG: 打印 checkpoint 的 keys
            if isinstance(checkpoint, dict):
                print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())[:10]}...")
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # DEBUG: 打印 state_dict 的一些 keys
            print(f"[DEBUG] State dict keys (first 10): {list(state_dict.keys())[:10]}")
            print(f"[DEBUG] Model keys (first 10): {list(model.state_dict().keys())[:10]}")

            # 加载权重，打印不匹配的 keys
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
    return model


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'device': str(device)})


@app.route('/inference', methods=['POST'])
def inference():
    """Model inference endpoint.

    Expected JSON payload:
    {
        "instruction": ["instruction text"],
        "history_action": ["history action string"],
        "left_current_frame": base64 or list,  # [1, 1, 3, 448, 448]
        "right_current_frame": base64 or list, # [1, 1, 3, 448, 448]
        "right_history_video": base64 or list, # [1, history_num, 3, 448, 448]
        "max_new_tokens": 4096
    }
    """
    try:
        data = request.get_json()

        instruction = data['instruction']
        history_action = data['history_action']
        max_new_tokens = data.get('max_new_tokens', 4096)

        # Decode tensors from lists
        left_current_frame = torch.tensor(data['left_current_frame'], dtype=torch.bfloat16).to(device)
        right_current_frame = torch.tensor(data['right_current_frame'], dtype=torch.bfloat16).to(device)
        right_history_video = torch.tensor(data['right_history_video'], dtype=torch.bfloat16).to(device)

        with torch.no_grad():
            outputs = model.inference(
                instruction=instruction,
                history_action=history_action,
                left_current_frame=left_current_frame,
                right_current_frame=right_current_frame,
                right_history_video=right_history_video,
                max_new_tokens=max_new_tokens,
            )

        response = {
            'generated_text': outputs['generated_text'],
            'left_point': outputs['left_point'].cpu().tolist() if 'left_point' in outputs else None,
            'right_point': outputs['right_point'].cpu().tolist() if 'right_point' in outputs else None,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint_path}...")
    model = load_model(args)
    print(f"Model loaded on {device}")

    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)

    # python model_server.py --checkpoint_path /home/CONNECT/yfang870/yunhengwang/StereoVLN/results/2026-02-05_22/checkpoint_1 --port 5000

