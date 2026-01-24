import torch
from dataclasses import dataclass


@dataclass 
class InternVLConfig:
    checkpoint_path = "/home/CONNECT/yfang870/yunhengwang/StereoVLN/model/base/InternVL3_5-2B"
    dtype = torch.bfloat16
    max_tokens = 4096
    image_size = (448, 448)