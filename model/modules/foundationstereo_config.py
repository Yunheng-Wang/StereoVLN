import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FoundationStereoConfig:
    checkpoint_path: str = "/home/CONNECT/yfang870/yunhengwang/StereoVLN/model/base/FoundationStereo/checkpoints/23-51-11"
    dtype: torch.dtype = torch.bfloat16
    dim: int = 2048

    camera_baseline: float = 0.1
    intrinsic: torch.Tensor = torch.zeros((3, 3), dtype=torch.float32)