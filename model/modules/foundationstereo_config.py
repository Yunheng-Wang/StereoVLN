import torch
from dataclasses import dataclass


@dataclass 
class FoundationStereoConfig:
    checkpoint_path = "/home/CONNECT/yfang870/yunhengwang/StereoVLN/model/base/FoundationStereo/checkpoints/23-51-11"
    dtype = torch.bfloat16
    dim = 2048

    # 相机的内参(临时)
    K = torch.tensor(
        [
            [754.6681, 0.0, 489.3795],
            [0.0, 754.6681, 265.16162],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # 双目相机之间的物理距离 (meters)
    camera_dis = 0.063