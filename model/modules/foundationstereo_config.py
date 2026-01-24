import torch
from dataclasses import dataclass


@dataclass 
class FoundationStereoConfig:
    checkpoint_path = "/home/CONNECT/yfang870/yunhengwang/StereoVLN/model/base/FoundationStereo/checkpoints/23-51-11"
    dtype = torch.bfloat16
    
    dim = 2048