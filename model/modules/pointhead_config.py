import torch
from dataclasses import dataclass


@dataclass 
class PointHeadConfig:
    dtype = torch.bfloat16
    img_size = (448, 448)
    mlp_ratio = 0.5
    dropout = 0.1

    dim = 2048