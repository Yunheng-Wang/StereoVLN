import torch
from dataclasses import dataclass


@dataclass 
class DINOv2Config:
    image_size = (448,448)
    dtype = torch.bfloat16