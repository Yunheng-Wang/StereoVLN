import torch
from model.StereoVLN import StereoVLN


if __name__ == "__main__":

    vlm_path = "model/base/InternVL3_5-2B"


    # 临时输入
    batch_size = 2
    image_size = 448

    instruction = ["Describe this video in detail."] * batch_size

    left_cur_video = torch.rand(batch_size, 1, 3, image_size, image_size, device="cuda", dtype=torch.bfloat16)* 255
    right_cur_video = torch.rand(batch_size, 1, 3, image_size, image_size, device="cuda", dtype=torch.bfloat16)* 255

    left_his_video = torch.rand(batch_size, 8, 3, image_size, image_size, device="cuda", dtype=torch.bfloat16)* 255
    right_his_video = torch.rand(batch_size, 8, 3, image_size, image_size, device="cuda", dtype=torch.bfloat16)* 255
    
    model = StereoVLN()
    model(instruction, left_cur_video, right_cur_video, left_his_video, right_his_video)



