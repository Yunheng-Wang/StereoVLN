import torch
from model.StereoVLN import StereoVLN


if __name__ == "__main__":

    vlm_path = "model/base/InternVL3_5-2B"


    # 临时输入
    batch_size = 8
    image_size = 448

    instruction = ["Describe this video in detail."] * batch_size
    history_action = ["<action>Move forward</action><action>Move forward</action><action>Move forward</action><action>Move forward</action>"] * batch_size



    left_cur_video = torch.rand(batch_size, 1, 3, image_size, image_size, device="cuda", dtype=torch.bfloat16)* 255
    right_cur_video = torch.rand(batch_size, 1, 3, image_size, image_size, device="cuda", dtype=torch.bfloat16)* 255

    right_his_video = torch.rand(batch_size, 8, 3, image_size, image_size, device="cuda", dtype=torch.bfloat16)* 255

    label_left_point = torch.rand(batch_size, 2, device="cuda", dtype=torch.bfloat16)* image_size
    label_right_point = torch.rand(batch_size, 2, device="cuda", dtype=torch.bfloat16)* image_size

    label_depth = torch.rand(batch_size, 1, image_size, image_size, device="cuda", dtype=torch.bfloat16)* 100
    label_answer = ["<action>Move forward</action><action>Move forward</action><action>Move forward</action><action>Move forward</action>"] * batch_size

    model = StereoVLN()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("total:", total_params, "->", total_params / 1e9, "B")
    print("trainable:", trainable_params, "->", trainable_params / 1e9, "B")

    model(instruction, history_action, left_cur_video, right_cur_video, right_his_video, label_left_point, label_right_point, label_depth, label_answer)



