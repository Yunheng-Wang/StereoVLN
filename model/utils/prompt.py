system_description = "You are an autonomous robot equipped with binocular eyes. Your task is to navigate based on given instructions. You must analyze your previous memories, your current left and right eye observations, and the depth information you feel to determine the safest and most efficient next move."
history_description = "The images I saw in the past are: "
current_left_description = "My current observation from the left camera is: "
current_right_description = "My current observation from the right camera is: "
depth_description = "My current sense of distance (depth) is: "

other = "现在你需要完成的任务是：{instruction}"

temple = "<|im_start|>system\n{system_description}<|im_end|>\n<|im_start|>user\n{history_description}{history_video}{current_left_description}{current_left_frame}\n{current_right_description}{current_right_frame}\n{depth_description}{depth_frame}<|im_end|>\n<|im_start|>assistant\n"



