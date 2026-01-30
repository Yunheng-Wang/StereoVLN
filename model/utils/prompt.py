system_description = "Imagine you are an autonomous robot equipped with binocular eyes. Your task is to navigate based on given instructions. You must analyze your previous memories, your current left and right eye observations, and the depth information you feel to determine the safest and most efficient next move."
history_description = "I have been given a video of historical observations: "
current_left_description = "My current observation from the left camera is: "
current_right_description = "My current observation from the right camera is: "
depth_description = "My current sense of distance (depth) is: "

other = "Your assigned task is: {instruction}. Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree, moving forward a certain distance, or stop if the task is completed."

temple = "<|im_start|>system\n{system_description}<|im_end|>\n<|im_start|>user\n{history_description}{history_video}{current_left_description}{current_left_frame}\n{current_right_description}{current_right_frame}\n{depth_description}{depth_frame}\n{other}<|im_end|>\n<|im_start|>assistant\n"




