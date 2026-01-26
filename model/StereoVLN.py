import torch 
import timm
from torch import nn
from .modules.foundationstereo import FoundationStereoModel
from .modules.foundationstereo_config import FoundationStereoConfig
from .modules.vlm import InternVLModel
from .modules.vlm_config import InternVLConfig
from .modules.dino import DINOv2
from .modules.dino_config import DINOv2Config
from .utils.prompt import temple, system_description, history_description, current_left_description, current_right_description, depth_description, other
from typing import List, Optional


class StereoVLN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 创建 VLM BackBone
        VLMConfig = InternVLConfig()
        self.VLM = InternVLModel(VLMConfig)
        # 2. 创建 FoundationStereo
        DepthConfig = FoundationStereoConfig()
        self.FoundationStereo = FoundationStereoModel(DepthConfig)
        # 3. 创建 DINOv2
        DINOConfig = DINOv2Config()
        self.DINO = DINOv2(DINOConfig)




    def forward(
        self, 
        instruction: str, 
        left_current_frame: torch.Tensor,   # [B, 1, 3, 448, 448] - 0~255
        right_current_frame: torch.Tensor,  # [B, 1, 3, 448, 448] - 0~255
        left_history_video: torch.Tensor,   # [B, 8, 3, 448, 448] - 0~255
        right_history_video: torch.Tensor,  # [B, 8, 3, 448, 448] - 0~255
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 0. 深度编码（左右当前帧） (196 tokens) -> 获取左视角深度
        depth_feature, depth_token, left_vit_feat, features_left, features_right = self.FoundationStereo.FoundationStereoEncoder(left_current_frame.squeeze(1), right_current_frame.squeeze(1))
        # 1. 左当前帧 点目标编码 (256 tokens)
        left_current_frame_token = self.DINO.DINOv2Encoder(left_current_frame.squeeze(1))
        # 2. 右当前帧&历史帧 语义编码
        right_history_video_token, right_current_frame_token = self.VLM.Encoder_Vsion(torch.cat([right_history_video, right_current_frame], dim = 1))
        # 3. 组织 prompt 
        batch_prompts = []
        B, N, _, _ = right_history_video_token.shape
        for b in range(B):
            # 3.1. 历史帧
            history_video_str = ""
            for i in range(N):
                t_count = right_history_video_token.shape[2]
                history_video_str += f"Frame {i+1}: <img>" + "<IMG_CONTEXT>" * t_count + "</img>\n"
            # 3.2. 左视角
            t_dino = left_current_frame_token.shape[1]
            left_frame_str = "<img>" + "<IMG_CONTEXT>" * t_dino + "</img>"
            # 3.3. 右视角
            t_vlm = right_current_frame_token.shape[2]
            right_frame_str = "<img>" + "<IMG_CONTEXT>" * t_vlm + "</img>"
            # 3.4. 深度
            t_dep = depth_token.shape[1]
            depth_frame_str = "<dep>" + "<IMG_CONTEXT>" * t_dep + "</dep>"
            # 3.5 填入模板
            prompt = temple.format(
                    system_description = system_description, 
                    history_description = history_description,
                    history_video = history_video_str,
                    current_left_description = current_left_description,
                    current_left_frame =  left_frame_str,
                    current_right_description = current_right_description, 
                    current_right_frame = right_frame_str,
                    depth_description = depth_description,
                    depth_frame = depth_frame_str,
                    other = other
                )
            batch_prompts.append(prompt)
        # 4. Tokenizer 处理文本
        model_inputs = self.VLM.tokenizer(batch_prompts, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(left_current_frame.device)
        attention_mask = model_inputs['attention_mask']
        input_embeds = self.VLM.model.language_model.get_input_embeddings()(input_ids)
        # 5. 输入视觉token
        visual_features = []
        for b in range(B):
            # 历史帧特征
            visual_features.append(right_history_video_token[b].view(-1, right_history_video_token.shape[-1]))
            # 左眼 DINO
            visual_features.append(left_current_frame_token[b])
            # 右眼 VLM
            visual_features.append(right_current_frame_token[b].view(-1, right_current_frame_token.shape[-1]))
            # 深度
            visual_features.append(depth_token[b])
        flatten_visual_feats = torch.cat(visual_features, dim=0)
        img_context_token_id = self.VLM.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        mask = (input_ids == img_context_token_id)
        input_embeds[mask] = flatten_visual_feats.to(input_embeds.dtype)
        # 6. 位置编码
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        position_ids = position_ids.to(input_embeds.device)
        # 7. 输入到 VLM Backbone 中
        output = self.VLM.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        # 8. 整理输出token
        output_tokens = output.hidden_states[-1]
        l_hist  = N * right_history_video_token.shape[2]
        l_left  = left_current_frame_token.shape[1]      
        l_right = right_current_frame_token.shape[2]     
        l_dep   = depth_token.shape[1]     
        all_visual_output_tokens = output_tokens[mask].view(B, l_hist + l_left + l_right + l_dep, -1)     
        depth_output_tokens = all_visual_output_tokens[:, -l_dep:, :]         
        left_current_output_tokens = all_visual_output_tokens[:, l_hist : l_hist + l_left, :]
        right_current_output_tokens = all_visual_output_tokens[:, l_hist + l_left : l_hist + l_left + l_right, :]

        # 9. 输入到 LLM Head 层
        


        

        # 10. 深度图估计
        depth = self.FoundationStereo.FoundationStereoDecoder(depth_feature, depth_output_tokens, left_current_frame.squeeze(1), left_vit_feat, features_left, features_right, iters = 10)
        
        pass






        pass




