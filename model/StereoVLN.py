import torch 
import timm
from torch import nn
import torch.nn.functional as F
from .modules.foundationstereo import FoundationStereoModel
from .modules.foundationstereo_config import FoundationStereoConfig
from .modules.vlm import InternVLModel
from .modules.vlm_config import InternVLConfig
from .modules.dino import DINOv2
from .modules.dino_config import DINOv2Config
from .modules.pointhead import PointHead
from .modules.pointhead_config import PointHeadConfig
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
        # 4. 左眼 Point Head
        LeftPointHeadConfig = PointHeadConfig()
        self.LeftPointHead = PointHead(LeftPointHeadConfig)
        # 5. 右眼 Point Head
        RightPointHeadConfig = PointHeadConfig()
        self.RightPointHead = PointHead(RightPointHeadConfig)


    def forward(
        self,
        instruction: str,
        left_current_frame: torch.Tensor,   # [B, 1, 3, 448, 448] - 0~255
        right_current_frame: torch.Tensor,  # [B, 1, 3, 448, 448] - 0~255
        left_history_video: torch.Tensor,   # [B, 8, 3, 448, 448] - 0~255
        right_history_video: torch.Tensor,  # [B, 8, 3, 448, 448] - 0~255
        label_left_point: torch.Tensor,     # [B, 2]
        label_right_point: torch.Tensor,    # [B, 2]
        label_depth: torch.Tensor,          # [B, 1, 448, 448]
        label_answer: List[str]             # [B] - action sequences like "<action>Move forward</action><action>turn left</action>"
    ) -> dict:
        # 0. 深度编码（左右当前帧） (196 tokens) -> 获取左视角深度
        depth_feature, depth_token, left_vit_feat, features_left, features_right = self.FoundationStereo.FoundationStereoEncoder(left_current_frame.squeeze(1), right_current_frame.squeeze(1))
        # 1. 左当前帧 点目标编码 (256 tokens)
        left_current_frame_token = self.DINO.DINOv2Encoder(left_current_frame.squeeze(1))
        # 2. 右当前帧&历史帧 语义编码
        right_history_video_token, right_current_frame_token = self.VLM.Encoder_Vsion(torch.cat([right_history_video, right_current_frame], dim = 1))
        # 3. 组织 prompt
        batch_prompts = []
        prompt_lengths = []  # 记录每个样本的prompt长度（不含answer）
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
            # 3.6. 记录prompt长度（用于后续识别answer tokens位置）
            prompt_tokens = self.VLM.tokenizer(prompt, return_tensors='pt')['input_ids']
            prompt_lengths.append(prompt_tokens.shape[1])
            # 3.7. 添加label_answer用于teacher forcing
            prompt_with_answer = prompt + label_answer[b] + "<|im_end|>"
            batch_prompts.append(prompt_with_answer)
        # 4. Tokenizer 处理文本
        model_inputs = self.VLM.tokenizer(batch_prompts, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(left_current_frame.device)
        attention_mask = model_inputs['attention_mask'].to(left_current_frame.device)
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

        # 6. 组织 attention mask
        seq_len = input_ids.shape[1]
        custom_attention_mask = torch.ones((B, seq_len, seq_len), dtype=torch.float32, device=input_ids.device)
        # 6.1 mask掉 answer tokens 对 left 和 depth tokens 的 attention
        for b in range(B):
            # 6.1.1 找出所有visual token的位置
            visual_token_positions = torch.where(mask[b])[0]
            if len(visual_token_positions) == 0:
                continue
            # 6.1.2 找到 left 和 depth token的位置
            l_hist = N * right_history_video_token.shape[2]
            l_left = left_current_frame_token.shape[1]
            l_right = right_current_frame_token.shape[2]
            l_dep = depth_token.shape[1]
            left_start_idx = l_hist
            left_end_idx = l_hist + l_left
            depth_start_idx = l_hist + l_left + l_right
            depth_end_idx = l_hist + l_left + l_right + l_dep
            left_positions = visual_token_positions[left_start_idx:left_end_idx]
            depth_positions = visual_token_positions[depth_start_idx:depth_end_idx]
            # 6.1.3 mask掉answer tokens对left和depth tokens的attention
            answer_start_pos = prompt_lengths[b]
            if len(left_positions) > 0:
                custom_attention_mask[b, answer_start_pos:, left_positions] = 0
            if len(depth_positions) > 0:
                custom_attention_mask[b, answer_start_pos:, depth_positions] = 0
        # 6.2 应用 causal mask - 视觉token 相互可见
        causal_mask = torch.ones((B, seq_len, seq_len), device=input_ids.device)
        for b in range(B):
            answer_start_pos = prompt_lengths[b]
            if answer_start_pos < seq_len:
                answer_len = seq_len - answer_start_pos
                causal_mask[b, answer_start_pos:, answer_start_pos:] = torch.tril(
                    torch.ones((answer_len, answer_len), device=input_ids.device)
                )
        custom_attention_mask = custom_attention_mask * causal_mask
        # 6.3 应用 padding mask
        padding_mask = attention_mask.unsqueeze(1).expand(B, seq_len, seq_len)
        custom_attention_mask = custom_attention_mask * padding_mask 
        # 6.4 转换为适合模型的格式 (0 -> -inf, 1 -> 0)，并转为4D mask
        custom_attention_mask = (1.0 - custom_attention_mask) * torch.finfo(input_embeds.dtype).min
        # 添加head维度：[B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
        custom_attention_mask = custom_attention_mask.unsqueeze(1)
        # 7. 位置编码
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        position_ids = position_ids.to(input_embeds.device)
        # 8. 输入到 VLM Backbone 中（使用自定义attention mask）
        output = self.VLM.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=custom_attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        # 9. 整理输出token
        output_tokens = output.hidden_states[-1]
        l_hist  = N * right_history_video_token.shape[2]
        l_left  = left_current_frame_token.shape[1]      
        l_right = right_current_frame_token.shape[2]     
        l_dep   = depth_token.shape[1]     
        all_visual_output_tokens = output_tokens[mask].view(B, l_hist + l_left + l_right + l_dep, -1)     
        depth_output_tokens = all_visual_output_tokens[:, -l_dep:, :]         
        left_current_output_tokens = all_visual_output_tokens[:, l_hist : l_hist + l_left, :]
        right_current_output_tokens = all_visual_output_tokens[:, l_hist + l_left : l_hist + l_left + l_right, :]
        # 10. 输入到 Point Head 层
        left_point = self.LeftPointHead(left_current_output_tokens)
        right_point = self.RightPointHead(right_current_output_tokens)
        # 11. point loss 
        left_point_loss = F.smooth_l1_loss(left_point, label_left_point, reduction="mean")
        right_point_loss = F.smooth_l1_loss(right_point, label_right_point, reduction="mean")
        point_loss = left_point_loss + right_point_loss
        # 12. depth 估计 (左视角)
        depth = self.FoundationStereo.FoundationStereoDecoder(depth_feature, depth_output_tokens, left_current_frame.squeeze(1), left_vit_feat, features_left, features_right, iters = 10)
        # 13. depth loss
        label_depth_sq = label_depth.squeeze(1) 
        valid = torch.isfinite(depth) & torch.isfinite(label_depth_sq) & (label_depth_sq > 0)
        if valid.any():
            depth_loss = F.smooth_l1_loss(depth[valid], label_depth_sq[valid], reduction="mean")
        else:
            depth_loss = depth.sum() * 0.0
        # 13. 语言性动作估计 & loss
        # 13.1. 过 llm head (mask 掉 left 和 depth 的视觉 tokens)
        lm_relevant_mask = torch.ones(B, seq_len, dtype=torch.bool, device=input_ids.device)
        for b in range(B):
            visual_token_positions = torch.where(mask[b])[0]
            if len(visual_token_positions) > 0:
                left_start_idx = l_hist
                left_end_idx = l_hist + l_left
                depth_start_idx = l_hist + l_left + l_right
                depth_end_idx = l_hist + l_left + l_right + l_dep
                left_positions = visual_token_positions[left_start_idx:left_end_idx]
                depth_positions = visual_token_positions[depth_start_idx:depth_end_idx]
                if len(left_positions) > 0:
                    lm_relevant_mask[b, left_positions] = False
                if len(depth_positions) > 0:
                    lm_relevant_mask[b, depth_positions] = False
        lm_input_tokens = output_tokens.clone()
        lm_input_tokens[~lm_relevant_mask] = 0
        lm_logits = self.VLM.model.language_model.lm_head(lm_input_tokens)
        # 13.2. 计算语言动作的 loss
        language_loss = torch.tensor(0.0, device=input_ids.device)
        valid_token_count = 0
        for b in range(B):
            # 13.2.1 获取模型的回答
            answer_start = prompt_lengths[b]
            seq_length = input_ids.shape[1]
            if answer_start >= seq_length - 1:
                continue
            answer_logits = lm_logits[b, answer_start:-1, :]
            # 13.2.2 获取标签
            answer_labels = input_ids[b, answer_start+1:]
            answer_mask = attention_mask[b, answer_start+1:]
            # 13.2.3 计算 loss
            if answer_mask.sum() > 0:
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                per_token_loss = loss_fct(answer_logits, answer_labels)
                masked_loss = per_token_loss * answer_mask.float()
                language_loss += masked_loss.sum()
                valid_token_count += answer_mask.sum().item()
        # 13.3. 平均loss
        if valid_token_count > 0:
            language_loss = language_loss / valid_token_count
        else:
            language_loss = language_loss * 0.0

        # 14. 返回所有losses
        return {
            'point_loss': point_loss,
            'depth_loss': depth_loss,
            'language_loss': language_loss,
            'total_loss': point_loss + depth_loss + language_loss
        }




