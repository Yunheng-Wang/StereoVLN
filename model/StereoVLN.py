import torch
import timm
import logging
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
from .utils.prompt import temple, system_description, history_description, current_left_description, current_right_description, depth_description, other, history_action_description
from typing import List, Optional
from .StereoVLNConfig import StereoVLNConfig

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> float:
    """Count model parameters and return size in billions."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e9  # Convert to billions


class StereoVLN(nn.Module):
    def __init__(self, config: StereoVLNConfig):
        super().__init__()
        # 1. 创建 VLM BackBone（启用 Gradient Checkpointing）
        VLMConfig = InternVLConfig(
            checkpoint_path = config.vlm_checkpoints_path,
            image_size = config.image_size,
            dtype = config.dtype,
            max_tokens = config.max_tokens,
        )
        self.VLM = InternVLModel(VLMConfig)
        if hasattr(self.VLM.model, 'language_model') and hasattr(self.VLM.model.language_model, 'gradient_checkpointing_enable'):
            self.VLM.model.language_model.gradient_checkpointing_enable()
        logger.info(f"VLM's LLM Backbone loaded: {count_parameters(self.VLM.model.language_model):.3f}B parameters")
        logger.info(f"VLM's Vision Encoder loaded: {count_parameters(self.VLM.model.vision_model):.3f}B parameters")
        # 2. 创建 FoundationStereo
        DepthConfig = FoundationStereoConfig(
            checkpoint_path = config.foundationstereo_checkpoints_path,
            dtype = config.dtype,
            dim = config.dim,
            camera_baseline = config.camera_baseline,
            intrinsic = config.camera_k,
        )
        self.FoundationStereo = FoundationStereoModel(DepthConfig)
        logger.info(f"FoundationStereo loaded: {count_parameters(self.FoundationStereo):.3f}B parameters")
        # 3. 创建 DINOv2
        DINOConfig = DINOv2Config(
            image_size = config.image_size,
            dtype = config.dtype,
        )
        self.DINO = DINOv2(DINOConfig)
        logger.info(f"DINOv2 loaded: {count_parameters(self.DINO):.3f}B parameters")
        # 4. 左眼 Point Head
        LeftPointHeadConfig = PointHeadConfig(
            img_size = config.image_size,
            dtype = config.dtype,
            dim = config.dim,
            mlp_ratio = config.mlp_ratio,
            dropout = config.dropout,
        )
        self.LeftPointHead = PointHead(LeftPointHeadConfig)
        logger.info(f"LeftPointHead loaded: {count_parameters(self.LeftPointHead):.3f}B parameters")
        # 5. 右眼 Point Head
        RightPointHeadConfig = PointHeadConfig(
            img_size = config.image_size,
            dtype = config.dtype,
            dim = config.dim,
            mlp_ratio = config.mlp_ratio,
            dropout = config.dropout,
        )
        self.RightPointHead = PointHead(RightPointHeadConfig)
        logger.info(f"RightPointHead loaded: {count_parameters(self.RightPointHead):.3f}B parameters")
        # 6. 打印总参数量
        logger.info(f"StereoVLN Total: {count_parameters(self):.3f}B parameters")


    def forward(
        self,
        instruction: List[str],
        history_action: List[str],          
        left_current_frame: torch.Tensor,   # [B, 1, 3, 448, 448] - 0~255
        right_current_frame: torch.Tensor,  # [B, 1, 3, 448, 448] - 0~255
        right_history_video: torch.Tensor,  # [B, 8, 3, 448, 448] - 0~255
        label_left_point: torch.Tensor,     # [B, 2]
        label_right_point: torch.Tensor,    # [B, 2]
        label_depth: torch.Tensor,          # [B, 1, 448, 448]
        label_answer: List[str]             # [B] - action sequences like "<action>Move forward</action><action>turn left</action>"
    ) -> dict:
        B = right_history_video.shape[0]
        N = right_history_video.shape[1]  # 历史帧数量
        # 0. 检测每个样本中哪些历史帧是全零的
        zero_frame_mask = (right_history_video.view(B, N, -1).sum(dim=-1) == 0)
        # 1. 深度编码（左右当前帧） (196 tokens) -> 获取左视角深度
        depth_feature, depth_token, left_vit_feat, features_left, features_right = self.FoundationStereo.FoundationStereoEncoder(left_current_frame.squeeze(1), right_current_frame.squeeze(1))
        # 2. 左当前帧 点目标编码 (256 tokens)
        left_current_frame_token = self.DINO.DINOv2Encoder(left_current_frame.squeeze(1))
        # 3. 右当前帧&历史帧 语义编码
        right_history_video_token, right_current_frame_token = self.VLM.Encoder_Vsion(torch.cat([right_history_video, right_current_frame], dim = 1))
        # 4. 组织 prompt
        batch_prompts = []
        prompt_lengths = []  # 记录每个样本的prompt长度（不含answer）
        valid_history_counts = []  # 记录每个样本中有效（非零）历史帧的数量
        for b in range(B):
            # 4.1. 历史帧 - 只处理非全零的帧
            history_video_str = ""
            valid_frame_count = 0
            for i in range(N):
                # 判断是否是全0 （如果全是全0 就是初始步，跳过）
                if zero_frame_mask[b, i]:
                    continue
                valid_frame_count += 1
                t_count = right_history_video_token.shape[2]
                history_video_str += f"Frame {valid_frame_count}: <img>" + "<IMG_CONTEXT>" * t_count + "</img>\n"
            valid_history_counts.append(valid_frame_count)
            # 4.2. 左视角
            t_dino = left_current_frame_token.shape[1]
            left_frame_str = "<img>" + "<IMG_CONTEXT>" * t_dino + "</img>"
            # 4.3. 右视角
            t_vlm = right_current_frame_token.shape[2]
            right_frame_str = "<img>" + "<IMG_CONTEXT>" * t_vlm + "</img>"
            # 4.4. 深度
            t_dep = depth_token.shape[1]
            depth_frame_str = "<dep>" + "<IMG_CONTEXT>" * t_dep + "</dep>"
            # 4.5 组织 text instruction
            instruction_text = other.format(
                instruction = instruction[b],
                history_action_description = history_action_description,
                history_action = history_action[b]
            )
            # 4.6 填入模板
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
                    other = instruction_text
                )
            # 4.7. 添加label_answer用于teacher forcing
            # prompt 以 "<|im_start|>assistant\n" 结尾，可以用它来定位 answer 起始位置
            prompt_with_answer = prompt + label_answer[b] + "<|im_end|>"
            batch_prompts.append(prompt_with_answer)
        # 5. Tokenizer 处理文本
        model_inputs = self.VLM.tokenizer(batch_prompts, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(left_current_frame.device)
        attention_mask = model_inputs['attention_mask'].to(left_current_frame.device)
        # 5.1 准确计算 prompt_lengths：搜索 "<|im_start|>assistant\n" 的位置
        # 这是 prompt 模板的固定结尾，用它来定位 answer 起始位置
        answer_start_marker = "<|im_start|>assistant\n"
        marker_tokens = self.VLM.tokenizer(answer_start_marker, add_special_tokens=False)['input_ids']
        marker_len = len(marker_tokens)
        for b in range(B):
            # 在 input_ids 中搜索 marker token 序列（从后往前搜索，因为它在 prompt 末尾）
            found = False
            for pos in range(input_ids.shape[1] - marker_len, -1, -1):
                if input_ids[b, pos:pos+marker_len].tolist() == marker_tokens:
                    # 找到 marker，answer 从 marker 之后开始
                    prompt_lengths.append(pos + marker_len)
                    found = True
                    break
            if not found:
                # 回退：从 batch_prompts 中提取 prompt 部分并 tokenize
                prompt_part = batch_prompts[b].rsplit(answer_start_marker, 1)[0] + answer_start_marker
                prompt_tokens = self.VLM.tokenizer(prompt_part, return_tensors='pt')['input_ids']
                prompt_lengths.append(prompt_tokens.shape[1])

        input_embeds = self.VLM.model.language_model.get_input_embeddings()(input_ids)
        # 6. 输入视觉token
        visual_features = []
        for b in range(B):
            # 历史帧特征 - 只添加非全零的帧
            for i in range(N):
                if not zero_frame_mask[b, i]:  # 如果不是全零帧
                    # 添加该历史帧的 token
                    visual_features.append(right_history_video_token[b, i])
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
        # 7. 组织 attention mask
        seq_len = input_ids.shape[1]
        custom_attention_mask = torch.ones((B, seq_len, seq_len), dtype=torch.float32, device=input_ids.device)
        # 7.1 mask掉 answer tokens 对 left 和 depth tokens 的 attention
        for b in range(B):
            # 7.1.1 找出所有visual token的位置
            visual_token_positions = torch.where(mask[b])[0]
            if len(visual_token_positions) == 0:
                continue
            # 7.1.2 找到 left 和 depth token的位置（注意：每个样本的历史帧数量可能不同）
            l_hist = valid_history_counts[b] * right_history_video_token.shape[2]
            l_left = left_current_frame_token.shape[1]
            l_right = right_current_frame_token.shape[2]
            l_dep = depth_token.shape[1]
            left_start_idx = l_hist
            left_end_idx = l_hist + l_left
            depth_start_idx = l_hist + l_left + l_right
            depth_end_idx = l_hist + l_left + l_right + l_dep
            left_positions = visual_token_positions[left_start_idx:left_end_idx]
            depth_positions = visual_token_positions[depth_start_idx:depth_end_idx]
            # 7.1.3 mask掉answer tokens对left和depth tokens的attention
            answer_start_pos = prompt_lengths[b]
            if len(left_positions) > 0:
                custom_attention_mask[b, answer_start_pos:, left_positions] = 0
            if len(depth_positions) > 0:
                custom_attention_mask[b, answer_start_pos:, depth_positions] = 0
        # 7.2 应用 causal mask - prompt 部分双向 attention，answer 部分 causal attention
        # 重要：prompt tokens 不能看到 answer tokens，否则推理时会不一致！
        causal_mask = torch.ones((B, seq_len, seq_len), device=input_ids.device)
        for b in range(B):
            answer_start_pos = prompt_lengths[b]
            if answer_start_pos < seq_len:
                # 关键修复：prompt tokens 不能看到 answer tokens
                causal_mask[b, :answer_start_pos, answer_start_pos:] = 0
                # Answer 部分应用 causal mask（只能看之前的）
                answer_len = seq_len - answer_start_pos
                causal_mask[b, answer_start_pos:, answer_start_pos:] = torch.tril(
                    torch.ones((answer_len, answer_len), device=input_ids.device)
                )
        custom_attention_mask = custom_attention_mask * causal_mask
        del causal_mask  # 立即释放
        # 7.3 应用 padding mask
        padding_mask = attention_mask.unsqueeze(1).expand(B, seq_len, seq_len)
        custom_attention_mask = custom_attention_mask * padding_mask
        del padding_mask  # 立即释放
        # 7.4 转换为适合模型的格式 (0 -> -inf, 1 -> 0)，并转为4D mask
        # 先转换 dtype 以避免精度问题
        custom_attention_mask = custom_attention_mask.to(input_embeds.dtype)
        custom_attention_mask = (1.0 - custom_attention_mask) * torch.finfo(custom_attention_mask.dtype).min
        # 添加head维度：[B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
        custom_attention_mask = custom_attention_mask.unsqueeze(1)
        # 8. 位置编码
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        position_ids = position_ids.to(input_embeds.device)
        # 9. 输入到 VLM Backbone 中（使用混合精度 + Gradient Checkpointing 节省显存）
        # 使用 bfloat16 混合精度可额外节省约 50% 显存
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self.VLM.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=custom_attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        del custom_attention_mask  # 立即释放大型 attention mask
        del input_embeds  # 释放 input embeddings
        # 10. 整理输出token
        output_tokens = output.hidden_states[-1]
        del output
        l_left  = left_current_frame_token.shape[1]
        l_right = right_current_frame_token.shape[2]
        l_dep   = depth_token.shape[1]
        left_current_output_tokens_list = []
        right_current_output_tokens_list = []
        depth_output_tokens_list = []

        for b in range(B):
            # 获取该样本的所有 visual tokens
            visual_token_mask = mask[b]
            sample_visual_tokens = output_tokens[b][visual_token_mask]  # [total_visual_tokens, hidden_dim]
            l_hist_b = valid_history_counts[b] * right_history_video_token.shape[2]

            left_start = l_hist_b
            left_end = l_hist_b + l_left
            right_start = left_end
            right_end = right_start + l_right
            depth_start = right_end
            depth_end = depth_start + l_dep
            left_current_output_tokens_list.append(sample_visual_tokens[left_start:left_end])
            right_current_output_tokens_list.append(sample_visual_tokens[right_start:right_end])
            depth_output_tokens_list.append(sample_visual_tokens[depth_start:depth_end])

        left_current_output_tokens = torch.stack(left_current_output_tokens_list, dim=0)
        right_current_output_tokens = torch.stack(right_current_output_tokens_list, dim=0)
        depth_output_tokens = torch.stack(depth_output_tokens_list, dim=0)

        # 11. 输入到 Point Head 层
        left_point = self.LeftPointHead(left_current_output_tokens)
        right_point = self.RightPointHead(right_current_output_tokens)
        # 12. point loss
        left_point_loss = F.smooth_l1_loss(left_point, label_left_point, reduction="mean")
        right_point_loss = F.smooth_l1_loss(right_point, label_right_point, reduction="mean")
        point_loss = (left_point_loss + right_point_loss) / 2.0  # 取平均以保持与其他 loss 量级一致
        # 13. depth 估计 (左视角)
        depth = self.FoundationStereo.FoundationStereoDecoder(depth_feature, depth_output_tokens, left_current_frame.squeeze(1), left_vit_feat, features_left, features_right, iters = 10)
        # 14. depth loss
        label_depth_sq = label_depth.squeeze(1)
        valid = torch.isfinite(depth) & torch.isfinite(label_depth_sq) & (label_depth_sq > 0)
        if valid.any():
            depth_loss = F.smooth_l1_loss(depth[valid], label_depth_sq[valid], reduction="mean")
            # 裁剪 depth loss 以防止极端值破坏训练
            depth_loss = torch.clamp(depth_loss, max=10.0)
        else:
            depth_loss = (depth * 0.0).sum()
            depth_loss = depth_loss.to(depth.dtype)  # 确保类型一致
        # 15. 语言性动作估计 & loss
        # 优化：创建 answer mask，一次性计算所有 answer tokens 的 logits
        # 这样避免在循环中多次调用 lm_head，解决 DDP 参数重用问题
        answer_mask_2d = torch.zeros((B, input_ids.shape[1]), dtype=torch.bool, device=input_ids.device)
        for b in range(B):
            answer_start = prompt_lengths[b]
            # 关键修复：包含 answer_start-1 位置，这样才能训练预测第一个 answer token
            if answer_start > 0 and answer_start < input_ids.shape[1]:
                answer_mask_2d[b, answer_start-1:-1] = True
        # 一次性计算所有位置的 logits（只计算有 answer 的位置）
        if answer_mask_2d.any():
            # 提取所有 answer positions 的 hidden states
            answer_hidden = output_tokens[answer_mask_2d]  # [total_answer_tokens, dim]
            answer_logits = self.VLM.model.language_model.lm_head(answer_hidden)  # [total_answer_tokens, vocab]
            # 创建对应的 labels mask
            label_mask_2d = torch.zeros((B, input_ids.shape[1]), dtype=torch.bool, device=input_ids.device)
            for b in range(B):
                answer_start = prompt_lengths[b]
                # 关键修复：从 answer_start 开始，这样 label 包含第一个 answer token
                if answer_start > 0 and answer_start < input_ids.shape[1]:
                    label_mask_2d[b, answer_start:] = True
            # 提取对应的 labels 和 attention mask
            answer_labels = input_ids[label_mask_2d]  # [total_answer_tokens]
            answer_attention = attention_mask[label_mask_2d]  # [total_answer_tokens]
            # 计算 loss
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(answer_logits, answer_labels)
            masked_loss = per_token_loss * answer_attention.float()

            if answer_attention.sum() > 0:
                language_loss = masked_loss.sum() / answer_attention.sum()
            else:
                language_loss = torch.tensor(0.0, device=input_ids.device)
        else:
            language_loss = torch.tensor(0.0, device=input_ids.device)
        # 16. 返回所有losses
        return {
            'point_loss': point_loss,
            'depth_loss': depth_loss,
            'language_loss': language_loss,
        }


    @torch.no_grad()
    def inference(
        self,
        instruction: List[str],
        history_action: List[str],
        left_current_frame: torch.Tensor,   # [B, 1, 3, 448, 448] - 0~255
        right_current_frame: torch.Tensor,  # [B, 1, 3, 448, 448] - 0~255
        right_history_video: torch.Tensor,  # [B, 8, 3, 448, 448] - 0~255
        max_new_tokens: int = 32,
    ) -> dict:
        B = right_history_video.shape[0]
        N = right_history_video.shape[1]  # 历史帧数量
        # 0. 检测每个样本中哪些历史帧是全零的
        zero_frame_mask = (right_history_video.view(B, N, -1).sum(dim=-1) == 0)
        # 1. 深度编码（左右当前帧） -> 获取左视角深度
        depth_feature, depth_token, left_vit_feat, features_left, features_right = self.FoundationStereo.FoundationStereoEncoder(left_current_frame.squeeze(1), right_current_frame.squeeze(1))
        # 2. 左当前帧 点目标编码
        left_current_frame_token = self.DINO.DINOv2Encoder(left_current_frame.squeeze(1))
        # 3. 右当前帧&历史帧 语义编码
        right_history_video_token, right_current_frame_token = self.VLM.Encoder_Vsion(
            torch.cat([right_history_video, right_current_frame], dim=1)
        )
        # 4. 组织 prompt
        batch_prompts = []
        valid_history_counts = []
        for b in range(B):
            # 4.1. 历史帧 - 只处理非全零的帧
            history_video_str = ""
            valid_frame_count = 0
            for i in range(N):
                if zero_frame_mask[b, i]:
                    continue
                valid_frame_count += 1
                t_count = right_history_video_token.shape[2]
                history_video_str += f"Frame {valid_frame_count}: <img>" + "<IMG_CONTEXT>" * t_count + "</img>\n"
            valid_history_counts.append(valid_frame_count)
            # 4.2. 左视角
            t_dino = left_current_frame_token.shape[1]
            left_frame_str = "<img>" + "<IMG_CONTEXT>" * t_dino + "</img>"
            # 4.3. 右视角
            t_vlm = right_current_frame_token.shape[2]
            right_frame_str = "<img>" + "<IMG_CONTEXT>" * t_vlm + "</img>"
            # 4.4. 深度
            t_dep = depth_token.shape[1]
            depth_frame_str = "<dep>" + "<IMG_CONTEXT>" * t_dep + "</dep>"
            # 4.5 组织 text instruction
            instruction_text = other.format(
                instruction=instruction[b],
                history_action_description=history_action_description,
                history_action=history_action[b]
            )
            # 4.6 填入模板
            prompt = temple.format(
                system_description=system_description,
                history_description=history_description,
                history_video=history_video_str,
                current_left_description=current_left_description,
                current_left_frame=left_frame_str,
                current_right_description=current_right_description,
                current_right_frame=right_frame_str,
                depth_description=depth_description,
                depth_frame=depth_frame_str,
                other=instruction_text
            )
            batch_prompts.append(prompt)
        # 5. Tokenizer 处理文本
        model_inputs = self.VLM.tokenizer(batch_prompts, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(left_current_frame.device)
        attention_mask = model_inputs['attention_mask'].to(left_current_frame.device)
        input_embeds = self.VLM.model.language_model.get_input_embeddings()(input_ids)
        # 6. 输入视觉token
        visual_features = []
        for b in range(B):
            # 历史帧特征 - 只添加非全零的帧
            for i in range(N):
                if not zero_frame_mask[b, i]:
                    visual_features.append(right_history_video_token[b, i])
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
        # 7. 位置编码
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        position_ids = position_ids.to(input_embeds.device)
        # 8. 构建与训练一致的 attention mask（prompt 部分双向 attention）
        seq_len = input_ids.shape[1]
        custom_attention_mask = torch.ones((B, seq_len, seq_len), dtype=torch.float32, device=input_embeds.device)
        # 应用 padding mask
        padding_mask = attention_mask.unsqueeze(1).expand(B, seq_len, seq_len)
        custom_attention_mask = custom_attention_mask * padding_mask
        # 转换格式 (0 -> -inf, 1 -> 0)
        custom_attention_mask = custom_attention_mask.to(input_embeds.dtype)
        custom_attention_mask = (1.0 - custom_attention_mask) * torch.finfo(custom_attention_mask.dtype).min
        custom_attention_mask = custom_attention_mask.unsqueeze(1)  # [B, 1, seq_len, seq_len]
        # 9. 获取 hidden states 用于 point/depth prediction
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self.VLM.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=custom_attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=True,
            )

        # 保存 past_key_values 和 logits 供后续生成使用，避免重复 forward pass
        cached_past_key_values = output.past_key_values
        cached_logits = output.logits[:, -1, :]  # 最后一个位置的 logits，用于生成第一个 token
        output_tokens = output.hidden_states[-1]
        # 9. 提取视觉 tokens 用于 point 和 depth 预测
        l_left = left_current_frame_token.shape[1]
        l_right = right_current_frame_token.shape[2]
        l_dep = depth_token.shape[1]
        left_current_output_tokens_list = []
        right_current_output_tokens_list = []
        depth_output_tokens_list = []
        for b in range(B):
            visual_token_mask = mask[b]
            sample_visual_tokens = output_tokens[b][visual_token_mask]
            l_hist_b = valid_history_counts[b] * right_history_video_token.shape[2]
            left_start = l_hist_b
            left_end = l_hist_b + l_left
            right_start = left_end
            right_end = right_start + l_right
            depth_start = right_end
            depth_end = depth_start + l_dep
            left_current_output_tokens_list.append(sample_visual_tokens[left_start:left_end])
            right_current_output_tokens_list.append(sample_visual_tokens[right_start:right_end])
            depth_output_tokens_list.append(sample_visual_tokens[depth_start:depth_end])
        left_current_output_tokens = torch.stack(left_current_output_tokens_list, dim=0)
        right_current_output_tokens = torch.stack(right_current_output_tokens_list, dim=0)
        depth_output_tokens = torch.stack(depth_output_tokens_list, dim=0)
        # 10. Point Head 预测
        left_point = self.LeftPointHead(left_current_output_tokens)
        right_point = self.RightPointHead(right_current_output_tokens)
        # 11. Depth 预测
        depth = self.FoundationStereo.FoundationStereoDecoder(
            depth_feature, depth_output_tokens, left_current_frame.squeeze(1),
            left_vit_feat, features_left, features_right, iters=10
        )
        # 12. 手动生成动作序列（与训练时保持一致的 attention mask）
        # 找出 left 和 depth token 的位置，生成时 mask 掉对它们的 attention
        img_context_token_id = self.VLM.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        eos_token_id = self.VLM.tokenizer.convert_tokens_to_ids("<|im_end|>")
        pad_token_id = self.VLM.tokenizer.pad_token_id

        # 预计算每个样本中 left 和 depth token 的位置
        left_positions_list = []
        depth_positions_list = []
        for b in range(B):
            visual_token_positions = torch.where(mask[b])[0]
            l_hist_b = valid_history_counts[b] * right_history_video_token.shape[2]
            left_start_idx = l_hist_b
            left_end_idx = l_hist_b + l_left
            depth_start_idx = l_hist_b + l_left + l_right
            depth_end_idx = l_hist_b + l_left + l_right + l_dep
            left_positions_list.append(visual_token_positions[left_start_idx:left_end_idx])
            depth_positions_list.append(visual_token_positions[depth_start_idx:depth_end_idx])
        # 手动生成循环（复用步骤 9 的 past_key_values，避免重复 forward pass）
        generated_ids_list = [[] for _ in range(B)]  # 每个样本生成的 token ids
        finished = [False] * B
        past_key_values = cached_past_key_values  # 复用步骤 9 的 KV cache
        logits = cached_logits  # 复用步骤 9 的 logits 作为第一个 token 的预测

        for step in range(max_new_tokens):
            if all(finished):
                break

            # Greedy decoding
            next_token_ids = torch.argmax(logits, dim=-1)  # [B]

            # 更新生成结果
            for b in range(B):
                if not finished[b]:
                    token_id = next_token_ids[b].item()
                    generated_ids_list[b].append(token_id)
                    if token_id == eos_token_id:
                        finished[b] = True

            if all(finished):
                break

            # 准备下一步的输入
            next_token_embeds = self.VLM.model.language_model.get_input_embeddings()(next_token_ids).unsqueeze(1)  # [B, 1, dim]

            # 构建 attention mask：新 token 可以看到 prompt 中除了 left/depth 之外的所有 token
            total_len = past_key_values[0][0].shape[2] + 1  # past_len + 1
            # 创建 attention mask [B, 1, 1, total_len]
            attn_mask = torch.ones((B, 1, 1, total_len), dtype=next_token_embeds.dtype, device=next_token_embeds.device)
            # mask 掉 left 和 depth positions
            for b in range(B):
                if not finished[b]:
                    if len(left_positions_list[b]) > 0:
                        attn_mask[b, 0, 0, left_positions_list[b]] = 0
                    if len(depth_positions_list[b]) > 0:
                        attn_mask[b, 0, 0, depth_positions_list[b]] = 0
            # 转换格式 (0 -> -inf, 1 -> 0)
            attn_mask = (1.0 - attn_mask) * torch.finfo(attn_mask.dtype).min

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.VLM.model.language_model(
                    inputs_embeds=next_token_embeds,
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]

        # 13. 解码生成的 token
        generated_texts = []
        for b in range(B):
            if len(generated_ids_list[b]) > 0:
                text = self.VLM.tokenizer.decode(generated_ids_list[b], skip_special_tokens=False)
                text = text.replace("<|im_end|>", "").strip()
            else:
                text = ""
            generated_texts.append(text)

        return {
            'left_point': left_point,
            'right_point': right_point,
            'depth': depth,
            'generated_text': generated_texts,
        }




