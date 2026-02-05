import torch
import random
import yaml
import logging
import os
import json
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import wandb
import math
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
from datetime import datetime
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

from model.StereoVLNConfig import StereoVLNConfig
from model.StereoVLN import StereoVLN
from transformers import get_scheduler
from data.datasetloader import Dataset_Normal
from utils.save import save_model_hook

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


logger = logging.getLogger(__name__)
logging.getLogger("accelerate").setLevel(logging.ERROR)


def setup_logging(rank, save_path):
    logging.basicConfig(level=logging.INFO, format=f'[Rank {rank}] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter(f'[Rank {rank}] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if rank == 0:
        log_file = os.path.join(save_path, 'training.log')
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter) 
        logging.getLogger().addHandler(file_handler) 


def build_model_and_optimizer(config, num_all_episodes, world_size):
    # 1. 加载相机参数
    data_root = config.main.data_root
    train_dir = os.path.join(data_root, "train")
    subdir = os.listdir(train_dir)[0] 
    param_dir = os.path.join(train_dir, subdir)
    intrinsics_path = os.path.join(param_dir, "intrinsics.txt")
    camera_k = torch.tensor(
        np.loadtxt(intrinsics_path), dtype=torch.float32
    )
    baseline_path = os.path.join(param_dir, "baseline.txt")
    camera_baseline = float(np.loadtxt(baseline_path))
    # 2. 创建模型
    model_config = StereoVLNConfig(
        image_size = (config.main.image_size, config.main.image_size),
        dtype = torch.bfloat16 if config.main.dtype == "bf16" else torch.float16,
        dim = config.model.dim,

        # VLM Setting
        max_tokens = config.model.vlm.max_tokens,
        vlm_checkpoints_path = config.model.vlm.vlm_checkpoints_path,

        # Depth Estimation Setting
        camera_k = camera_k,
        camera_baseline = camera_baseline,
        foundationstereo_checkpoints_path = config.model.foundationstereo.foundationstereo_checkpoints_path,

        # Point Head Setting
        mlp_ratio = config.model.pointhead.mlp_ratio,
        dropout = config.model.pointhead.dropout
    )
    model = StereoVLN(model_config)
    # 2. 创建 optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
        betas=tuple(config.training.optimizer.betas),
        eps=config.training.optimizer.eps
    )
    # 3. 创建 scheduler
    gloal_batch_size = config.main.batch_size * config.main.gradient.grad_accumulation_steps * world_size
    max_training_steps = math.ceil((config.main.training_epoch * num_all_episodes) / gloal_batch_size)
    scheduler = get_scheduler(
        name=config.training.scheduler.type,  
        optimizer=optimizer,
        num_warmup_steps=int(max_training_steps * config.training.scheduler.warmup_ratio),
        num_training_steps=max_training_steps

    )
    return model, optimizer, scheduler, max_training_steps


def build_dataloader(config, world_size, rank):
    def seed_worker(worker_id):
        worker_seed = 42 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    # 1. 加载数据
    train_dataset = Dataset_Normal(config)
    # 2. 分布式采样器
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    else:
        train_sampler = None
    # 3. 构建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = config.main.batch_size,
        shuffle = True if train_sampler is None else False,
        sampler = train_sampler,
        num_workers = config.main.cpu_workers_num,
        pin_memory = True,
        drop_last = True,
        worker_init_fn = seed_worker,
    )

    return train_dataloader, train_dataset.num_episodes




def learning():
    # 1. 加载配置参数 & 加载保存根目录
    config = OmegaConf.load('train.yaml')
    os.makedirs(config.main.save_root, exist_ok=True)
    # 2. 配置分布式
    accelerator = Accelerator(
        gradient_accumulation_steps = config.main.gradient.grad_accumulation_steps,
        mixed_precision = config.main.dtype,
        project_dir = config.main.save_root,
        project_config = ProjectConfiguration(total_limit= 20),
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=3600))]
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    save_path = os.path.join(config.main.save_root, datetime.now().strftime("%Y-%m-%d_%H"))
    os.makedirs(save_path, exist_ok=True)
    setup_logging(rank, save_path)

    # 3. 加载数据
    train_dataloader, num_all_episodes = build_dataloader(config, world_size, rank)
    # 4. 加载模型和优化器
    if rank == 0:
        print("Loading StereoVLN Model ... ")
    model, optimizer, scheduler, max_training_steps = build_model_and_optimizer(config, num_all_episodes, world_size)
    # 5. 配置模型保存设置
    accelerator.register_save_state_pre_hook(lambda models, weights, output_dir: save_model_hook(models, weights, output_dir, accelerator))
    # 6. 分布式分发 (包括 dataloader)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    # 7. 初始化 wandb (仅在主进程)
    if rank == 0:
        wandb.init(
            project=config.main.get("wandb_project", "StereoVLN"),
            name=config.main.get("wandb_run_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
            config=OmegaConf.to_container(config, resolve=True),
            dir=save_path
        )
    # 8. 训练
    if rank == 0:
        print("Start Training ...")
    epoch = 0
    global_step = 0 # 记录了实际更新的步数 （一个全局batch_size更新一次）
    data_iter = iter(train_dataloader)
    epoch_completed = False
    accumulated_total_loss = 0.0
    accumulated_point_loss = 0.0
    accumulated_depth_loss = 0.0
    accumulated_language_loss = 0.0
    accumulation_count = 0
    while (global_step < max_training_steps):
        ## 8.0. 配置为训练模式
        model.train()
        ## 8.1. 加载一个batch数据
        try:
            batch = next(data_iter)
            epoch_completed = False  # 重置标志
        except StopIteration:
            epoch += 1
            epoch_completed = True  # 标记 epoch 完成
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        ## 8.2 预处理batch
        device = accelerator.device
        dtype = torch.bfloat16 if config.main.dtype == "bf16" else torch.float16
        batch["left_current_frame"] = batch["left_current_frame"].to(device, dtype=dtype)
        batch["right_current_frame"] = batch["right_current_frame"].to(device, dtype=dtype)
        batch["right_history_video"] = batch["right_history_video"].to(device, dtype=dtype)
        batch["label_left_point"] = batch["label_left_point"].to(device, dtype=dtype)
        batch["label_right_point"] = batch["label_right_point"].to(device, dtype=dtype)
        batch["label_depth"] = batch["label_depth"].to(device, dtype=dtype)
        ## 8.3 前向传播
        with accelerator.accumulate(model):
            outputs = model(
                instruction=batch["instruction"],
                history_action=batch["history_action"],
                left_current_frame=batch["left_current_frame"],
                right_current_frame=batch["right_current_frame"],
                right_history_video=batch["right_history_video"],
                label_left_point=batch["label_left_point"],
                label_right_point=batch["label_right_point"],
                label_depth=batch["label_depth"],
                label_answer=batch["label_answer"]
            )
            total_loss = config.training.weight_pointsloss * outputs['point_loss'] + config.training.weight_depthloss * outputs['depth_loss'] + config.training.weight_languageloss * outputs['language_loss']
            accumulated_total_loss += total_loss.detach()
            accumulated_point_loss += outputs['point_loss'].detach()
            accumulated_depth_loss += outputs['depth_loss'].detach()
            accumulated_language_loss += outputs['language_loss'].detach()
            accumulation_count += 1
            ## 8.4 反向传播
            accelerator.backward(total_loss)
            ## 8.5 梯度裁剪
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.main.gradient.grad_clip_norm)
            ## 8.6 更新参数
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1 
        ## 8.7 记录日志
        if accelerator.sync_gradients:
            # 计算平均值
            avg_total_loss = accumulated_total_loss / accumulation_count
            avg_point_loss = accumulated_point_loss / accumulation_count
            avg_depth_loss = accumulated_depth_loss / accumulation_count
            avg_language_loss = accumulated_language_loss / accumulation_count
            
            if rank == 0:
                logging.info(
                    f"Epoch: {epoch} | "
                    f"Step: {global_step}/{max_training_steps} | "
                    f"Total Loss: {avg_total_loss.item():.4f} | "
                    f"Point Loss: {avg_point_loss.item():.4f} | "
                    f"Depth Loss: {avg_depth_loss.item():.4f} | "
                    f"Language Loss: {avg_language_loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                wandb.log({
                    "train/total_loss": avg_total_loss.item(),
                    "train/point_loss": avg_point_loss.item(),
                    "train/depth_loss": avg_depth_loss.item(),
                    "train/language_loss": avg_language_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }, step=global_step)
            
            # 重置累积变量
            accumulated_total_loss = 0.0
            accumulated_point_loss = 0.0
            accumulated_depth_loss = 0.0
            accumulated_language_loss = 0.0
            accumulation_count = 0        
        ## 8.8 保存模型
        if epoch_completed:
            if rank == 0:
                logging.info(f"Saving model at epoch {epoch}, step {global_step}...")
            checkpoint_dir = os.path.join(save_path, "checkpoint_" + str(epoch))
            accelerator.save_state(str(checkpoint_dir))
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
            # 2. 保存配置
            cfg = OmegaConf.to_container(config, resolve=True)
            with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                json.dump(cfg, f, indent=2)
            dist.barrier()
    

    # 9. 关闭 wandb
    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    learning()


    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 8 --num_machines 1 --mixed_precision fp16 --dynamo_backend no train.py