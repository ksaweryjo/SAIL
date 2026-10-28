import argparse
import importlib
from itertools import chain

import torch
from checkpointing import load_checkpoint
from loss import InfoNCE_Loss
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from models.contrastive_model import ContrastiveModel
from torch_datasets.spoken_coco_dataset import SpokenCocoDataset
from torch_datasets.transforms import (
    Audio_TrainTransform,
    Audio_ValTransform,
    Img_TrainTransform,
    Img_ValTransform,
)
from training.configs.baseline_config import Config


def get_config(name: str) -> Config:
    try:
        module_name = f"training.configs.{name}"
        module = importlib.import_module(module_name)
        if hasattr(module, "Config"):
            config_cls = module.Config
            if isinstance(config_cls, type) and issubclass(config_cls, Config):
                return config_cls()
            elif isinstance(config_cls, Config):
                return config_cls
        print(f"Using config: {name}")
    except (ImportError, AttributeError) as e:
        print(f"Error loading config {name}: {e}")
        print("Using default Config")

    return Config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script for SAIL model")
    parser.add_argument(
        "--config", type=str, default="baseline_config", help="Name of the config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint (provide path)",
    )

    return parser.parse_args()


def build_model(config: Config) -> ContrastiveModel:
    model = ContrastiveModel(
        visual_model_name=config.visual_model_name,
        audio_model_name=config.audio_model_name,
        shared_dim=config.shared_dim,
        projection_type=config.projection_type,
        dropout=config.dropout,
        freeze_visual_backbone=config.freeze_visual_backbone,
        freeze_audio_backbone=config.freeze_audio_backbone,
    ).to(config.device)

    return model


def build_criterion(config: Config) -> InfoNCE_Loss:
    criterion = InfoNCE_Loss(
        learn_temperature=config.learn_temperature,
        init_temperature=config.init_temperature,
    ).to(config.device)
    return criterion


def build_optimizer(
    config: Config, model: ContrastiveModel, criterion: InfoNCE_Loss
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in chain(model.named_parameters(), criterion.named_parameters()):
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        params=[
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
    )
    return optimizer


def build_dataloader(config: Config) -> tuple[DataLoader, DataLoader]:
    train_dataset = SpokenCocoDataset(
        img_dir=config.train_image_dir,
        audio_dir=config.train_audio_dir,
        ann_file=config.train_json_file,
        img_transform=Img_TrainTransform(
            size=config.size,
            crop_scale=config.crop_scale,
            hflip_p=config.hflip_p,
            jitter_params=config.jitter_params,
        ),
        audio_transform=Audio_TrainTransform(
            sample_rate=config.sample_rate, normalize=config.normalize
        ),
    )

    val_dataset = SpokenCocoDataset(
        img_dir=config.val_image_dir,
        audio_dir=config.val_audio_dir,
        ann_file=config.val_json_file,
        img_transform=Img_ValTransform(size=config.size, use_ccrop=config.use_ccrop),
        audio_transform=Audio_ValTransform(
            sample_rate=config.sample_rate, normalize=config.normalize
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Config, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if not config.use_cosine_schedule:
        return None
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    return scheduler


def resume_from_checkpoint(
    checkpoint_path: str,
    model: ContrastiveModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    device: str,
) -> tuple[int, float]:
    epoch, val_loss = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    model.to(device)
    print(f"Resumed from epoch {epoch}, val_loss: {val_loss:.4f}")
    return epoch + 1, val_loss


def main() -> None:
    args = parse_args()
    config = get_config(args.config)
    train_loader, val_loader = build_dataloader(config)
    model = build_model(config)
    criterion = build_criterion(config)
    optimizer = build_optimizer(config, model, criterion)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    if args.resume is not None:
        start_epoch, best_val_loss = resume_from_checkpoint(
            args.resume, model, optimizer, scheduler, config.device
        )
    else:
        start_epoch, best_val_loss = 1, float("inf")


if __name__ == "__main__":
    main()
