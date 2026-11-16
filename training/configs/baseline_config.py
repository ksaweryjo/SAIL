from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch


@dataclass
class Config:
    name: str = "baseline"
    dataset: str = "coco"
    
    # Img augmentation
    size: int = 224
    crop_scale: Optional[Tuple[float, float]] = (0.9, 1.0)
    hflip_p: Optional[float] = None
    jitter_params: Optional[Tuple[float, float, float, float]] = None
    use_ccrop: bool = True
    
    # Audio augmentation
    sample_rate: int = 16000
    normalize: bool = True
    train_max_sec: float = 9.8133
    val_max_sec: float = 9.6427
    train_max_sec_f: float = 8.192
    val_max_sec_f: float = 8.192
    flickr_val_fraction: float = 0.2

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-6
    warmup_epochs: int = 2
    use_cosine_schedule: bool = True

    # Model
    visual_model_name: str = "dinov2"
    audio_model_name: str = "ntu-spml/distilhubert"
    freeze_audio_backbone: bool = True
    freeze_visual_backbone: bool = True
    shared_dim: int = 512
    projection_type: str = "linear"
    dropout: float = 0.1
    loss_type: str = "infonce"

    # Training
    epochs: int = 10
    batch_size: int = 32
    use_amp: bool = True # we will default to bf16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learn_temperature: bool = True
    init_temperature: float = 0.07
    
    ## Sampler
    max_train_pairs: Optional[int] = None
    max_val_pairs: Optional[int] = None
    sample_seed: int = 123

    # Paths
    data_dir: Path = Path("data/spokencoco")
    train_image_dir: Path = data_dir / "train2014"
    train_audio_dir: Path = data_dir / "wavs/train"
    train_json_file: Path = data_dir / "SpokenCOCO_train.json"
    val_image_dir: Path = data_dir / "val2014"
    val_audio_dir: Path = data_dir / "wavs/val"
    val_json_file: Path = data_dir / "SpokenCOCO_val.json"
    checkpoint_dir: str = "checkpoints"

    # Logging
    use_wandb: bool = True
    log_every_n_steps: int = 50
    wandb_entity: str = "hush-ai"
    wandb_project: str = "Speech-To-Image-SAIL"
    wandb_group: str = "baseline"
    wandb_tags: List[str] = field(default_factory=lambda: ["baseline"])
    
    grad_clip_norm: Optional[float] = 1.0