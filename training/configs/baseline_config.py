from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


@dataclass
class Config:
    name: str = "baseline"
    
    # Img augmentation
    size: int = 224
    crop_scale: Optional[Tuple[float, float]] = (0.9, 1.0)
    hflip_p: Optional[float] = None
    jitter_params: Optional[Tuple[float, float, float, float]] = None
    use_ccrop: bool = True
    
    # Audio augmentation
    sample_rate: int = 16000
    normalize: bool = True

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-6
    warmup_epochs: int = 1
    use_cosine_schedule: bool = True

    # Model
    visual_model_name: str = "dinov2"
    audio_model_name: str = "facebook/hubert-large-ls960-ft"
    freeze_audio_backbone: bool = True
    freeze_visual_backbone: bool = True
    shared_dim: int = 512
    projection_type: str = "mlp"
    dropout: float = 0.1

    # Training
    epochs: int = 10
    batch_size: int = 256
    use_fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learn_temperature: bool = True
    init_temperature: float = 0.07

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
    wandb_entity: str = "hush-ai"
    wandb_project: str = "Speech-To-Image-SAIL"