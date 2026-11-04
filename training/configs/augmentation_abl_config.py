from dataclasses import dataclass, field
from typing import List, Tuple

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    '''Experiment group G - test heavier image augmentation'''
    name: str = "heavy_aug"
    epochs: int = 20

    # Experiment parameters
    crop_scale: Tuple[float, float] | None = (0.5, 1.0)
    hflip_p: float | None = 0.5
    jitter_params: Tuple[float, float, float, float] | None = (0.4, 0.4, 0.2, 0.1)
    
    # "best" model from previous experiment groups
    freeze_audio_backbone: bool = False
    loss_type: str = "multi_positive"
    
    # wandb
    wandb_group: str = "augmentation_ablations"
    wandb_tags: List[str] = field(default_factory=lambda: ["dinov2", "distilhubert", "multipos", "temp_on", "linear", "unfrozen_audio", "frozen_image"])