from dataclasses import dataclass, field
from typing import List

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    '''Experiment group F - test unfrozen encoders'''
    name: str = "unfrozen_both"
    epochs: int = 20
    
    # Experiment parameters
    freeze_audio_backbone: bool = False
    freeze_visual_backbone: bool = False
    audio_backbone_lr: float | None = 5e-6
    visual_backbone_lr: float | None = 5e-6
    projection_lr: float | None = 1e-4
    
    # "best" model from previous experiment groups
    loss_type: str = "multi_positive"
       
    # wandb
    wandb_group: str = "unfrozen"
    wandb_tags: List[str] = field(default_factory=lambda: ["dinov2", "distilhubert", "multipos", "temp_on", "linear", "unfrozen_audio", "unfrozen_image"])