from dataclasses import dataclass, field
from typing import List

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    '''Experiment group E - test projection type ablation'''
    name: str = "projection_type"
    epochs: int = 20
    
    # Experiment parameters
    projection_type: str = "mlp"
    
    # "best" model from previous experiment groups
    loss_type: str = "multi_positive"
    
    # wandb
    wandb_group: str = "projection_type"
    wandb_tags: List[str] = field(default_factory=lambda: ["dinov2", "distilhubert", "multipos", "temp_on", "mlp", "frozen_audio", "frozen_image"])