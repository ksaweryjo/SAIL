from dataclasses import dataclass, field
from typing import List

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    '''Experiment group D - test if learnable temperature is beneficial'''
    name: str = "temp_off"
    epochs: int = 20
    
    # Experiment parameters
    learn_temperature: bool = False
    
    # "best" model from previous experiment groups
    loss_type: str = "multi_positive"
    
    # wandb
    wandb_group: str = "temp"
    wandb_tags: List[str] = field(default_factory=lambda: ["dinov2", "distilhubert", "multipos", "temp_off", "linear", "frozen_audio", "frozen_image"])