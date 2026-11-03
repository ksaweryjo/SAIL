from dataclasses import dataclass, field
from typing import List

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    '''Experiment group C - test multi-positive loss'''
    name: str = "multipos_loss"

    epochs: int = 20
    audio_model_name: str = "ntu-spml/distilhubert"
    loss_type: str = "multi_positive"
    wandb_group: str = "loss"
    wandb_tags: List[str] = field(default_factory=lambda: ["dinov2", "distilhubert", "multipos", "temp_on", "linear", "frozen_audio", "frozen_image"])