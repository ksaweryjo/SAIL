from dataclasses import dataclass, field
from typing import List

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    '''Experiment group F - test unfrozen encoders'''
    name: str = "unfrozen_audio"

    epochs: int = 20
    audio_model_name: str = "ntu-spml/distilhubert"
    freeze_audio_backbone: bool = False
    loss_type: str = "multi_positive"
    projection_type: str = "mlp"
    wandb_group: str = "unfrozen"
    wandb_tags: List[str] = field(default_factory=lambda: ["dinov2", "distilhubert", "multipos", "temp_on", "mlp", "unfrozen_audio", "frozen_image"])