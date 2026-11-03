from dataclasses import dataclass, field
from typing import List

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    '''Experiment group A - test different audio encoders'''
    name: str = "dinov2_distilhubert"

    epochs: int = 20
    audio_model_name: str = "ntu-spml/distilhubert"
    wandb_group: str = "audio_encoder"
    wandb_tags: List[str] = field(default_factory=lambda: ["dinov2", "distilhubert", "infonce", "temp_on", "linear", "frozen_audio", "frozen_image"])