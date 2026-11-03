from dataclasses import dataclass, field
from typing import List

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    '''Experiment group B - test different visual encoders with the best performing audio encoder'''
    name: str = "clip_distilhubert"

    epochs: int = 20
    visual_model_name: str = "clip"
    audio_model_name: str = "ntu-spml/distilhubert"
    wandb_group: str = "image_encoder"
    wandb_tags: List[str] = field(default_factory=lambda: ["clip", "distilhubert", "infonce", "temp_on", "linear", "frozen_audio", "frozen_image"])