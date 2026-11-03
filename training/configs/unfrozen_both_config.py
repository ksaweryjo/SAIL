from dataclasses import dataclass, field
from typing import List

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    '''Experiment group F - test unfrozen encoders'''
    name: str = "unfrozen_both"

    epochs: int = 20
    audio_model_name: str = "ntu-spml/distilhubert"
    visual_model_name: str = "dinov2"
    freeze_audio_backbone: bool = False
    freeze_visual_backbone: bool = False
    loss_type: str = "multi_positive"
    projection_type: str = "mlp"
    learn_temperature: bool = False
    wandb_group: str = "unfrozen"
    wandb_tags: List[str] = field(default_factory=lambda: ["dinov2", "distilhubert", "multipos", "temp_off", "mlp", "unfrozen_audio", "unfrozen_image"])