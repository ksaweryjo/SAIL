from dataclasses import dataclass, field
from typing import List

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    '''Experiment group E - test projection type ablation'''
    name: str = "projection_type"

    epochs: int = 20
    audio_model_name: str = "ntu-spml/distilhubert"
    loss_type: str = "multi_positive"
    projection_type: str = "mlp"
    learn_temperature: bool = False
    wandb_group: str = "projection_type"
    wandb_tags: List[str] = field(default_factory=lambda: ["dinov2", "distilhubert", "multipos", "temp_off", "mlp", "frozen_audio", "frozen_image"])