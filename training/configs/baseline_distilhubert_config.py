from dataclasses import dataclass, field
from typing import List

from training.configs.baseline_config import Config as Baseline


@dataclass
class Config(Baseline):
    name: str = "baseline_distilhubert"

    epochs: int = 20
    audio_model_name: str = "ntu-spml/distilhubert"
    wandb_group: str = "baselines"
    wandb_tags: List[str] = field(default_factory=lambda: ["baseline", "distilhubert"])