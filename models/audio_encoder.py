import torch
import torch.nn as nn
from transformers import HubertModel


class AudioEncoder(nn.Module):
    def __init__(self, model_name: str = "facebook/hubert-large-ls960-ft") -> None:
        super().__init__()
        self.model = HubertModel.from_pretrained(
            model_name, apply_spec_augment=False, layerdrop=0.0
        )
        self.feature_dim = self.model.config.hidden_size

    def freeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).last_hidden_state
