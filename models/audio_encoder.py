from typing import cast

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

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = self.model(x, attention_mask=attention_mask).last_hidden_state
        if attention_mask is None:
            return hidden_states, None

        feature_mask = self.model._get_feature_vector_attention_mask(
            hidden_states.size(1), cast(torch.LongTensor, attention_mask.long())
        )
        return hidden_states, feature_mask
