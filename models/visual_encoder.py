import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, Dinov2Model


class VisualEncoder(nn.Module):
    def __init__(self, model_name: str = "dinov2") -> None:
        super().__init__()
        if model_name == "dinov2":
            self.model = Dinov2Model.from_pretrained("facebook/dinov2-base")
            self.feature_dim = self.model.config.hidden_size
        elif model_name == "clip":
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch16"
            )
            self.feature_dim = self.model.config.projection_dim
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def freeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, Dinov2Model):
            return self.model(x).last_hidden_state[
                :, 0
            ]  # Use the CLS token representation
        elif isinstance(self.model, CLIPVisionModelWithProjection):
            return self.model(x).image_embeds
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")
