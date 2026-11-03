import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE_Loss(nn.Module):
    def __init__(
        self, learn_temperature: bool = True, init_temperature: float = 0.07
    ) -> None:
        super().__init__()
        if learn_temperature:
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(1 / init_temperature))
            )
        else:
            self.register_buffer(
                "logit_scale", torch.tensor(math.log(1 / init_temperature))
            )
        self.max_logit_scale = math.log(100.0)

    def forward(
        self,
        audio_emb: torch.Tensor,
        img_emb: torch.Tensor,
        image_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss, _, _, logits = self.compute_components(audio_emb, img_emb, image_ids)
        return loss, logits

    def compute_components(
        self,
        audio_emb: torch.Tensor,
        img_emb: torch.Tensor,
        image_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_embeddings(audio_emb, img_emb)
        audio_emb = F.normalize(audio_emb, dim=-1)
        img_emb = F.normalize(img_emb, dim=-1)

        local_scale = self.logit_scale.clamp(max=self.max_logit_scale)
        logits = torch.exp(local_scale) * (audio_emb @ img_emb.T)  # logits per audio-image pair

        labels = torch.arange(logits.size(0), device=audio_emb.device)
        loss_a2i = F.cross_entropy(logits, labels)
        loss_i2a = F.cross_entropy(logits.T, labels)  # symmetric loss for image-audio pairs
        loss = (loss_a2i + loss_i2a) / 2

        return loss, loss_a2i, loss_i2a, logits


class MultiPositiveInfoNCE_Loss(InfoNCE_Loss):
    def compute_components(
        self,
        audio_emb: torch.Tensor,
        img_emb: torch.Tensor,
        image_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_embeddings(audio_emb, img_emb)
        if image_ids is None or image_ids.ndim != 1:
            raise ValueError("Multi-positive loss requires one image ID per sample")
        if image_ids.size(0) != audio_emb.size(0):
            raise ValueError("image_ids must have the same batch size as embeddings")

        audio_emb = F.normalize(audio_emb, dim=-1)
        img_emb = F.normalize(img_emb, dim=-1)
        local_scale = self.logit_scale.clamp(max=self.max_logit_scale)
        logits = torch.exp(local_scale) * (audio_emb @ img_emb.T)
        positive_mask = image_ids[:, None] == image_ids[None, :]
        positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
        loss_a2i = (
            torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
        ).mean()
        loss_i2a = (
            torch.logsumexp(logits.T, dim=1)
            - torch.logsumexp(positive_logits.T, dim=1)
        ).mean()
        loss = (loss_a2i + loss_i2a) / 2
        return loss, loss_a2i, loss_i2a, logits


def _validate_embeddings(
    audio_emb: torch.Tensor, img_emb: torch.Tensor
) -> None:
    if audio_emb.ndim != 2 or img_emb.ndim != 2:
        raise ValueError("Audio and image embeddings must have shape [batch, dim]")
    if audio_emb.shape != img_emb.shape:
        raise ValueError(
            "Audio and image embeddings must have identical batch and feature dimensions"
        )
