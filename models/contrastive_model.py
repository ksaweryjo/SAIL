import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_encoder import AudioEncoder
from models.projections import ProjectionHead
from models.visual_encoder import VisualEncoder


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        visual_model_name: str = "dinov2",
        audio_model_name: str = "facebook/hubert-large-ls960-ft",
        shared_dim: int = 512,
        projection_type: str = "mlp",
        dropout: float = 0.1,
        freeze_visual_backbone: bool = False,
        freeze_audio_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.visual_encoder = VisualEncoder(model_name=visual_model_name)
        self.visual_backbone_frozen = freeze_visual_backbone
        if freeze_visual_backbone:
            self.visual_encoder.freeze_backbone()

        self.audio_encoder = AudioEncoder(model_name=audio_model_name)
        self.audio_backbone_frozen = freeze_audio_backbone
        if freeze_audio_backbone:
            self.audio_encoder.freeze_backbone()

        self.audio_projection = ProjectionHead(
            self.audio_encoder.feature_dim, shared_dim, dropout, projection_type
        )
        self.visual_projection = ProjectionHead(
            self.visual_encoder.feature_dim, shared_dim, dropout, projection_type
        )

    def encode_audio(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        if self.audio_backbone_frozen:
            with torch.no_grad():
                feats, feature_mask = self.audio_encoder(audio, attention_mask)
        else:
            feats, feature_mask = self.audio_encoder(audio, attention_mask)
        if feature_mask is None:
            feats = feats.mean(dim=1)
        else:
            mask = feature_mask.unsqueeze(-1).to(feats.dtype)
            feats = (feats * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        emb = self.audio_projection(feats)
        if normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

    def encode_visual(self, visual: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        if self.visual_backbone_frozen:
            with torch.no_grad():
                feats = self.visual_encoder(visual)
        else:
            feats = self.visual_encoder(visual)
        emb = self.visual_projection(feats)
        if normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_emb = self.encode_audio(audio, audio_attention_mask)
        visual_emb = self.encode_visual(visual)
        return audio_emb, visual_emb
