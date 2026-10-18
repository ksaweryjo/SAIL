import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE_Loss(nn.Module):
    def __init__(self, learn_temperature: bool = True, init_temperature: float = 0.07) -> None:
        super().__init__()
        if learn_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temperature)))
        else:
            self.register_buffer("logit_scale", torch.tensor(math.log(1 / init_temperature)))
        self.max_logit_scale = math.log(100.0)
            
    def forward(self, audio_emb: torch.Tensor, img_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_emb = F.normalize(audio_emb, dim=-1)
        img_emb = F.normalize(img_emb, dim=-1)
        
        local_scale = self.logit_scale.clamp(max=self.max_logit_scale)
        logits = torch.exp(local_scale) * (audio_emb @ img_emb.T) # logits per audio-image pair
        
        labels = torch.arange(logits.size(0), device=audio_emb.device)
        loss_a2i = F.cross_entropy(logits, labels)
        loss_i2a = F.cross_entropy(logits.T , labels) # symmetric loss for image-audio pairs
        loss = (loss_a2i + loss_i2a) / 2
        
        return loss, logits