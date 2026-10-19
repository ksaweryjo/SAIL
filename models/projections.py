import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        projection_type: str = "mlp",
    ) -> None:
        super().__init__()
        if projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
            )
        elif projection_type == "linear":
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
