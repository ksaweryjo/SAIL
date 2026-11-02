import torch
import torch.nn.functional as F


def retrieval_metrics(
    audio_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    audio_embeddings = F.normalize(audio_embeddings, dim=-1)
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    similarity = audio_embeddings @ image_embeddings.T
    target = torch.arange(similarity.size(0), device=similarity.device)
    metrics: dict[str, float] = {}

    for direction, scores in (("a2i", similarity), ("i2a", similarity.T)):
        ranks = (scores.argsort(dim=1, descending=True) == target.unsqueeze(1)).nonzero(
            as_tuple=False
        )[:, 1].float() + 1
        for k in ks:
            metrics[f"recall_{direction}@{k}"] = (ranks <= k).float().mean().item()
        metrics[f"mrr_{direction}"] = (1.0 / ranks).mean().item()
        metrics[f"median_rank_{direction}"] = ranks.median().item()

    positive = similarity.diagonal()
    negative_mask = ~torch.eye(
        similarity.size(0), dtype=torch.bool, device=similarity.device
    )
    negatives = similarity.masked_select(negative_mask)
    metrics["positive_similarity"] = positive.mean().item()
    negative_mean = negatives.mean() if negatives.numel() else torch.zeros_like(
        positive.mean()
    )
    metrics["negative_similarity"] = negative_mean.item()
    metrics["similarity_gap"] = (positive.mean() - negative_mean).item()
    return metrics


def embedding_diagnostics(
    audio_embeddings: torch.Tensor, image_embeddings: torch.Tensor
) -> dict[str, float]:
    return {
        "audio_embedding_norm": audio_embeddings.norm(dim=-1).mean().item(),
        "image_embedding_norm": image_embeddings.norm(dim=-1).mean().item(),
    }
