import torch
import torch.nn.functional as F


def retrieval_metrics(
    audio_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    image_ids: torch.Tensor | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    if audio_embeddings.ndim != 2 or image_embeddings.ndim != 2:
        raise ValueError("Retrieval embeddings must have shape [batch, dim]")
    if audio_embeddings.shape != image_embeddings.shape:
        raise ValueError("Audio and image embeddings must have identical shapes")
    if audio_embeddings.size(0) == 0:
        raise ValueError("Retrieval metrics require at least one sample")
    if image_ids is None:
        image_ids = torch.arange(
            audio_embeddings.size(0), device=audio_embeddings.device
        )
    if image_ids.ndim != 1 or image_ids.size(0) != audio_embeddings.size(0):
        raise ValueError("image_ids must have one value per embedding")

    audio_embeddings = F.normalize(audio_embeddings, dim=-1)
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    similarity = audio_embeddings @ image_embeddings.T
    positive_mask = image_ids[:, None] == image_ids[None, :]
    metrics: dict[str, float] = {}

    for direction, scores in (("a2i", similarity), ("i2a", similarity.T)):
        ranked_positive = positive_mask.gather(
            1, scores.argsort(dim=1, descending=True)
        )
        ranks = ranked_positive.float().argmax(dim=1) + 1
        for k in ks:
            hits = ranked_positive[:, : min(k, ranked_positive.size(1))].any(dim=1)
            metrics[f"recall_{direction}@{k}"] = hits.float().mean().item()
        metrics[f"mrr_{direction}"] = (1.0 / ranks).mean().item()
        metrics[f"median_rank_{direction}"] = float(ranks.median().item())

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
