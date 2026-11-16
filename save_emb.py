import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.contrastive_model import ContrastiveModel
from torch_datasets.flickr_audio_image_dataset import FlickrAudioImageDataset
from torch_datasets.spoken_coco_dataset import SpokenCocoDataset
from torch_datasets.transforms import Audio_ValTransform
from training.checkpointing import load_checkpoint
from training.configs.baseline_config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save pooled AudioEncoder features for SpokenCOCO or Flickr8k."
    )
    parser.add_argument(
        "--dataset",
        choices=("spokencoco", "flickr8k"),
        default="spokencoco",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    project_root = Path(__file__).resolve().parent
    checkpoint = args.checkpoint or (
        project_root / config.checkpoint_dir / "final_model_best.pth"
    )
    if args.dataset == "spokencoco":
        data_dir = (project_root / config.data_dir).resolve()
        output = args.output or (
            project_root
            / config.checkpoint_dir
            / f"{args.split}_audio_embeddings.pt"
        )
        image_dir = data_dir / f"{args.split}2014"
        audio_dir = data_dir / "wavs" / args.split
        annotation_file = data_dir / f"SpokenCOCO_{args.split}.json"
        dataset = SpokenCocoDataset(
            img_dir=image_dir,
            audio_dir=audio_dir,
            ann_file=annotation_file,
            audio_transform=Audio_ValTransform(
                sample_rate=config.sample_rate, normalize=config.normalize
            ),
            sample_rate=config.sample_rate,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
    else:
        output = args.output or (
            project_root / config.checkpoint_dir / "flickr8k_audio_embeddings.pt"
        )
        dataset = FlickrAudioImageDataset(
            split=args.split,
            audio_transform=Audio_ValTransform(
                sample_rate=config.sample_rate, normalize=config.normalize
            ),
            sample_rate=config.sample_rate,
            val_fraction=config.flickr_val_fraction,
            audio_only=True,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

    model = ContrastiveModel(
        visual_model_name=config.visual_model_name,
        audio_model_name=config.audio_model_name,
        shared_dim=config.shared_dim,
        projection_type=config.projection_type,
        dropout=config.dropout,
        freeze_visual_backbone=True,
        freeze_audio_backbone=True,
    )
    load_checkpoint(checkpoint, model)
    model.to(config.device).eval()

    embeddings: dict[str, torch.Tensor] = {}
    offset = 0
    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc=f"Saving {args.dataset} audio embeddings",
            unit="batch",
        )
        for batch in progress:
            if args.dataset == "spokencoco":
                _, audio, lengths, _, _ = batch
                keys = [
                    Path(dataset.data[offset + index][1]).resolve().as_posix()
                    for index in range(audio.size(0))
                ]
            else:
                audio, lengths, keys = batch
            audio = audio.to(config.device)
            lengths = lengths.to(config.device, dtype=torch.long)
            attention_mask = (
                torch.arange(audio.size(1), device=config.device)[None, :]
                < lengths[:, None]
            )
            features, feature_mask = model.audio_encoder(audio, attention_mask)
            mask = feature_mask.unsqueeze(-1).to(features.dtype)
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            for key, feature in zip(keys, pooled.cpu(), strict=True):
                embeddings[key] = feature
            offset += audio.size(0)
            progress.set_postfix(saved=len(embeddings))

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings}, output)
    print(f"Saved {len(embeddings)} audio embeddings to {output}")


if __name__ == "__main__":
    main()
