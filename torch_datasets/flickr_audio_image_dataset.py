import io
import random
from typing import Callable, Optional

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class FlickrAudioImageDataset(Dataset):
    def __init__(
        self,
        split: str,
        img_transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
        sample_rate: int = 16000,
        max_pairs: Optional[int] = None,
        sample_seed: int = 123,
        val_fraction: float = 0.2,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        if max_pairs is not None and max_pairs < 1:
            raise ValueError("max_pairs must be at least 1 or None")

        self.img_transform = img_transform
        self.audio_transform = audio_transform
        self.sample_rate = sample_rate

        audio_dataset = load_dataset(
            "deep9539/flickr-audio-image", "audio", split="train"
        ).cast_column("audio", Audio(decode=False))
        image_dataset = load_dataset(
            "deep9539/flickr-audio-image", "images", split="train"
        )
        qrels_dataset = load_dataset(
            "deep9539/flickr-audio-image", "qrels", split="train"
        )

        audio_ids = audio_dataset["id"]
        self._audio_by_id = {
            audio_id: index for index, audio_id in enumerate(audio_ids)
        }
        image_ids = list(image_dataset["id"])
        sampler = random.Random(sample_seed)
        sampler.shuffle(image_ids)
        val_count = max(1, round(len(image_ids) * val_fraction))
        val_image_ids = set(image_ids[:val_count])
        selected_image_ids = (
            val_image_ids
            if split == "val"
            else set(image_ids[val_count:])
        )
        self._image_by_id = {
            image_id: index
            for index, image_id in enumerate(image_dataset["id"])
            if image_id in selected_image_ids
        }
        self._image_id_to_int = {
            image_id: index for index, image_id in enumerate(sorted(image_ids))
        }
        self._audio_dataset = audio_dataset
        self._image_dataset = image_dataset

        self.data: list[tuple[int, str, str]] = []
        for image_id, audio_id in zip(
            qrels_dataset["image_id"], qrels_dataset["audio_id"], strict=True
        ):
            if image_id not in self._image_by_id:
                continue
            if audio_id not in self._audio_by_id:
                raise ValueError(f"Audio relation references missing id {audio_id!r}")

            self.data.append(
                (self._audio_by_id[audio_id], image_id, audio_id)
            )

        if max_pairs is not None and max_pairs < len(self.data):
            self.data = sampler.sample(self.data, max_pairs)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, str, int]:
        audio_index, image_id, _ = self.data[idx]
        image_record = self._image_dataset[self._image_by_id[image_id]]
        img = pil_to_tensor(image_record["image"]).float() / 255.0
        if self.img_transform is not None:
            img = self.img_transform(img)

        audio_record = self._audio_dataset[audio_index]["audio"]
        audio_source = audio_record["bytes"]
        if audio_source is None:
            audio_source = audio_record["path"]
            audio, sampling_rate = sf.read(
                audio_source, dtype="float32", always_2d=True
            )
        else:
            audio, sampling_rate = sf.read(
                io.BytesIO(audio_source), dtype="float32", always_2d=True
            )
        if sampling_rate != self.sample_rate:
            raise ValueError(
                f"Unexpected sample rate {sampling_rate} "
                f"for audio id {self.data[idx][2]!r}"
            )
        audio = torch.from_numpy(audio.mean(axis=1))
        if self.audio_transform is not None:
            audio, length = self.audio_transform(audio)
        else:
            length = audio.shape[-1]

        return (
            img,
            audio,
            length,
            image_record["text"],
            self._image_id_to_int[image_id],
        )
