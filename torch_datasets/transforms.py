from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import v2


class Img_TrainTransform:
    def __init__(
        self,
        size: int = 224,
        crop_scale: Optional[Tuple] = (0.5, 1.0),
        hflip_p: Optional[float] = 0.5,
        jitter_params: Optional[Tuple[float, float, float, float]] = (
            0.4,
            0.4,
            0.2,
            0.1,
        ),
        dataset: str = "coco",
    ) -> None:
        self.size = size
        self.crop_scale = crop_scale
        self.hflip_p = hflip_p
        self.jitter_params = jitter_params
        if dataset == "coco":
            mean = (0.47116698, 0.44670493, 0.40732929)
            std = (0.27892825, 0.27424563, 0.28907873)
        elif dataset == "flickr":
            mean = (0.45764304, 0.44532845, 0.40295026)
            std = (0.27939190, 0.27229444, 0.28636890)
        else:
            raise ValueError(f"Unknown dataset {dataset!r}")
        self.transforms_list: list[Callable] = []
        
        if self.crop_scale is not None:
            self.transforms_list.append(
                v2.RandomResizedCrop(size=self.size, scale=self.crop_scale)
            )
        else:
            self.transforms_list.append(v2.Resize(size=self.size))
        
        if self.hflip_p is not None:
            self.transforms_list.append(v2.RandomHorizontalFlip(p=self.hflip_p))
        
        if self.jitter_params is not None:
            self.transforms_list.append(v2.ColorJitter(*self.jitter_params))
        
        self.transforms_list.append(
            v2.Normalize(
                mean=list(mean),
                std=list(std)
                )
            )        
        self.transform = v2.Compose(self.transforms_list)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img)


class Img_ValTransform:
    def __init__(
        self,
        size: int = 224,
        use_ccrop: bool = False,
        dataset: str = "coco",
    ) -> None:
        self.size = size
        if dataset == "coco":
            mean = (0.46761145, 0.44558814, 0.40681093)
            std = (0.27877759, 0.27342102, 0.28721333)
        elif dataset == "flickr":
            mean = (0.46055005, 0.44935683, 0.40774844)
            std = (0.27833551, 0.27052908, 0.28458256)
        else:
            raise ValueError(f"Unknown dataset {dataset!r}")
        self.transforms_list: list[Callable] = []
        if use_ccrop:
            self.transforms_list.append(v2.Resize(self.size + 32))
            self.transforms_list.append(v2.CenterCrop(self.size))
        else:
            self.transforms_list.append(v2.Resize((self.size, self.size)))
        self.transforms_list.append(
            v2.Normalize(
                mean=list(mean),
                std=list(std)
            )
        )
        self.transform = v2.Compose(self.transforms_list)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img)

class Audio_TrainTransform:
    def __init__(
        self,
        sample_rate: int = 16000,
        max_sec: float = 9.8133,
        normalize: bool = True,
    ) -> None:
        self.max_samp = int(sample_rate * max_sec)
        self.normalize = normalize

    def __call__(self, wav: torch.Tensor) -> Tuple[torch.Tensor, int]:
        n = wav.shape[-1]
        if n > self.max_samp:
            start = torch.randint(0, n - self.max_samp + 1, (1,)).item()
            wav = wav[start : start + self.max_samp]

        length = wav.shape[-1]

        if self.normalize:
            wav = (wav - wav.mean()) / torch.sqrt(wav.var() + 1e-7)

        wav = F.pad(wav, (0, self.max_samp - length))
        return wav, length
    
class Audio_ValTransform:
    def __init__(
        self,
        sample_rate: int = 16000,
        max_sec: float = 9.6427,
        normalize: bool = True,
    ) -> None:
        self.max_samp = int(sample_rate * max_sec)
        self.normalize = normalize

    def __call__(self, wav: torch.Tensor) -> Tuple[torch.Tensor, int]:
        wav = wav[: self.max_samp]
        length = wav.shape[-1]

        if self.normalize:
            wav = (wav - wav.mean()) / torch.sqrt(wav.var() + 1e-7)

        wav = F.pad(wav, (0, self.max_samp - length))
        return wav, length