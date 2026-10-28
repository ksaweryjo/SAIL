import math
import time

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from models.contrastive_model import ContrastiveModel
from training.checkpointing import save_checkpoint
from training.configs.baseline_config import Config
from training.loss import InfoNCE_Loss


class Trainer:
    def __init__(
        self,
        config: Config,
        criterion: InfoNCE_Loss,
        model: ContrastiveModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.start_epoch = start_epoch
        self.best_val_loss = best_val_loss

        self.device = config.device
        self.device_type = torch.device(config.device).type
        self.amp_enabled = config.use_amp
        self.clip_params = [
            p
            for p in list(model.parameters()) + list(criterion.parameters())
            if p.requires_grad
        ]
        self.global_step = 0

        self.use_wandb = config.use_wandb
        if self.use_wandb:
            wandb.define_metric("global_step")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="global_step")
            wandb.define_metric("train/epoch_loss", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")

    def _log(self, metrics: dict[str, float | int]) -> None:
        if self.use_wandb:
            wandb.log(metrics)

    def _wandb_run(self) -> Run | None:
        return wandb.run if self.use_wandb else None

    def _to_device(self, batch: list) -> dict[str, torch.Tensor]:
        img, audio, length, text = batch

        return {
            "img": img.to(self.device, non_blocking=True),
            "audio": audio.to(self.device, non_blocking=True),
        }

    def _forward(
        self, img: torch.Tensor, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.amp_enabled,
        ):
            img_emb, audio_emb = self.model(img, audio)

        audio_emb = audio_emb.float()
        img_emb = img_emb.float()
        loss, _ = self.criterion(audio_emb, img_emb)
        return loss, audio_emb, img_emb

    def _temperature(self) -> float:
        scale = min(self.criterion.logit_scale.item(), self.criterion.max_logit_scale)
        return 1.0 / math.exp(scale)

    @staticmethod
    def _recall(
        audio_emb: torch.Tensor,
        img_emb: torch.Tensor,
        ks: tuple[int, ...] = (1, 5, 10),
    ) -> dict[str, float]:
        sim = audio_emb @ img_emb.T
        n = sim.size(0)
        max_k = min(max(ks), n)
        target = torch.arange(n, device=sim.device).unsqueeze(1)

        metrics: dict[str, float] = {}
        for name, s in (("a2i", sim), ("i2a", sim.T)):
            top = s.topk(max_k, dim=1).indices
            for k in ks:
                hits = (top[:, : min(k, max_k)] == target).any(dim=1)
                metrics[f"val/recall_{name}@{k}"] = hits.float().mean().item()
        return metrics

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        self.criterion.train()
        total_loss = torch.zeros((), device=self.device)
        n = 0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False):
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            loss, _, _ = self._forward(batch["img"], batch["audio"])
            loss.backward()

            max_norm = self.config.grad_clip_norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.clip_params, max_norm if max_norm else float("inf")
            )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            total_loss += loss.detach()
            n += 1

            if self.global_step % self.config.log_every_n_steps == 0:
                loss_val = loss.item()
                if math.isnan(loss_val):
                    raise RuntimeError(f"Loss is NaN at global step {self.global_step}")
                self._log(
                    {
                        "global_step": self.global_step,
                        "train/loss": loss_val,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm.item(),
                        "train/temperature": self._temperature(),
                        "train/steps_per_sec": n / (time.time() - t0),
                    }
                )

        return (total_loss / max(n, 1)).item()

    @torch.no_grad()
    def validate_one_epoch(
        self, val_loader: DataLoader, epoch: int
    ) -> dict[str, float]:
        self.model.eval()
        self.criterion.eval()
        total_loss = torch.zeros((), device=self.device)
        n = 0
        aud_embs: list[torch.Tensor] = []
        img_embs: list[torch.Tensor] = []

        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False):
            batch = self._to_device(batch)
            loss, audio_emb, img_emb = self._forward(batch["img"], batch["audio"])
            total_loss += loss
            n += 1
            aud_embs.append(F.normalize(audio_emb, dim=-1).cpu())
            img_embs.append(F.normalize(img_emb, dim=-1).cpu())

        audio_all = torch.cat(aud_embs, dim=0)
        img_all = torch.cat(img_embs, dim=0)

        metrics = self._recall(audio_all, img_all)
        metrics["val/loss"] = (total_loss / max(n, 1)).item()
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.global_step = (self.start_epoch - 1) * len(train_loader)

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.validate_one_epoch(val_loader, epoch)
            val_loss = val_metrics["val/loss"]

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"R@1 a2i {val_metrics['val/recall_a2i@1']:.3f} "
                f"i2a {val_metrics['val/recall_i2a@1']:.3f}"
            )

            self._log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    **val_metrics,
                }
            )

            is_best = val_loss < self.best_val_loss
            run = self._wandb_run()
            if is_best:
                self.best_val_loss = val_loss
                if run is not None:
                    run.summary["best_val_loss"] = val_loss
                    run.summary["best_epoch"] = epoch

            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": (
                        self.scheduler.state_dict()
                        if self.scheduler is not None
                        else None
                    ),
                    "val_loss": val_loss,
                    "best_val_loss": self.best_val_loss,
                    "wandb_run_id": run.id if run is not None else None,
                },
                checkpoint_dir=self.config.checkpoint_dir,
                config_name=self.config.name,
                is_best=is_best,
            )
