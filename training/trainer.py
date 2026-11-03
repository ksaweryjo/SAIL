import math
import time

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from models.contrastive_model import ContrastiveModel
from training.checkpointing import save_checkpoint
from training.configs.baseline_config import Config
from training.loss import InfoNCE_Loss
from training.metrics import embedding_diagnostics, retrieval_metrics


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
            for name in (
                "loss",
                "a2i_loss",
                "i2a_loss",
                "lr",
                "grad_norm",
                "temperature",
            ):
                wandb.define_metric(f"train/{name}", step_metric="global_step")
            for name in ("epoch_loss", "epoch_a2i_loss", "epoch_i2a_loss"):
                wandb.define_metric(f"train/{name}", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("performance/batch_time", step_metric="global_step")
            wandb.define_metric(
                "performance/step_samples_per_sec", step_metric="global_step"
            )
            wandb.define_metric("performance/epoch_*", step_metric="epoch")

    def _log(self, metrics: dict[str, float | int]) -> None:
        if self.use_wandb:
            wandb.log(metrics)

    def _wandb_run(self) -> Run | None:
        return wandb.run if self.use_wandb else None

    def _to_device(self, batch: list) -> dict[str, torch.Tensor]:
        img, audio, length, text, image_ids = batch
        lengths = length.to(self.device, dtype=torch.long)
        attention_mask = (
            torch.arange(audio.size(1), device=self.device)[None, :]
            < lengths[:, None]
        )

        return {
            "img": img.to(self.device, non_blocking=True),
            "audio": audio.to(self.device, non_blocking=True),
            "audio_attention_mask": attention_mask,
            "image_ids": image_ids.to(self.device, dtype=torch.long),
        }

    def _forward(
        self,
        audio: torch.Tensor,
        img: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        image_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.amp_enabled,
        ):
            audio_emb, img_emb = self.model(audio, img, audio_attention_mask)

        audio_emb = audio_emb.float()
        img_emb = img_emb.float()
        loss, loss_a2i, loss_i2a, _ = self.criterion.compute_components(
            audio_emb, img_emb, image_ids
        )
        return loss, loss_a2i, loss_i2a, audio_emb, img_emb

    def _temperature(self) -> float:
        scale = min(self.criterion.logit_scale.item(), self.criterion.max_logit_scale)
        return 1.0 / math.exp(scale)

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        self.criterion.train()
        total_loss = torch.zeros((), device=self.device)
        total_a2i_loss = torch.zeros((), device=self.device)
        total_i2a_loss = torch.zeros((), device=self.device)
        n = 0
        samples = 0
        t0 = time.time()
        last_batch_time = t0
        metrics: dict[str, float] = {}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False):
            batch_start = time.time()
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            loss, loss_a2i, loss_i2a, _, _ = self._forward(
                batch["audio"],
                batch["img"],
                batch["audio_attention_mask"],
                batch["image_ids"],
            )
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
            total_a2i_loss += loss_a2i.detach()
            total_i2a_loss += loss_i2a.detach()
            n += 1
            samples += batch["audio"].size(0)
            last_batch_time = time.time() - batch_start

            if self.global_step % self.config.log_every_n_steps == 0:
                loss_val = loss.item()
                if math.isnan(loss_val):
                    raise RuntimeError(f"Loss is NaN at global step {self.global_step}")
                self._log(
                    {
                        "global_step": self.global_step,
                        "train/loss": loss_val,
                        "train/a2i_loss": loss_a2i.item(),
                        "train/i2a_loss": loss_i2a.item(),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm.item(),
                        "train/temperature": self._temperature(),
                        "performance/batch_time": last_batch_time,
                        "performance/step_samples_per_sec": (
                            batch["audio"].size(0) / last_batch_time
                        ),
                    }
                )

        epoch_time = time.time() - t0
        metrics["train/epoch_loss"] = (total_loss / max(n, 1)).item()
        metrics["train/epoch_a2i_loss"] = (total_a2i_loss / max(n, 1)).item()
        metrics["train/epoch_i2a_loss"] = (total_i2a_loss / max(n, 1)).item()
        metrics["performance/epoch_time"] = epoch_time
        metrics["performance/epoch_steps_per_sec"] = n / max(epoch_time, 1e-12)
        metrics["performance/epoch_samples_per_sec"] = (
            samples / max(epoch_time, 1e-12)
        )
        return metrics

    @torch.no_grad()
    def validate_one_epoch(
        self, val_loader: DataLoader, epoch: int
    ) -> dict[str, float]:
        self.model.eval()
        self.criterion.eval()
        total_loss = torch.zeros((), device=self.device)
        total_a2i_loss = torch.zeros((), device=self.device)
        total_i2a_loss = torch.zeros((), device=self.device)
        n = 0
        aud_embs: list[torch.Tensor] = []
        img_embs: list[torch.Tensor] = []
        image_id_batches: list[torch.Tensor] = []

        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False):
            batch = self._to_device(batch)
            loss, loss_a2i, loss_i2a, audio_emb, img_emb = self._forward(
                batch["audio"],
                batch["img"],
                batch["audio_attention_mask"],
                batch["image_ids"],
            )
            total_loss += loss
            total_a2i_loss += loss_a2i
            total_i2a_loss += loss_i2a
            n += 1
            aud_embs.append(audio_emb.cpu())
            img_embs.append(img_emb.cpu())
            image_id_batches.append(batch["image_ids"].cpu())

        audio_all = torch.cat(aud_embs, dim=0)
        img_all = torch.cat(img_embs, dim=0)
        all_image_ids = torch.cat(image_id_batches, dim=0)

        raw_metrics = retrieval_metrics(audio_all, img_all, all_image_ids)
        raw_metrics.update(embedding_diagnostics(audio_all, img_all))
        metrics = {f"val/{name}": value for name, value in raw_metrics.items()}
        metrics["val/a2i_loss"] = (total_a2i_loss / max(n, 1)).item()
        metrics["val/i2a_loss"] = (total_i2a_loss / max(n, 1)).item()
        metrics["val/loss"] = (total_loss / max(n, 1)).item()
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.global_step = (self.start_epoch - 1) * len(train_loader)

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.validate_one_epoch(val_loader, epoch)
            val_loss = val_metrics["val/loss"]
            train_loss = train_metrics["train/epoch_loss"]

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"R@1 a2i {val_metrics['val/recall_a2i@1']:.3f} "
                f"i2a {val_metrics['val/recall_i2a@1']:.3f}"
            )

            self._log(
                {
                    "epoch": epoch,
                    **train_metrics,
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
