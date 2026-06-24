from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, config_to_dict
from .data import make_data_loaders
from .torch_models import make_torch_model


def run(config: ExperimentConfig) -> None:
    if config.train.mixed_precision:
        os.environ.setdefault("XLA_USE_BF16", "1")

    import torch_xla.distributed.xla_multiprocessing as xmp

    spawn_kwargs: dict[str, Any] = {"args": (config,)}
    if config.runtime.torch_xla_spawn_processes is not None:
        spawn_kwargs["nprocs"] = config.runtime.torch_xla_spawn_processes
    xmp.spawn(_train_worker, **spawn_kwargs)


def _train_worker(index: int, config: ExperimentConfig) -> None:
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    ordinal = _xla_ordinal()
    torch.manual_seed(config.train.seed + ordinal)
    device = xm.xla_device()
    world_size = _xla_world_size()

    if ordinal == 0:
        print(json.dumps(config_to_dict(config), indent=2, sort_keys=True))
        print(f"torch_xla device={device} ordinal={ordinal} world_size={world_size}")

    loaders = make_data_loaders(
        config,
        process_count=world_size,
        process_index=ordinal,
    )
    model = make_torch_model(config).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.train.learning_rate,
        momentum=config.train.momentum,
        nesterov=config.train.nesterov,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_make_lr_lambda(config, loaders.steps_per_epoch),
    )

    start_epoch = 1
    if config.train.resume:
        start_epoch = _restore_checkpoint(config, model, optimizer, scheduler, device)

    for epoch in range(start_epoch, config.train.epochs + 1):
        if loaders.train_sampler is not None:
            loaders.train_sampler.set_epoch(epoch)

        train_loader = pl.MpDeviceLoader(loaders.train, device)
        epoch_start = time.time()
        train_metrics = _train_one_epoch(
            config,
            model,
            criterion,
            optimizer,
            scheduler,
            train_loader,
            epoch,
            loaders.steps_per_epoch,
        )
        if ordinal == 0:
            elapsed = time.time() - epoch_start
            print(
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} elapsed={elapsed:.1f}s"
            )

        if epoch % config.train.eval_every_epochs == 0:
            eval_loader = pl.MpDeviceLoader(loaders.eval, device)
            eval_metrics = _evaluate(model, criterion, eval_loader)
            if ordinal == 0:
                print(
                    f"epoch={epoch:03d} eval_loss={eval_metrics['loss']:.4f} "
                    f"eval_acc={eval_metrics['accuracy']:.4f}"
                )

        should_checkpoint = (
            config.train.checkpoint_every_epochs > 0
            and epoch % config.train.checkpoint_every_epochs == 0
        )
        if should_checkpoint:
            _save_checkpoint(config, epoch, model, optimizer, scheduler)


def _train_one_epoch(
    config: ExperimentConfig,
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    epoch: int,
    steps_per_epoch: int,
) -> dict[str, float]:
    import torch
    import torch_xla.core.xla_model as xm

    model.train()
    totals = torch.zeros(3, device=xm.xla_device())
    for step, (images, labels) in enumerate(train_loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        scheduler.step()

        with torch.no_grad():
            correct = (logits.argmax(dim=1) == labels).sum().float()
            batch_metrics = torch.stack(
                [
                    loss.detach() * labels.numel(),
                    correct,
                    torch.tensor(float(labels.numel()), device=labels.device),
                ]
            )
            totals += batch_metrics
            if step % config.train.log_every_steps == 0:
                reduced = xm.all_reduce(xm.REDUCE_SUM, totals)
                if _xla_ordinal() == 0:
                    count = max(float(reduced[2].item()), 1.0)
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"epoch={epoch:03d} step={step:05d}/{steps_per_epoch:05d} "
                        f"loss={float(reduced[0].item()) / count:.4f} "
                        f"acc={float(reduced[1].item()) / count:.4f} lr={lr:.6f}"
                    )

    reduced = xm.all_reduce(xm.REDUCE_SUM, totals)
    return _metrics_from_tensor(reduced)


def _evaluate(model, criterion, eval_loader) -> dict[str, float]:
    import torch
    import torch_xla.core.xla_model as xm

    model.eval()
    totals = torch.zeros(3, device=xm.xla_device())
    with torch.no_grad():
        for images, labels in eval_loader:
            logits = model(images)
            loss = criterion(logits, labels)
            correct = (logits.argmax(dim=1) == labels).sum().float()
            totals += torch.stack(
                [
                    loss.detach() * labels.numel(),
                    correct,
                    torch.tensor(float(labels.numel()), device=labels.device),
                ]
            )
    reduced = xm.all_reduce(xm.REDUCE_SUM, totals)
    return _metrics_from_tensor(reduced)


def _make_lr_lambda(config: ExperimentConfig, steps_per_epoch: int):
    total_steps = max(config.train.epochs * steps_per_epoch, 1)
    warmup_steps = max(config.train.warmup_epochs * steps_per_epoch, 0)
    end_scale = config.train.end_learning_rate / config.train.learning_rate

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1.0e-8)
        progress_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / progress_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_scale + (1.0 - end_scale) * cosine

    return lr_lambda


def _metrics_from_tensor(tensor) -> dict[str, float]:
    count = max(float(tensor[2].item()), 1.0)
    return {
        "loss": float(tensor[0].item()) / count,
        "accuracy": float(tensor[1].item()) / count,
    }


def _restore_checkpoint(config: ExperimentConfig, model, optimizer, scheduler, device) -> int:
    import torch
    import torch_xla.core.xla_model as xm

    ckpt_path = _latest_checkpoint(Path(config.train.ckpt_dir))
    if ckpt_path is None:
        if _xla_ordinal() == 0:
            print(f"No checkpoint found in {config.train.ckpt_dir}; starting from epoch 1")
        return 1

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint["epoch"]) + 1
    if _xla_ordinal() == 0:
        print(f"Restored {ckpt_path}; starting from epoch {start_epoch}")
    return start_epoch


def _save_checkpoint(config: ExperimentConfig, epoch: int, model, optimizer, scheduler) -> None:
    import torch_xla.core.xla_model as xm

    ckpt_dir = Path(config.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:05d}.pt"
    xm.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config_to_dict(config),
        },
        str(ckpt_path),
    )
    if _xla_ordinal() == 0:
        _prune_checkpoints(ckpt_dir, config.train.keep_checkpoints)


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    for ckpt_path in checkpoints[:-keep]:
        ckpt_path.unlink(missing_ok=True)


def _xla_world_size() -> int:
    import torch_xla.core.xla_model as xm

    if hasattr(xm, "xrt_world_size"):
        return int(xm.xrt_world_size())
    import torch_xla.runtime as xr

    return int(xr.world_size())


def _xla_ordinal() -> int:
    import torch_xla.core.xla_model as xm

    if hasattr(xm, "get_ordinal"):
        return int(xm.get_ordinal())
    import torch_xla.runtime as xr

    return int(xr.global_ordinal())
