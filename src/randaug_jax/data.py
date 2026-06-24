from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .augment import build_torchvision_transform
from .config import ExperimentConfig


@dataclass
class DataLoaders:
    train: Any
    eval: Any
    train_sampler: Optional[Any]
    local_batch_size: int
    steps_per_epoch: int


def make_data_loaders(
    config: ExperimentConfig,
    *,
    process_count: int = 1,
    process_index: int = 0,
) -> DataLoaders:
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    if config.train.global_batch_size % process_count != 0:
        raise ValueError(
            "train.global_batch_size must be divisible by process_count "
            f"({config.train.global_batch_size} vs {process_count})"
        )

    local_batch_size = config.train.global_batch_size // process_count
    train_dataset = _make_dataset(config, train=True)
    eval_dataset = _make_dataset(config, train=False)

    train_sampler = None
    eval_sampler = None
    train_shuffle = True
    if process_count > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=process_count,
            rank=process_index,
            shuffle=True,
            drop_last=True,
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=process_count,
            rank=process_index,
            shuffle=False,
            drop_last=False,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        drop_last=True,
        persistent_workers=config.dataset.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        drop_last=False,
        persistent_workers=config.dataset.num_workers > 0,
    )
    return DataLoaders(
        train=train_loader,
        eval=eval_loader,
        train_sampler=train_sampler,
        local_batch_size=local_batch_size,
        steps_per_epoch=len(train_loader),
    )


def torch_batch_to_numpy(batch: tuple[Any, Any]) -> dict[str, np.ndarray]:
    images, labels = batch
    images = images.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()
    labels = labels.detach().cpu().numpy()
    return {
        "image": images.astype(np.float32, copy=False),
        "label": labels.astype(np.int32, copy=False),
    }


def pad_batch(batch: dict[str, np.ndarray], target_size: int) -> dict[str, np.ndarray]:
    size = batch["label"].shape[0]
    if size > target_size:
        raise ValueError(f"Batch has {size} examples but target_size={target_size}")

    mask = np.zeros((target_size,), dtype=np.float32)
    mask[:size] = 1.0
    if size == target_size:
        batch["mask"] = mask
        return batch

    padded = {}
    for key, value in batch.items():
        pad_shape = (target_size - size, *value.shape[1:])
        padding = np.zeros(pad_shape, dtype=value.dtype)
        padded[key] = np.concatenate([value, padding], axis=0)
    padded["mask"] = mask
    return padded


def shard_batch(batch: dict[str, np.ndarray], local_device_count: int) -> dict[str, np.ndarray]:
    def _shard(value: np.ndarray) -> np.ndarray:
        if value.shape[0] % local_device_count != 0:
            raise ValueError(
                f"Batch dimension {value.shape[0]} is not divisible by local devices "
                f"({local_device_count})"
            )
        return value.reshape((local_device_count, value.shape[0] // local_device_count, *value.shape[1:]))

    return {key: _shard(value) for key, value in batch.items()}


def _make_dataset(config: ExperimentConfig, *, train: bool):
    from torchvision import datasets

    transform = build_torchvision_transform(config.dataset, config.augment, train=train)
    root = config.dataset.data_dir
    name = config.dataset.name

    if name == "cifar10":
        return datasets.CIFAR10(
            root=root,
            train=train,
            download=config.dataset.download,
            transform=transform,
        )
    if name == "cifar100":
        return datasets.CIFAR100(
            root=root,
            train=train,
            download=config.dataset.download,
            transform=transform,
        )
    if name == "fake":
        size = config.dataset.fake_train_size if train else config.dataset.fake_eval_size
        return datasets.FakeData(
            size=size,
            image_size=(3, config.dataset.image_size, config.dataset.image_size),
            num_classes=config.dataset.num_classes,
            transform=transform,
        )

    raise ValueError("dataset.name must be one of: cifar10, cifar100, fake")
