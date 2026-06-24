from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    data_dir: str = "./data"
    image_size: int = 32
    num_classes: int = 10
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
    num_workers: int = 4
    pin_memory: bool = True
    download: bool = True
    fake_train_size: int = 4096
    fake_eval_size: int = 1024


@dataclass
class AugmentConfig:
    policy: str = "baseline"
    random_crop_padding: int = 4
    horizontal_flip: bool = True
    randaug_num_ops: int = 2
    randaug_magnitude: int = 9
    randaug_num_magnitude_bins: int = 31
    randaug_fill: str = "mean"


@dataclass
class ModelConfig:
    name: str = "cifar_resnet"
    depth: int = 20
    width: int = 16
    dropout_rate: float = 0.0


@dataclass
class TrainConfig:
    seed: int = 0
    global_batch_size: int = 1024
    epochs: int = 200
    learning_rate: float = 0.4
    end_learning_rate: float = 0.0
    warmup_epochs: int = 5
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5.0e-4
    label_smoothing: float = 0.0
    mixed_precision: bool = True
    log_every_steps: int = 50
    eval_every_epochs: int = 1
    checkpoint_every_epochs: int = 10
    keep_checkpoints: int = 3
    ckpt_dir: str = "./checkpoints/cifar10_randaugment"
    resume: bool = False


@dataclass
class RuntimeConfig:
    backend: str = "jax"
    torch_xla_spawn_processes: int | None = None


@dataclass
class ExperimentConfig:
    experiment_name: str = "cifar10_randaugment"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    cfg = ExperimentConfig()
    _merge_dataclass(cfg, data)
    for override in overrides or []:
        _apply_override(cfg, override)
    _finalize(cfg)
    return cfg


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def _merge_dataclass(instance: Any, values: Mapping[str, Any]) -> None:
    allowed = {field.name for field in fields(instance)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys for {type(instance).__name__}: {unknown}")

    for field in fields(instance):
        if field.name not in values:
            continue
        current = getattr(instance, field.name)
        value = values[field.name]
        if is_dataclass(current) and isinstance(value, Mapping):
            _merge_dataclass(current, value)
        else:
            setattr(instance, field.name, _coerce_value(current, value))


def _apply_override(cfg: ExperimentConfig, override: str) -> None:
    key, sep, raw_value = override.partition("=")
    if not sep or not key:
        raise ValueError(f"Override must look like section.field=value, got: {override}")

    target = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown override path: {key}")
        target = getattr(target, part)

    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ValueError(f"Unknown override path: {key}")

    parsed_value = yaml.safe_load(raw_value)
    current = getattr(target, leaf)
    setattr(target, leaf, _coerce_value(current, parsed_value))


def _coerce_value(current: Any, value: Any) -> Any:
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _finalize(cfg: ExperimentConfig) -> None:
    cfg.runtime.backend = cfg.runtime.backend.lower()
    cfg.dataset.name = cfg.dataset.name.lower()
    cfg.augment.policy = cfg.augment.policy.lower()
    cfg.model.name = cfg.model.name.lower()

    if cfg.dataset.name == "cifar10":
        cfg.dataset.num_classes = 10
    elif cfg.dataset.name == "cifar100":
        cfg.dataset.num_classes = 100

    if cfg.train.global_batch_size <= 0:
        raise ValueError("train.global_batch_size must be positive")
    if cfg.augment.policy not in {"none", "baseline", "randaugment"}:
        raise ValueError("augment.policy must be one of: none, baseline, randaugment")
    if cfg.runtime.backend not in {"jax", "torch_xla"}:
        raise ValueError("runtime.backend must be one of: jax, torch_xla")
