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
    validation_split: float = 0.0
    fake_train_size: int = 4096
    fake_eval_size: int = 1024


@dataclass
class AugmentConfig:
    policy: str = "baseline"
    basic_aug: bool = True
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
    max_train_steps: int = -1
    max_eval_steps: int = -1
    learning_rate: float = 0.4
    end_learning_rate: float = 0.0
    warmup_epochs: int = 5
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5.0e-4
    lr_schedule: str = "cosine"
    lr_decay_epochs: tuple[int, ...] = (100, 150)
    lr_decay_rate: float = 0.1
    label_smoothing: float = 0.0
    mixed_precision: bool = True
    log_every_steps: int = 50
    eval_every_epochs: int = 1
    eval_on_test_each_epoch: bool = True
    final_test: bool = False
    checkpoint_every_epochs: int = 10
    keep_checkpoints: int = 3
    ckpt_dir: str = "./checkpoints/cifar10_randaugment"
    checkpoint_dir: str = "./checkpoints"
    resume: bool = False
    resume_checkpoint: str = ""
    save_csv: bool = False
    output_dir: str = "./outputs"
    output_name: str = ""
    run_name: str = ""
    save_checkpoint: bool = True
    save_best_only: bool = False


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
    data = _normalize_config_data(data)

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

    key = _override_alias(key)
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


def _override_alias(key: str) -> str:
    aliases = {
        "dataset": "dataset.name",
        "data_dir": "dataset.data_dir",
        "model": "model.name",
        "method": "augment.policy",
        "basic_aug": "augment.basic_aug",
        "batch_size": "train.global_batch_size",
        "epochs": "train.epochs",
        "max_train_steps": "train.max_train_steps",
        "max_eval_steps": "train.max_eval_steps",
        "validation_split": "dataset.validation_split",
        "num_workers": "dataset.num_workers",
        "pin_memory": "dataset.pin_memory",
        "eval_on_test_each_epoch": "train.eval_on_test_each_epoch",
        "final_test": "train.final_test",
        "learning_rate": "train.learning_rate",
        "min_learning_rate": "train.end_learning_rate",
        "momentum": "train.momentum",
        "weight_decay": "train.weight_decay",
        "lr_schedule": "train.lr_schedule",
        "lr_decay_epochs": "train.lr_decay_epochs",
        "lr_decay_rate": "train.lr_decay_rate",
        "seed": "train.seed",
        "save_csv": "train.save_csv",
        "output_dir": "train.output_dir",
        "output_name": "train.output_name",
        "run_name": "train.run_name",
        "save_checkpoint": "train.save_checkpoint",
        "checkpoint_dir": "train.checkpoint_dir",
        "save_best_only": "train.save_best_only",
        "resume_checkpoint": "train.resume_checkpoint",
        "backend": "runtime.backend",
    }
    return aliases.get(key, key)


def _coerce_value(current: Any, value: Any) -> Any:
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _normalize_config_data(data: Mapping[str, Any]) -> dict[str, Any]:
    if not _looks_like_flat_config(data):
        return dict(data)

    run_name = str(data.get("run_name") or data.get("output_name") or "experiment")
    checkpoint_dir = str(data.get("checkpoint_dir", "./checkpoints"))
    method = str(data.get("method", "baseline"))

    nested: dict[str, Any] = {
        "experiment_name": run_name,
        "runtime": {
            "backend": str(data.get("backend", "torch_xla")),
        },
        "dataset": {
            "name": data.get("dataset", "cifar10"),
            "data_dir": data.get("data_dir", "./data"),
            "validation_split": data.get("validation_split", 0.0),
            "num_workers": data.get("num_workers", 0),
            "pin_memory": data.get("pin_memory", False),
        },
        "augment": {
            "policy": method,
            "basic_aug": data.get("basic_aug", True),
        },
        "model": {
            "name": data.get("model", "preact_resnet18"),
        },
        "train": {
            "seed": data.get("seed", 0),
            "global_batch_size": data.get("batch_size", 128),
            "epochs": data.get("epochs", 200),
            "max_train_steps": data.get("max_train_steps", -1),
            "max_eval_steps": data.get("max_eval_steps", -1),
            "learning_rate": data.get("learning_rate", 0.1),
            "end_learning_rate": data.get("min_learning_rate", 0.0),
            "warmup_epochs": data.get("warmup_epochs", 0),
            "momentum": data.get("momentum", 0.9),
            "weight_decay": data.get("weight_decay", 5.0e-4),
            "lr_schedule": data.get("lr_schedule", "cosine"),
            "lr_decay_epochs": data.get("lr_decay_epochs", [100, 150]),
            "lr_decay_rate": data.get("lr_decay_rate", 0.1),
            "eval_on_test_each_epoch": data.get("eval_on_test_each_epoch", True),
            "final_test": data.get("final_test", False),
            "save_csv": data.get("save_csv", False),
            "output_dir": data.get("output_dir", "./outputs"),
            "output_name": data.get("output_name", ""),
            "run_name": run_name,
            "save_checkpoint": data.get("save_checkpoint", True),
            "checkpoint_every_epochs": data.get("checkpoint_every_epochs", 1),
            "ckpt_dir": "",
            "checkpoint_dir": checkpoint_dir,
            "save_best_only": data.get("save_best_only", False),
            "resume": bool(data.get("resume_checkpoint", "")),
            "resume_checkpoint": data.get("resume_checkpoint", ""),
        },
    }

    if "randaug_num_ops" in data:
        nested["augment"]["randaug_num_ops"] = data["randaug_num_ops"]
    if "randaug_magnitude" in data:
        nested["augment"]["randaug_magnitude"] = data["randaug_magnitude"]
    if "randaug_num_magnitude_bins" in data:
        nested["augment"]["randaug_num_magnitude_bins"] = data["randaug_num_magnitude_bins"]
    return nested


def _looks_like_flat_config(data: Mapping[str, Any]) -> bool:
    return (
        isinstance(data.get("dataset"), str)
        or isinstance(data.get("model"), str)
        or "method" in data
        or "batch_size" in data
    )


def _finalize(cfg: ExperimentConfig) -> None:
    cfg.runtime.backend = cfg.runtime.backend.lower()
    cfg.dataset.name = cfg.dataset.name.lower()
    cfg.augment.policy = cfg.augment.policy.lower()
    cfg.model.name = cfg.model.name.lower()
    cfg.train.lr_schedule = cfg.train.lr_schedule.lower()
    if isinstance(cfg.train.lr_decay_epochs, list):
        cfg.train.lr_decay_epochs = tuple(cfg.train.lr_decay_epochs)

    if cfg.dataset.name == "cifar10":
        cfg.dataset.num_classes = 10
    elif cfg.dataset.name == "cifar100":
        cfg.dataset.num_classes = 100

    if cfg.train.global_batch_size <= 0:
        raise ValueError("train.global_batch_size must be positive")
    if not 0.0 <= cfg.dataset.validation_split < 1.0:
        raise ValueError("dataset.validation_split must be in [0, 1)")
    if cfg.augment.policy not in {"none", "baseline", "randaugment"}:
        raise ValueError("augment.policy must be one of: none, baseline, randaugment")
    if cfg.runtime.backend not in {"jax", "torch_xla"}:
        raise ValueError("runtime.backend must be one of: jax, torch_xla")
    if cfg.train.lr_schedule not in {"cosine", "step", "constant"}:
        raise ValueError("train.lr_schedule must be one of: cosine, step, constant")
    if not cfg.train.run_name:
        cfg.train.run_name = cfg.experiment_name
