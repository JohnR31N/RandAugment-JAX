from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from .config import ExperimentConfig


class ResidualBlock(nn.Module):
    features: int
    stride: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        residual = x
        y = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
        )(x)
        y = _batch_norm(y, train=train, dtype=self.dtype)
        y = nn.relu(y)
        y = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
        )(y)
        y = _batch_norm(y, train=train, dtype=self.dtype)

        if residual.shape != y.shape:
            residual = nn.Conv(
                self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                dtype=self.dtype,
            )(residual)
            residual = _batch_norm(residual, train=train, dtype=self.dtype)

        return nn.relu(y + residual)


class CifarResNet(nn.Module):
    stage_sizes: Sequence[int]
    width: int
    num_classes: int
    dropout_rate: float
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = nn.Conv(
            self.width,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
        )(x)
        x = _batch_norm(x, train=train, dtype=self.dtype)
        x = nn.relu(x)

        for stage_index, blocks in enumerate(self.stage_sizes):
            features = self.width * (2**stage_index)
            for block_index in range(blocks):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                x = ResidualBlock(features=features, stride=stride, dtype=self.dtype)(x, train=train)

        x = jnp.mean(x, axis=(1, 2))
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.num_classes, dtype=jnp.float32)(x)
        return x.astype(jnp.float32)


def make_model(config: ExperimentConfig) -> CifarResNet:
    if config.model.name != "cifar_resnet":
        raise ValueError("Only model.name='cifar_resnet' is implemented in this scaffold")

    if (config.model.depth - 2) % 6 != 0:
        raise ValueError("For CIFAR ResNet, model.depth must satisfy (depth - 2) % 6 == 0")
    blocks_per_stage = (config.model.depth - 2) // 6
    dtype = jnp.bfloat16 if config.train.mixed_precision else jnp.float32
    return CifarResNet(
        stage_sizes=(blocks_per_stage, blocks_per_stage, blocks_per_stage),
        width=config.model.width,
        num_classes=config.dataset.num_classes,
        dropout_rate=config.model.dropout_rate,
        dtype=dtype,
    )


def _batch_norm(x: jnp.ndarray, *, train: bool, dtype: jnp.dtype) -> jnp.ndarray:
    return nn.BatchNorm(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1.0e-5,
        dtype=dtype,
        axis_name="batch" if train else None,
    )(x)
