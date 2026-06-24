from __future__ import annotations

import torch
from torch import nn

from .config import ExperimentConfig


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class TorchCifarResNet(nn.Module):
    def __init__(
        self,
        *,
        blocks_per_stage: int,
        width: int,
        num_classes: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.in_channels = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(width, blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(width * 2, blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(width * 4, blocks_per_stage, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(width * 4, num_classes)

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def make_torch_model(config: ExperimentConfig) -> nn.Module:
    if config.model.name != "cifar_resnet":
        raise ValueError("Only model.name='cifar_resnet' is implemented for torch_xla")
    if (config.model.depth - 2) % 6 != 0:
        raise ValueError("For CIFAR ResNet, model.depth must satisfy (depth - 2) % 6 == 0")

    return TorchCifarResNet(
        blocks_per_stage=(config.model.depth - 2) // 6,
        width=config.model.width,
        num_classes=config.dataset.num_classes,
        dropout_rate=config.model.dropout_rate,
    )
