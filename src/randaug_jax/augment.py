from __future__ import annotations

from typing import Optional

from .config import AugmentConfig, DatasetConfig


def build_torchvision_transform(
    dataset: DatasetConfig,
    augment: AugmentConfig,
    *,
    train: bool,
):
    """Build a torchvision transform while keeping RandAugment in PyTorch land."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    ops = []
    if dataset.image_size != 32:
        ops.append(transforms.Resize(dataset.image_size, interpolation=InterpolationMode.BILINEAR))

    if train and augment.policy != "none":
        if augment.random_crop_padding > 0:
            ops.append(
                transforms.RandomCrop(
                    dataset.image_size,
                    padding=augment.random_crop_padding,
                    padding_mode="reflect",
                )
            )
        if augment.horizontal_flip:
            ops.append(transforms.RandomHorizontalFlip())

    if train and augment.policy == "randaugment":
        ops.append(
            transforms.RandAugment(
                num_ops=augment.randaug_num_ops,
                magnitude=augment.randaug_magnitude,
                num_magnitude_bins=augment.randaug_num_magnitude_bins,
                interpolation=InterpolationMode.BILINEAR,
                fill=_randaugment_fill(dataset, augment),
            )
        )

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset.mean, std=dataset.std),
        ]
    )
    return transforms.Compose(ops)


def _randaugment_fill(dataset: DatasetConfig, augment: AugmentConfig) -> Optional[tuple[int, int, int]]:
    if augment.randaug_fill == "none":
        return None
    if augment.randaug_fill != "mean":
        raise ValueError("augment.randaug_fill currently supports 'mean' or 'none'")
    return tuple(int(round(channel * 255.0)) for channel in dataset.mean)
