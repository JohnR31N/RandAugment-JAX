# RandAugment-JAX

This repository is a small experiment scaffold for comparing a baseline input
pipeline against RandAugment on GCloud TPU.

The important design choice is that RandAugment is not reimplemented here. Both
backends call `torchvision.transforms.RandAugment` in host-side PyTorch
`DataLoader` workers.

You can choose the training backend from config:

- `runtime.backend: jax`: JAX/Flax TPU training, torchvision augmentation.
- `runtime.backend: torch_xla`: pure PyTorch/XLA TPU training, torchvision
  augmentation.

## What is included

- `configs/cifar10_baseline.yaml`: JAX baseline.
- `configs/cifar10_randaugment.yaml`: JAX + RandAugment.
- `configs/cifar10_torch_xla_baseline.yaml`: PyTorch/XLA baseline.
- `configs/cifar10_torch_xla_randaugment.yaml`: PyTorch/XLA + RandAugment.
- `configs/fake_smoke.yaml`: tiny synthetic-data config for syntax/runtime smoke
  checks.
- `src/randaug_jax/data.py`: PyTorch/torchvision input pipeline.
- `src/randaug_jax/train.py`: backend dispatcher.
- `src/randaug_jax/train_jax.py`: JAX/Flax TPU-oriented training loop.
- `src/randaug_jax/train_torch_xla.py`: PyTorch/XLA TPU-oriented training loop.
- `scripts/train_cifar10_randaugment_tpu.sh`: multi-host TPU launch entrypoint.
- `scripts/train_cifar10_randaugment_torch_xla_tpu.sh`: PyTorch/XLA TPU launch
  entrypoint.

## Install on a TPU VM

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tpu.txt
python -m pip install -e .
```

For PyTorch/XLA TPU runs:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch-xla-tpu.txt
python -m pip install -e .
```

Keep the `torch` and `torch_xla` minor versions matched if you pin versions.

For local CPU smoke checks, install the CPU extra instead:

```bash
python -m pip install -e ".[cpu]"
```

## Run

JAX baseline:

```bash
python -m randaug_jax.train --config configs/cifar10_baseline.yaml
```

JAX RandAugment competitor:

```bash
python -m randaug_jax.train --config configs/cifar10_randaugment.yaml
```

PyTorch/XLA baseline:

```bash
python -m randaug_jax.train --config configs/cifar10_torch_xla_baseline.yaml
```

PyTorch/XLA RandAugment competitor:

```bash
python -m randaug_jax.train --config configs/cifar10_torch_xla_randaugment.yaml
```

JAX multi-host TPU jobs should initialize JAX distributed:

```bash
bash scripts/train_cifar10_randaugment_tpu.sh configs/cifar10_randaugment.yaml
```

PyTorch/XLA TPU jobs use XLA multiprocessing:

```bash
bash scripts/train_cifar10_randaugment_torch_xla_tpu.sh configs/cifar10_torch_xla_randaugment.yaml
```

Override any config field from the command line:

```bash
python -m randaug_jax.train \
  --config configs/cifar10_randaugment.yaml \
  --override runtime.backend=torch_xla \
  --override augment.randaug_magnitude=15 \
  --override train.global_batch_size=2048
```

## Notes

- `augment.policy` supports `none`, `baseline`, and `randaugment`.
- `runtime.backend` supports `jax` and `torch_xla`.
- The torchvision RandAugment knobs are `randaug_num_ops`,
  `randaug_magnitude`, and `randaug_num_magnitude_bins`.
- In `jax` mode, PyTorch is used only for CPU-side data loading and
  augmentation.
- In `torch_xla` mode, the model, optimizer step, and TPU collectives are
  PyTorch/XLA.
- For custom datasets, add a dataset branch in `src/randaug_jax/data.py` while
  reusing `build_torchvision_transform`.
