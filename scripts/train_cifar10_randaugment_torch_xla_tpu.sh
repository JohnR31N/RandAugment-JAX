#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/cifar10_torch_xla_randaugment.yaml}"

python -m randaug_jax.train --config "${CONFIG}"
