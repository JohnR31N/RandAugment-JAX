#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/cifar10_randaugment.yaml}"

python -m randaug_jax.train \
  --config "${CONFIG}" \
  --init-distributed
