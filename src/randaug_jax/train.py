from __future__ import annotations

import argparse

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value, for example --override runtime.backend=torch_xla",
    )
    parser.add_argument(
        "--init-distributed",
        action="store_true",
        help="Initialize distributed runtime for backends that need it.",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    if config.runtime.backend == "jax":
        from . import train_jax

        train_jax.run(config, init_distributed=args.init_distributed)
        return

    if config.runtime.backend == "torch_xla":
        from . import train_torch_xla

        train_torch_xla.run(config)
        return

    raise ValueError("runtime.backend must be one of: jax, torch_xla")


if __name__ == "__main__":
    main()
