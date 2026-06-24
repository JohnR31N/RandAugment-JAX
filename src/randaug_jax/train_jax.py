from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.training import checkpoints, common_utils, train_state

from .config import ExperimentConfig, config_to_dict, load_config
from .data import make_data_loaders, pad_batch, shard_batch, torch_batch_to_numpy
from .models import make_model


class TrainState(train_state.TrainState):
    batch_stats: Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value, for example --override augment.randaug_magnitude=15",
    )
    parser.add_argument(
        "--init-distributed",
        action="store_true",
        help="Call jax.distributed.initialize(); use this on multi-host TPU jobs.",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    run(config, init_distributed=args.init_distributed)


def run(config: ExperimentConfig, *, init_distributed: bool = False) -> None:
    if init_distributed:
        jax.distributed.initialize()

    if _is_host0():
        print(json.dumps(config_to_dict(config), indent=2, sort_keys=True))
        print(_device_summary())

    loaders = make_data_loaders(
        config,
        process_count=jax.process_count(),
        process_index=jax.process_index(),
    )
    local_device_count = jax.local_device_count()
    if loaders.local_batch_size % local_device_count != 0:
        raise ValueError(
            "Per-process batch size must be divisible by local device count: "
            f"{loaders.local_batch_size} vs {local_device_count}"
        )

    rng = jax.random.PRNGKey(config.train.seed + jax.process_index())
    model = make_model(config)
    learning_rate_fn = _make_learning_rate_fn(config, loaders.steps_per_epoch)
    state = _create_train_state(config, model, rng, learning_rate_fn)

    if config.train.resume:
        state = checkpoints.restore_checkpoint(config.train.ckpt_dir, state)
    state = jax_utils.replicate(state)
    rngs = jax.random.split(rng, local_device_count)

    train_step = _make_train_step(config)
    eval_step = _make_eval_step(config)

    for epoch in range(1, config.train.epochs + 1):
        if loaders.train_sampler is not None:
            loaders.train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        last_metrics = None
        for step, torch_batch in enumerate(loaders.train, start=1):
            batch = torch_batch_to_numpy(torch_batch)
            batch = shard_batch(batch, local_device_count)
            state, metrics, rngs = train_step(state, batch, rngs)
            last_metrics = metrics

            if _is_host0() and step % config.train.log_every_steps == 0:
                host_metrics = _unreplicate_metrics(metrics)
                lr = float(learning_rate_fn((epoch - 1) * loaders.steps_per_epoch + step))
                print(
                    f"epoch={epoch:03d} step={step:05d}/{loaders.steps_per_epoch:05d} "
                    f"loss={host_metrics['loss']:.4f} "
                    f"acc={host_metrics['accuracy']:.4f} lr={lr:.6f}"
                )

        if _is_host0() and last_metrics is not None:
            host_metrics = _unreplicate_metrics(last_metrics)
            elapsed = time.time() - epoch_start
            print(
                f"epoch={epoch:03d} train_loss={host_metrics['loss']:.4f} "
                f"train_acc={host_metrics['accuracy']:.4f} elapsed={elapsed:.1f}s"
            )

        if epoch % config.train.eval_every_epochs == 0:
            eval_metrics = _evaluate(eval_step, state, loaders.eval, loaders.local_batch_size, local_device_count)
            if _is_host0():
                print(
                    f"epoch={epoch:03d} eval_loss={eval_metrics['loss']:.4f} "
                    f"eval_acc={eval_metrics['accuracy']:.4f}"
                )

        should_checkpoint = (
            config.train.checkpoint_every_epochs > 0
            and epoch % config.train.checkpoint_every_epochs == 0
        )
        if _is_host0() and should_checkpoint:
            Path(config.train.ckpt_dir).mkdir(parents=True, exist_ok=True)
            checkpoints.save_checkpoint(
                ckpt_dir=config.train.ckpt_dir,
                target=jax_utils.unreplicate(state),
                step=epoch,
                keep=config.train.keep_checkpoints,
                overwrite=True,
            )


def _create_train_state(
    config: ExperimentConfig,
    model: Any,
    rng: jax.Array,
    learning_rate_fn: optax.Schedule,
) -> TrainState:
    init_rng, dropout_rng = jax.random.split(rng)
    image_shape = (1, config.dataset.image_size, config.dataset.image_size, 3)
    variables = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        jnp.ones(image_shape, dtype=jnp.float32),
        train=False,
    )
    tx = _make_optimizer(config, learning_rate_fn)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables.get("batch_stats", {}),
        tx=tx,
    )


def _make_learning_rate_fn(config: ExperimentConfig, steps_per_epoch: int) -> optax.Schedule:
    total_steps = max(config.train.epochs * steps_per_epoch, 1)
    warmup_steps = max(config.train.warmup_epochs * steps_per_epoch, 0)
    if warmup_steps == 0:
        return optax.cosine_decay_schedule(
            init_value=config.train.learning_rate,
            decay_steps=total_steps,
            alpha=config.train.end_learning_rate / config.train.learning_rate,
        )

    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=config.train.learning_rate,
        transition_steps=warmup_steps,
    )
    cosine = optax.cosine_decay_schedule(
        init_value=config.train.learning_rate,
        decay_steps=max(total_steps - warmup_steps, 1),
        alpha=config.train.end_learning_rate / config.train.learning_rate,
    )
    return optax.join_schedules([warmup, cosine], [warmup_steps])


def _make_optimizer(config: ExperimentConfig, learning_rate_fn: optax.Schedule) -> optax.GradientTransformation:
    mask = _decay_mask
    return optax.chain(
        optax.masked(optax.add_decayed_weights(config.train.weight_decay), mask),
        optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.train.momentum,
            nesterov=config.train.nesterov,
        ),
    )


def _decay_mask(params: Any) -> Any:
    flat = traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] == "kernel" for path in flat}
    return traverse_util.unflatten_dict(flat_mask)


def _make_train_step(config: ExperimentConfig):
    num_classes = config.dataset.num_classes
    label_smoothing = config.train.label_smoothing

    @functools.partial(jax.pmap, axis_name="batch")
    def train_step(state: TrainState, batch: dict[str, jax.Array], rng: jax.Array):
        rng, dropout_rng = jax.random.split(rng)

        def loss_fn(params):
            variables = {"params": params, "batch_stats": state.batch_stats}
            (logits, mutable) = state.apply_fn(
                variables,
                batch["image"],
                train=True,
                mutable=["batch_stats"],
                rngs={"dropout": dropout_rng},
            )
            labels = common_utils.onehot(batch["label"], num_classes)
            if label_smoothing > 0:
                labels = optax.smooth_labels(labels, label_smoothing)
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            return loss, (logits, mutable["batch_stats"])

        (loss, (logits, batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads, batch_stats=batch_stats)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch["label"])
        metrics = jax.lax.pmean({"loss": loss, "accuracy": accuracy}, axis_name="batch")
        return new_state, metrics, rng

    return train_step


def _make_eval_step(config: ExperimentConfig):
    num_classes = config.dataset.num_classes

    @functools.partial(jax.pmap, axis_name="batch")
    def eval_step(state: TrainState, batch: dict[str, jax.Array]):
        variables = {"params": state.params, "batch_stats": state.batch_stats}
        logits = state.apply_fn(variables, batch["image"], train=False, mutable=False)
        labels = common_utils.onehot(batch["label"], num_classes)
        losses = optax.softmax_cross_entropy(logits, labels)
        mask = batch["mask"].astype(jnp.float32)
        correct = (jnp.argmax(logits, axis=-1) == batch["label"]).astype(jnp.float32)
        metrics = {
            "loss_sum": jnp.sum(losses * mask),
            "correct": jnp.sum(correct * mask),
            "count": jnp.sum(mask),
        }
        return jax.lax.psum(metrics, axis_name="batch")

    return eval_step


def _evaluate(eval_step, state: TrainState, eval_loader: Any, local_batch_size: int, local_device_count: int) -> dict[str, float]:
    totals = {"loss_sum": 0.0, "correct": 0.0, "count": 0.0}
    for torch_batch in eval_loader:
        batch = torch_batch_to_numpy(torch_batch)
        batch = pad_batch(batch, local_batch_size)
        batch = shard_batch(batch, local_device_count)
        metrics = eval_step(state, batch)
        host_metrics = {key: float(value[0]) for key, value in metrics.items()}
        for key in totals:
            totals[key] += host_metrics[key]

    count = max(totals["count"], 1.0)
    return {
        "loss": totals["loss_sum"] / count,
        "accuracy": totals["correct"] / count,
    }


def _unreplicate_metrics(metrics: dict[str, jax.Array]) -> dict[str, float]:
    return {key: float(value[0]) for key, value in metrics.items()}


def _device_summary() -> str:
    return (
        f"process_index={jax.process_index()} process_count={jax.process_count()} "
        f"local_devices={jax.local_device_count()} global_devices={jax.device_count()} "
        f"backend={jax.default_backend()}"
    )


def _is_host0() -> bool:
    return jax.process_index() == 0


if __name__ == "__main__":
    main()
