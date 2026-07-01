# SPDX-License-Identifier: Apache-2.0
"""YAML-only training entrypoint.

Usage::

    torchrun --nproc_per_node=<N> -m flashgen.train.entrypoint.train \
        --config path/to/run.yaml

Any unknown ``--dotted.key value`` arguments are applied as
overrides to the YAML config before parsing.  For example::

    torchrun --nproc_per_node=8 -m flashgen.train.entrypoint.train \
        --config path/to/run.yaml \
        --training.distributed.num_gpus 8 \
        --training.optimizer.learning_rate 1e-5
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

from flashgen.logger import init_logger

logger = init_logger(__name__)


def run_training_from_config(
    config_path: str,
    *,
    dry_run: bool = False,
    overrides: list[str] | None = None,
) -> None:
    """YAML-only training entrypoint (schema v2)."""

    from flashgen.distributed import (
        maybe_init_distributed_environment_and_model_parallel, )
    from flashgen.train import Trainer
    from flashgen.train.utils.checkpoint import (
        CheckpointConfig,
        CheckpointManager,
    )
    from flashgen.train.utils.builder import build_from_config
    from flashgen.train.utils.config import load_run_config

    # Enable deterministic mode for reproducibility.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg = load_run_config(config_path, overrides=overrides)
    tc = cfg.training

    # FlashGen is the NPU-only FastVideo fork. Keep training on the only
    # supported attention backend instead of inheriting CUDA-only defaults.
    if tc.vsa_sparsity > 0.0:
        raise ValueError("FlashGen NPU training only supports FLASHGEN_ATTENTION_BACKEND=TORCH_SDPA; "
                         "set training.vsa.sparsity to 0.")
    os.environ.setdefault("FLASHGEN_ATTENTION_BACKEND", "TORCH_SDPA")

    maybe_init_distributed_environment_and_model_parallel(
        tc.distributed.tp_size,
        tc.distributed.sp_size,
    )

    _, method, dataloader, start_step = build_from_config(cfg)

    if dry_run:
        logger.info("Dry-run: config parsed and "
                    "build_from_config succeeded.")
        return

    trainer = Trainer(
        tc,
        config=cfg.resolved_config(),
        callback_configs=cfg.callbacks,
    )

    # Attach the exact YAML used for this run to the
    # tracker (e.g., W&B Files).
    trainer.tracker.log_file(
        os.path.abspath(os.path.expanduser(config_path)),
        name="run.yaml",
    )

    ckpt_config = CheckpointConfig(
        save_steps=int(tc.checkpoint.training_state_checkpointing_steps or 0),
        keep_last=int(tc.checkpoint.checkpoints_total_limit or 0),
    )

    checkpoint_manager = CheckpointManager(
        method=method,
        dataloader=dataloader,
        output_dir=tc.checkpoint.output_dir,
        config=ckpt_config,
        callbacks=trainer.callbacks,
        raw_config=cfg.raw,
    )

    trainer.run(
        method,
        dataloader=dataloader,
        max_steps=tc.loop.max_train_steps,
        start_step=start_step,
        checkpoint_manager=checkpoint_manager,
    )


def main(
    args: Any,
    overrides: list[str] | None = None,
) -> None:
    config_path = str(args.config)
    dry_run = bool(args.dry_run)
    logger.info(
        "Starting training from config=%s",
        config_path,
    )
    run_training_from_config(
        config_path,
        dry_run=dry_run,
        overrides=overrides,
    )
    logger.info("Training completed")


if __name__ == "__main__":
    argv = sys.argv
    parser = argparse.ArgumentParser(description="YAML-only training entrypoint.", )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=("Path to training YAML config (schema v2)."),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Parse config and build runtime, "
              "but do not start training."),
    )
    args, unknown = parser.parse_known_args(argv[1:])
    main(args, overrides=unknown if unknown else None)
