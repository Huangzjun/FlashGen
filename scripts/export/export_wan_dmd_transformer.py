# SPDX-License-Identifier: Apache-2.0
"""Export a trained Wan DMD generator transformer for external inference runtimes.

The DMD training checkpoint already writes a consolidated generator transformer at:

    checkpoint-*/generator_inference_transformer/

This script copies that minimal Diffusers-style transformer component into:

    output_dir/transformer/

No inference pipeline is loaded here. The exported directory is intended for
MindIE SD or another runtime that can consume Wan Diffusers transformer weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REQUIRED_FILES = ("config.json", "diffusion_pytorch_model.safetensors")


def _resolve_transformer_dir(checkpoint_dir: Path) -> Path:
    candidates = [
        checkpoint_dir / "generator_inference_transformer",
        checkpoint_dir / "transformer",
    ]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in REQUIRED_FILES):
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find an exported generator transformer. "
        f"Expected {REQUIRED_FILES} under one of: {searched}")


def export_transformer(checkpoint_dir: Path, output_dir: Path, overwrite: bool = False) -> Path:
    source_dir = _resolve_transformer_dir(checkpoint_dir)
    target_dir = output_dir / "transformer"

    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{target_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        shutil.copy2(source_dir / name, target_dir / name)

    metadata = {
        "format": "diffusers_wan_transformer_component",
        "source_checkpoint": str(checkpoint_dir),
        "source_transformer": str(source_dir),
        "files": list(REQUIRED_FILES),
    }
    with open(output_dir / "export_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return target_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Flashgen Wan DMD transformer weights.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to a Flashgen checkpoint directory, e.g. outputs_dmd/wan_finetune/checkpoint-400.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory. The transformer component is written to output-dir/transformer.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output-dir/transformer if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported_dir = export_transformer(args.checkpoint_dir, args.output_dir, args.overwrite)
    print(f"Exported transformer to {exported_dir}")


if __name__ == "__main__":
    main()
