# FlashGen Modular Training Framework

This is FlashGen's NPU-only port of FastVideo's newer `train/` stack. It
replaces direct launches of the legacy monolithic scripts in
`flashgen/training/` for Wan DMD distillation.

## Supported Surface

- Model plugin: `flashgen.train.models.wan.wan.WanModel`
- Method: `flashgen.train.methods.distribution_matching.dmd2.DMD2Method`
- Callbacks: gradient clipping and EMA
- Runtime: Ascend NPU with `FLASHGEN_ATTENTION_BACKEND=TORCH_SDPA`
- Config: YAML plus dotted CLI overrides

Other FastVideo model families and CUDA attention paths are intentionally not
exposed in this fork.

## Usage

```bash
torchrun --nproc_per_node 8 -m flashgen.train.entrypoint.train \
  --config scripts/distill/wan_dmd_npu.yaml
```

See `scripts/distill/v1_distill_dmd_wan.sh` for the maintained launch script.
