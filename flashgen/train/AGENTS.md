# `flashgen/train/` — NPU Modular Training

YAML-driven trainer ported from FastVideo's new `train/` stack. This directory
is the preferred location for new FlashGen training work; the sibling
`flashgen/training/` stack is legacy maintenance mode.

## Current Scope

This fork is NPU-only. Keep the supported surface narrow unless a user asks for
more:

- `models/wan/wan.py`: Wan training model wrapper
- `methods/distribution_matching/dmd2.py`: DMD2 distillation
- `callbacks/grad_clip.py` and `callbacks/ema.py`
- `entrypoint/train.py`: YAML launch entrypoint

Do not reintroduce CUDA-only attention backends, generic FastVideo validation,
RL, or non-Wan model plugins without doing the NPU adaptation work.

## Layout

```
train/
├── trainer.py
├── entrypoint/
│   └── train.py
├── methods/
│   ├── base.py
│   └── distribution_matching/dmd2.py
├── models/
│   ├── base.py
│   └── wan/wan.py
├── callbacks/
│   ├── callback.py
│   ├── ema.py
│   └── grad_clip.py
└── utils/
    ├── config.py
    ├── training_config.py
    ├── builder.py
    ├── checkpoint.py
    ├── dataloader.py
    └── moduleloader.py
```

## Rules

- `flashgen.train.*` must not import from legacy `flashgen.training.*` pipeline
  classes. Shared utilities such as `training_utils.py` are acceptable when
  already used by both stacks.
- Attention backend must remain `FLASHGEN_ATTENTION_BACKEND=TORCH_SDPA`.
- Preserve FlashGen's existing Wan DMD behavior where it differs from
  FastVideo, especially NPU device handling, zero fallback for missing negative
  prompt embeddings, and the existing DMD CFG parameterization.
- Add config knobs to `utils/training_config.py` with explicit defaults.
