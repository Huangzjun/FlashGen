# Flashgen

Flashgen is a slim, NPU-oriented (Ascend) **step-distillation training library**.
It covers training only: distilling multi-step diffusion models into few-step
student models. Inference is handled by other libraries.

## Scope

- Built-in algorithm: DMD (Distribution Matching Distillation), three-network
  setup (student generator / frozen real-score teacher / trainable fake-score critic).
- Target model: Wan 2.1 1.3B T2V.
- Hardware: NPU (Ascend), HCCL communication, Torch SDPA attention only.
- Distillation steps: `[1000, 757, 522]` (default 3-step student).

## Training

Launch distributed DMD distillation via `torchrun`:

```bash
bash scripts/distill/v1_distill_dmd_wan.sh
```

Entry point: `flashgen.train.entrypoint.train` (YAML config via `scripts/distill/v1_distill_dmd_wan.sh`).

See `docs/架构设计.md` for the full architecture.
