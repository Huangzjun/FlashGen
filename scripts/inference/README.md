# Flashgen DMD Inference

This folder only keeps Wan DMD inference examples.

```powershell
$env:FLASHGEN_ATTENTION_BACKEND = "TORCH_SDPA"
flashgen generate --config scripts/inference/inference_wan_DMD_1_3B.yaml
```

Flashgen only ships the `TORCH_SDPA` attention backend.
