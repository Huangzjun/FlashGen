# Flashgen

Flashgen is a slim NPU-oriented fork scaffold based on `FastVideo_fork`.
This tree keeps the core runtime architecture needed for Wan 2.1 1.3B DMD inference:
configuration parsing, CLI generation, model loading, distributed/NPU platform
support, pipeline stages, worker execution, and the Wan DMD pipeline.

## Scope

Included DMD entry points:

- `WanDMDPipeline`
- `DmdDenoisingStage`
- `scripts/inference/inference_wan_DMD_1_3B.yaml`

The BASIC pipeline registry is intentionally limited to Wan 2.1 1.3B DMD.
Other model families, Wan I2V, Wan 2.2, and custom kernel attention backends
are not part of the supported Flashgen surface.

## Attention Backend

Flashgen only ships `TORCH_SDPA`. The current Wan 2.1 1.3B DMD flow does not
import or depend on `fastvideo_kernel`.

## Example

```powershell
$env:FLASHGEN_ATTENTION_BACKEND = "TORCH_SDPA"
flashgen generate --config scripts/inference/inference_wan_DMD_1_3B.yaml
```
