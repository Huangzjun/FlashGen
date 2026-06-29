# SPDX-License-Identifier: Apache-2.0
"""Wan 2.1 1.3B DMD inference preset."""

from flashgen.api.presets import InferencePreset, PresetStageSpec

_NEGATIVE_PROMPT_EN = ("Bright tones, overexposed, static, blurred details, subtitles,"
                       " style, works, paintings, images, static, overall gray, worst"
                       " quality, low quality, JPEG compression residue, ugly,"
                       " incomplete, extra fingers, poorly drawn hands, poorly drawn"
                       " faces, deformed, disfigured, misshapen limbs, fused fingers,"
                       " still picture, messy background, three legs, many people in"
                       " the background, walking backwards")

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Main DMD denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

FAST_WAN_T2V_480P = InferencePreset(
    name="fast_wan_t2v_480p",
    version=1,
    model_family="wan",
    description="FastWan 2.1 T2V 1.3B DMD at 480p",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 448,
        "width": 832,
        "num_frames": 61,
        "fps": 16,
        "guidance_scale": 3.0,
        "num_inference_steps": 3,
        "negative_prompt": _NEGATIVE_PROMPT_EN,
    },
)

ALL_PRESETS = (FAST_WAN_T2V_480P, )
