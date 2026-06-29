# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from flashgen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from flashgen.configs.models.dits import WanVideoConfig
from flashgen.configs.models.encoders import BaseEncoderOutput, T5Config
from flashgen.configs.models.vaes import WanVAEConfig
from flashgen.configs.pipelines.base import PipelineConfig


def t5_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0)
    return prompt_embeds_tensor


@dataclass
class WanT2V480PConfig(PipelineConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    # WanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 3.0

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (T5Config(), ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda: (t5_postprocess_text, ))

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32", ))

    # self-forcing params
    warp_denoising_step: bool = True

    # WanConfig-specific added parameters

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class FastWan2_1_T2V_480P_Config(WanT2V480PConfig):
    """Base configuration for FastWan T2V 1.3B 480P pipeline architecture with DMD"""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(default_factory=lambda: [1000, 757, 522])
