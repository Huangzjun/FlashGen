# SPDX-License-Identifier: Apache-2.0
"""NPU-only platform selection for Flashgen."""

import traceback
from typing import TYPE_CHECKING

from flashgen.logger import init_logger
from flashgen.platforms.interface import AttentionBackendEnum  # noqa: F401
from flashgen.platforms.interface import Platform, PlatformEnum
from flashgen.utils import resolve_obj_by_qualname

logger = init_logger(__name__)


def npu_platform_plugin() -> str:
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Flashgen is NPU-only: PyTorch_NPU must be installed.") from exc

    if not hasattr(torch, "npu"):
        raise RuntimeError("Flashgen is NPU-only: use an Ascend-adapted PyTorch with torch.npu.")

    if not torch.npu.is_available():
        raise RuntimeError("Flashgen is NPU-only: no available NPU device was detected.")

    logger.info("NPU is available")
    return "flashgen.platforms.npu.NPUPlatform"


def resolve_current_platform_cls_qualname() -> str:
    return npu_platform_plugin()


_current_platform = None
_init_trace: str = ""

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str):
    if name == "current_platform":
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    if name in globals():
        return globals()[name]
    raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = ["AttentionBackendEnum", "Platform", "PlatformEnum", "current_platform", "_init_trace"]
