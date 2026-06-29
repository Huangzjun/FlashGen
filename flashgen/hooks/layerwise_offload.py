from contextlib import contextmanager
from typing import Any

from torch import nn

from flashgen.hooks.hooks import ForwardHook
from flashgen.logger import init_logger

logger = init_logger(__name__)


class LayerwiseOffloadHook(ForwardHook):
    """NPU-only no-op placeholder for the removed CUDA layerwise offload path."""

    @classmethod
    def name(cls) -> str:
        return "LayerwiseOffloadHook"

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any):
        return output

    @contextmanager
    def mutate_params_scope(self):
        yield


def enable_layerwise_offload(model: nn.Module, is_replace: bool = False):
    logger.warning("Layerwise offload is disabled in Flashgen NPU-only runtime.")
