# SPDX-License-Identifier: Apache-2.0

from flashgen.train.methods.base import TrainingMethod

__all__ = [
    "TrainingMethod",
    "DMD2Method",
]


def __getattr__(name: str) -> object:
    if name == "DMD2Method":
        from flashgen.train.methods.distribution_matching.dmd2 import DMD2Method
        return DMD2Method
    raise AttributeError(name)
