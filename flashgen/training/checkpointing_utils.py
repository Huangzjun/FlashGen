# SPDX-License-Identifier: Apache-2.0
import random
from typing import Any

import numpy as np
import torch
import torch.distributed.checkpoint.stateful
from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
                                                     set_model_state_dict, set_optimizer_state_dict)


class ModelWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        state_dict = get_model_state_dict(self.model)

        param_requires_grad = {
            k.replace("._checkpoint_wrapped_module.", ".")
            for k, v in self.model.named_parameters() if v.requires_grad
        }

        filtered_state_dict = {k: v for k, v in state_dict.items() if k in param_requires_grad}

        return filtered_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )


class OptimizerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        return get_optimizer_state_dict(  # type: ignore[no-any-return]
            self.model,
            self.optimizer,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


class SchedulerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def state_dict(self) -> dict[str, Any]:
        return {"scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict["scheduler"])


class RandomStateWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, noise_generator: torch.Generator | None = None) -> None:
        self.noise_generator = noise_generator

    def state_dict(self) -> dict[str, Any]:
        state = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        if hasattr(torch, "npu") and torch.npu.is_available():
            state["npu_rng_state"] = torch.npu.get_rng_state()
            if torch.npu.device_count() > 1:
                state["npu_rng_state_all"] = torch.npu.get_rng_state_all()

        if self.noise_generator is not None:
            state["noise_generator_state"] = self.noise_generator.get_state()

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "torch_rng_state" in state_dict:
            torch.set_rng_state(state_dict["torch_rng_state"])

        if "numpy_rng_state" in state_dict:
            np.random.set_state(state_dict["numpy_rng_state"])

        if "python_rng_state" in state_dict:
            random.setstate(state_dict["python_rng_state"])

        # Restore NPU random state. The cuda_* keys are accepted for
        # compatibility with checkpoints produced before the NPU-only rename.
        if hasattr(torch, "npu") and torch.npu.is_available():
            if "npu_rng_state" in state_dict:
                torch.npu.set_rng_state(state_dict["npu_rng_state"])
            elif "cuda_rng_state" in state_dict:
                torch.npu.set_rng_state(state_dict["cuda_rng_state"])
            if "npu_rng_state_all" in state_dict:
                torch.npu.set_rng_state_all(state_dict["npu_rng_state_all"])
            elif "cuda_rng_state_all" in state_dict:
                torch.npu.set_rng_state_all(state_dict["cuda_rng_state_all"])

        # Restore noise generator state
        if "noise_generator_state" in state_dict and self.noise_generator is not None:
            self.noise_generator.set_state(state_dict["noise_generator_state"])
