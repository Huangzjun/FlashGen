# SPDX-License-Identifier: Apache-2.0
"""Train-only model configuration registry for Flashgen."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flashgen.configs.pipelines.base import PipelineConfig
from flashgen.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
from flashgen.flashgen_args import WorkloadType
from flashgen.logger import init_logger
from flashgen.utils import maybe_download_model_index, verify_model_config_and_directory

logger = init_logger(__name__)


@dataclass
class ConfigInfo:
    pipeline_config_cls: type[PipelineConfig]
    workload_types: tuple[WorkloadType, ...]
    model_family: str | None = None


_CONFIG_REGISTRY: dict[str, ConfigInfo] = {}
_MODEL_HF_PATH_TO_NAME: dict[str, str] = {}
_MODEL_NAME_DETECTORS: list[tuple[str, Callable[[str], bool]]] = []


def register_configs(
    pipeline_config_cls: type[PipelineConfig],
    workload_types: tuple[WorkloadType, ...],
    hf_model_paths: list[str] | None = None,
    model_detectors: list[Callable[[str], bool]] | None = None,
    model_family: str | None = None,
) -> None:
    model_id = str(len(_CONFIG_REGISTRY))
    _CONFIG_REGISTRY[model_id] = ConfigInfo(
        pipeline_config_cls=pipeline_config_cls,
        workload_types=workload_types,
        model_family=model_family,
    )

    for path in hf_model_paths or []:
        _MODEL_HF_PATH_TO_NAME[path] = model_id

    for detector in model_detectors or []:
        _MODEL_NAME_DETECTORS.append((model_id, detector))


def get_model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _get_config_info(model_path: str, *, raise_on_missing: bool = True) -> ConfigInfo | None:
    if model_path in _MODEL_HF_PATH_TO_NAME:
        return _CONFIG_REGISTRY.get(_MODEL_HF_PATH_TO_NAME[model_path])

    model_name = get_model_short_name(model_path.lower())
    for registered_model_hf_id in sorted(_MODEL_HF_PATH_TO_NAME.keys(), key=len, reverse=True):
        registered_model_name = get_model_short_name(registered_model_hf_id.lower())
        if registered_model_name == model_name:
            return _CONFIG_REGISTRY.get(_MODEL_HF_PATH_TO_NAME[registered_model_hf_id])

    if os.path.exists(model_path):
        config = verify_model_config_and_directory(model_path)
    else:
        config = maybe_download_model_index(model_path)

    pipeline_name = config.get("_class_name", "").lower()
    for model_id, detector in _MODEL_NAME_DETECTORS:
        if detector(model_path.lower()) or detector(pipeline_name):
            return _CONFIG_REGISTRY.get(model_id)

    if raise_on_missing:
        raise RuntimeError(f"No model info found for model path: {model_path}")
    return None


def _register_configs() -> None:
    register_configs(
        pipeline_config_cls=FastWan2_1_T2V_480P_Config,
        workload_types=(WorkloadType.T2V, ),
        hf_model_paths=[
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ],
        model_detectors=[lambda path: "wandmdpipeline" in path.lower()],
        model_family="wan",
    )


def get_pipeline_config_cls_from_name(pipeline_name_or_path: str) -> type[PipelineConfig]:
    config_info = _get_config_info(pipeline_name_or_path, raise_on_missing=False)
    if config_info is None:
        raise ValueError(
            f"No match found for pipeline {pipeline_name_or_path}, please check the pipeline name or path.")
    return config_info.pipeline_config_cls


def get_model_family(model_path: str) -> str | None:
    config_info = _get_config_info(model_path, raise_on_missing=False)
    if config_info is None:
        return None
    return config_info.model_family


def get_default_preset(model_path: str) -> None:
    return None


def get_preset_selection(model_path: str) -> tuple[None, str | None]:
    return None, get_model_family(model_path)


def get_registered_model_paths() -> list[str]:
    return sorted(_MODEL_HF_PATH_TO_NAME.keys())


def get_registered_models_with_workloads(workload_type: str | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(_MODEL_HF_PATH_TO_NAME.keys()):
        model_id = _MODEL_HF_PATH_TO_NAME[path]
        config_info = _CONFIG_REGISTRY.get(model_id)
        if config_info is None:
            continue
        workload_values = [w.value for w in config_info.workload_types]
        if workload_type is not None and workload_type.lower() not in workload_values:
            continue
        result.append({
            "id": path,
            "label": path.split("/")[-1].replace("-", " ").replace("_", " "),
            "workload_types": workload_values,
        })
    return result


_register_configs()


__all__ = [
    "ConfigInfo",
    "get_default_preset",
    "get_model_family",
    "get_pipeline_config_cls_from_name",
    "get_preset_selection",
    "get_registered_model_paths",
    "get_registered_models_with_workloads",
]
