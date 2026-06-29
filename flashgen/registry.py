# SPDX-License-Identifier: Apache-2.0
"""Wan DMD-only registry for Flashgen."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from flashgen.api.sampling_param import SamplingParam
from flashgen.configs.pipelines.base import PipelineConfig
from flashgen.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
from flashgen.fastvideo_args import WorkloadType
from flashgen.logger import init_logger
from flashgen.utils import maybe_download_model_index, verify_model_config_and_directory

logger = init_logger(__name__)

if TYPE_CHECKING:
    from flashgen.pipelines.composed_pipeline_base import ComposedPipelineBase
    from flashgen.pipelines.pipeline_registry import PipelineType

_PIPELINE_REGISTRY: dict[str, dict[str, type[ComposedPipelineBase]]] = {}
_PIPELINE_CONFIG_REGISTRY: dict[str, tuple[type[PipelineConfig], type[SamplingParam]]] = {}


@dataclasses.dataclass
class ConfigInfo:
    sampling_param_cls: type[SamplingParam] | None
    pipeline_config_cls: type[PipelineConfig]
    workload_types: tuple[WorkloadType, ...]
    model_family: str | None = None
    default_preset: str | None = None


_CONFIG_REGISTRY: dict[str, ConfigInfo] = {}
_MODEL_HF_PATH_TO_NAME: dict[str, str] = {}
_MODEL_NAME_DETECTORS: list[tuple[str, Callable[[str], bool]]] = []


def _discover_and_register_pipelines() -> None:
    if _PIPELINE_REGISTRY:
        return

    from flashgen.pipelines.pipeline_registry import import_pipeline_classes

    pipeline_classes = import_pipeline_classes()
    for pipeline_type, pipeline_dict in pipeline_classes.items():
        _PIPELINE_REGISTRY[pipeline_type] = pipeline_dict
        for pipeline_cls in pipeline_dict.values():
            if pipeline_cls is None:
                continue
            if hasattr(pipeline_cls, "pipeline_config_cls") and hasattr(pipeline_cls, "sampling_params_cls"):
                _PIPELINE_CONFIG_REGISTRY[pipeline_cls.__name__] = (
                    pipeline_cls.pipeline_config_cls,
                    pipeline_cls.sampling_params_cls,
                )


def get_pipeline_config_classes(pipeline_class_name: str) -> tuple[type[PipelineConfig], type[SamplingParam]] | None:
    _discover_and_register_pipelines()
    return _PIPELINE_CONFIG_REGISTRY.get(pipeline_class_name)


def register_configs(
    sampling_param_cls: type[SamplingParam] | None,
    pipeline_config_cls: type[PipelineConfig],
    workload_types: tuple[WorkloadType, ...],
    hf_model_paths: list[str] | None = None,
    model_detectors: list[Callable[[str], bool]] | None = None,
    model_family: str | None = None,
    default_preset: str | None = None,
) -> None:
    model_id = str(len(_CONFIG_REGISTRY))
    _CONFIG_REGISTRY[model_id] = ConfigInfo(
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=pipeline_config_cls,
        workload_types=workload_types,
        model_family=model_family,
        default_preset=default_preset,
    )

    if hf_model_paths:
        for path in hf_model_paths:
            _MODEL_HF_PATH_TO_NAME[path] = model_id

    if model_detectors:
        for detector in model_detectors:
            _MODEL_NAME_DETECTORS.append((model_id, detector))


def get_model_short_name(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/")[-1]
    return model_id


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
        sampling_param_cls=None,
        pipeline_config_cls=FastWan2_1_T2V_480P_Config,
        workload_types=(WorkloadType.T2V, ),
        hf_model_paths=[
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        ],
        model_detectors=[lambda path: "wandmdpipeline" in path.lower()],
        model_family="wan",
        default_preset="fast_wan_t2v_480p",
    )


@dataclasses.dataclass
class ModelInfo:
    pipeline_cls: type[ComposedPipelineBase]
    sampling_param_cls: type[SamplingParam]
    pipeline_config_cls: type[PipelineConfig]


@lru_cache(maxsize=32)
def get_model_info(
    model_path: str,
    pipeline_type: PipelineType | str | None = None,
    workload_type: WorkloadType | None = None,
    override_pipeline_cls_name: str | None = None,
) -> ModelInfo:
    from flashgen.pipelines.pipeline_registry import PipelineType, get_pipeline_registry

    if pipeline_type is None:
        pipeline_type = PipelineType.BASIC
    elif isinstance(pipeline_type, str):
        pipeline_type = PipelineType.from_string(pipeline_type)

    if workload_type is None:
        workload_type = WorkloadType.T2V

    if os.path.exists(model_path):
        config = verify_model_config_and_directory(model_path)
    else:
        config = maybe_download_model_index(model_path)

    pipeline_name = config.get("_class_name")
    if override_pipeline_cls_name:
        logger.info("Overriding pipeline class name from %s to %s", pipeline_name, override_pipeline_cls_name)
        pipeline_name = override_pipeline_cls_name

    if pipeline_name is None:
        raise ValueError("Model config does not contain a _class_name attribute. Only diffusers format is supported.")

    pipeline_registry = get_pipeline_registry(pipeline_type)
    pipeline_cls = pipeline_registry.resolve_pipeline_cls(pipeline_name, pipeline_type, workload_type)

    config_info = _get_config_info(model_path, raise_on_missing=True)
    assert config_info is not None, "config_info must be resolved"

    return ModelInfo(
        pipeline_cls=pipeline_cls,
        sampling_param_cls=config_info.sampling_param_cls or SamplingParam,
        pipeline_config_cls=config_info.pipeline_config_cls,
    )


def get_pipeline_config_cls_from_name(pipeline_name_or_path: str) -> type[PipelineConfig]:
    config_info = _get_config_info(pipeline_name_or_path, raise_on_missing=False)
    if config_info is None:
        raise ValueError(
            f"No match found for pipeline {pipeline_name_or_path}, please check the pipeline name or path.")
    return config_info.pipeline_config_cls


def get_sampling_param_cls_for_name(pipeline_name_or_path: str) -> Any | None:
    config_info = _get_config_info(pipeline_name_or_path, raise_on_missing=False)
    if config_info is None:
        logger.warning("No match found for pipeline %s, using default sampling param.", pipeline_name_or_path)
        return None
    return config_info.sampling_param_cls


def _register_presets() -> None:
    from flashgen.api.presets import register_preset
    from flashgen.pipelines.basic.wan.presets import ALL_PRESETS as WAN_PRESETS

    for preset in WAN_PRESETS:
        register_preset(preset)


_register_configs()
_register_presets()


def get_model_family(model_path: str) -> str | None:
    config_info = _get_config_info(model_path, raise_on_missing=False)
    if config_info is None:
        return None
    return config_info.model_family


def get_default_preset(model_path: str) -> str | None:
    config_info = _get_config_info(model_path, raise_on_missing=False)
    if config_info is None:
        return None
    return config_info.default_preset


def get_preset_selection(model_path: str) -> tuple[str | None, str | None]:
    config_info = _get_config_info(model_path, raise_on_missing=False)
    if config_info is None:
        return None, None
    return config_info.default_preset, config_info.model_family


def get_registered_model_paths() -> list[str]:
    return sorted(_MODEL_HF_PATH_TO_NAME.keys())


def get_registered_models_with_workloads(workload_type: str | None = None, ) -> list[dict[str, Any]]:
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


__all__ = [
    "ConfigInfo",
    "ModelInfo",
    "get_default_preset",
    "get_model_family",
    "get_model_info",
    "get_pipeline_config_cls_from_name",
    "get_registered_model_paths",
    "get_registered_models_with_workloads",
    "get_sampling_param_cls_for_name",
    "get_pipeline_config_classes",
]
