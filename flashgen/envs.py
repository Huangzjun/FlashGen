# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/envs.py

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    FLASHGEN_RINGBUFFER_WARNING_INTERVAL: int = 60
    FLASHGEN_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    FLASHGEN_CACHE_ROOT: str = os.path.expanduser("~/.cache/flashgen")
    FLASHGEN_CONFIG_ROOT: str = os.path.expanduser("~/.config/flashgen")
    FLASHGEN_CONFIGURE_LOGGING: int = 1
    FLASHGEN_RAY_PER_WORKER_GPUS: float = 1.0
    FLASHGEN_LOGGING_LEVEL: str = "INFO"
    FLASHGEN_LOGGING_PREFIX: str = ""
    FLASHGEN_LOGGING_CONFIG_PATH: str | None = None
    FLASHGEN_TRACE_FUNCTION: int = 0
    FLASHGEN_ATTENTION_BACKEND: str | None = None
    FLASHGEN_WORKER_MULTIPROC_METHOD: str = "spawn"
    FLASHGEN_TARGET_DEVICE: str = "npu"
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    CMAKE_BUILD_TYPE: str | None = None
    VERBOSE: bool = False
    FLASHGEN_TORCH_PROFILER_DIR: str | None = None
    FLASHGEN_TORCH_PROFILER_RECORD_SHAPES: bool = False
    FLASHGEN_TORCH_PROFILER_WITH_PROFILE_MEMORY: bool = False
    FLASHGEN_TORCH_PROFILER_WITH_STACK: bool = True
    FLASHGEN_TORCH_PROFILER_WITH_FLOPS: bool = False
    FLASHGEN_TORCH_PROFILER_WAIT_STEPS: int = 2
    FLASHGEN_TORCH_PROFILER_WARMUP_STEPS: int = 1
    FLASHGEN_TORCH_PROFILER_ACTIVE_STEPS: int = 2
    FLASHGEN_TORCH_PROFILE_REGIONS: str = ""
    FLASHGEN_TRACE_ACTIVATIONS: bool = False
    FLASHGEN_TRACE_LAYERS: str = ""
    FLASHGEN_TRACE_STATS: str = "abs_mean,sum"
    FLASHGEN_TRACE_OUTPUT: str = "/tmp/fv_trace_<pid>.jsonl"
    FLASHGEN_TRACE_STEPS: str = ""
    FLASHGEN_SERVER_DEV_MODE: bool = False
    FLASHGEN_STAGE_LOGGING: bool = False
    FLASHGEN_CFG_GATE_STEP: float = 1.0
    FLASHGEN_HOST_IP: str = ""
    FLASHGEN_LOOPBACK_IP: str = ""


def get_default_cache_root() -> str:
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root() -> str:
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

environment_variables: dict[str, Callable[[], Any]] = {

    # ================== Installation Time Env Vars ==================

    # Target device of Flashgen. This fork keeps the NPU runtime path.
    "FLASHGEN_TARGET_DEVICE":
    lambda: os.getenv("FLASHGEN_TARGET_DEVICE", "npu"),

    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),

    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS":
    lambda: os.getenv("NVCC_THREADS", None),

    # If set, flashgen will use precompiled binaries (*.so)
    "FLASHGEN_USE_PRECOMPILED":
    lambda: bool(os.environ.get("FLASHGEN_USE_PRECOMPILED")) or bool(
        os.environ.get("FLASHGEN_PRECOMPILED_WHEEL_LOCATION")),

    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),

    # If set, flashgen will print verbose logs during installation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),

    # Root directory for FLASHGEN configuration files
    # Defaults to `~/.config/flashgen` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how flashgen finds its configuration files
    # during runtime, but also affects how flashgen installs its configuration
    # files during **installation**.
    "FLASHGEN_CONFIG_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "FLASHGEN_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "flashgen"),
        )),

    # ================== Runtime Env Vars ==================

    # Root directory for FLASHGEN cache files
    # Defaults to `~/.cache/flashgen` unless `XDG_CACHE_HOME` is set
    "FLASHGEN_CACHE_ROOT":
    lambda: os.path.expanduser(os.getenv(
        "FLASHGEN_CACHE_ROOT",
        os.path.join(get_default_cache_root(), "flashgen"),
    )),

    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    "FLASHGEN_HOST_IP":
    lambda: os.getenv("FLASHGEN_HOST_IP", ""),

    # Used to force set up loopback IP
    "FLASHGEN_LOOPBACK_IP":
    lambda: os.getenv("FLASHGEN_LOOPBACK_IP", ""),

    # Number of GPUs per worker in Ray, if it is set to be a fraction,
    # it allows ray to schedule multiple actors on a single GPU,
    # so that users can colocate other actors on the same GPUs as Flashgen.
    "FLASHGEN_RAY_PER_WORKER_GPUS":
    lambda: float(os.getenv("FLASHGEN_RAY_PER_WORKER_GPUS", "1.0")),

    # Interval in seconds to log a warning message when the ring buffer is full
    "FLASHGEN_RINGBUFFER_WARNING_INTERVAL":
    lambda: int(os.environ.get("FLASHGEN_RINGBUFFER_WARNING_INTERVAL", "60")),

    "FLASHGEN_NCCL_SO_PATH":
    lambda: os.environ.get("FLASHGEN_NCCL_SO_PATH", None),

    # HCCL can be found in the locations specified by `LD_LIBRARY_PATH`.
    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),

    # Internal flag to enable Dynamo fullgraph capture
    "FLASHGEN_TEST_DYNAMO_FULLGRAPH_CAPTURE":
    lambda: bool(os.environ.get("FLASHGEN_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") != "0"),

    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK":
    lambda: int(os.environ.get("LOCAL_RANK", "0")),

    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES":
    lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),

    # timeout for each iteration in the engine
    "FLASHGEN_ENGINE_ITERATION_TIMEOUT_S":
    lambda: int(os.environ.get("FLASHGEN_ENGINE_ITERATION_TIMEOUT_S", "60")),

    # Logging configuration
    # If set to 0, flashgen will not configure logging
    # If set to 1, flashgen will configure logging using the default configuration
    #    or the configuration file specified by FLASHGEN_LOGGING_CONFIG_PATH
    "FLASHGEN_CONFIGURE_LOGGING":
    lambda: int(os.getenv("FLASHGEN_CONFIGURE_LOGGING", "1")),
    "FLASHGEN_LOGGING_CONFIG_PATH":
    lambda: os.getenv("FLASHGEN_LOGGING_CONFIG_PATH"),

    # this is used for configuring the default logging level
    "FLASHGEN_LOGGING_LEVEL":
    lambda: os.getenv("FLASHGEN_LOGGING_LEVEL", "INFO"),

    # if set, FLASHGEN_LOGGING_PREFIX will be prepended to all log messages
    "FLASHGEN_LOGGING_PREFIX":
    lambda: os.getenv("FLASHGEN_LOGGING_PREFIX", ""),

    # Trace function calls
    # If set to 1, flashgen will trace function calls
    # Useful for debugging
    "FLASHGEN_TRACE_FUNCTION":
    lambda: int(os.getenv("FLASHGEN_TRACE_FUNCTION", "0")),

    # Backend for attention computation
    # Available option:
    # - "TORCH_SDPA": use torch scaled dot-product attention
    "FLASHGEN_ATTENTION_BACKEND":
    lambda: os.getenv("FLASHGEN_ATTENTION_BACKEND", None),

    # Use dedicated multiprocess context for workers.
    "FLASHGEN_WORKER_MULTIPROC_METHOD":
    lambda: os.getenv("FLASHGEN_WORKER_MULTIPROC_METHOD", "spawn"),

    # Enables torch profiler if set. Path to the directory where torch profiler
    # traces are saved. Note that it must be an absolute path.
    "FLASHGEN_TORCH_PROFILER_DIR":
    lambda: (None if os.getenv("FLASHGEN_TORCH_PROFILER_DIR", None) is None else os.path.expanduser(
        os.getenv("FLASHGEN_TORCH_PROFILER_DIR", "."))),

    # Enable torch profiler to record shapes if set
    # FLASHGEN_TORCH_PROFILER_RECORD_SHAPES=1. If not set, torch profiler will
    # not record shapes.
    "FLASHGEN_TORCH_PROFILER_RECORD_SHAPES":
    lambda: bool(os.getenv("FLASHGEN_TORCH_PROFILER_RECORD_SHAPES", "0") != "0"),

    # Enable torch profiler to profile memory if set
    # FLASHGEN_TORCH_PROFILER_WITH_PROFILE_MEMORY=1. If not set, torch profiler
    # will not profile memory.
    "FLASHGEN_TORCH_PROFILER_WITH_PROFILE_MEMORY":
    lambda: bool(os.getenv("FLASHGEN_TORCH_PROFILER_WITH_PROFILE_MEMORY", "0") != "0"),

    # Enable torch profiler to profile stack if set
    # FLASHGEN_TORCH_PROFILER_WITH_STACK=1. If not set, torch profiler WILL
    # profile stack by default.
    "FLASHGEN_TORCH_PROFILER_WITH_STACK":
    lambda: bool(os.getenv("FLASHGEN_TORCH_PROFILER_WITH_STACK", "1") != "0"),

    # Enable torch profiler to profile flops if set
    # FLASHGEN_TORCH_PROFILER_WITH_FLOPS=1. If not set, torch profiler will
    # not profile flops.
    "FLASHGEN_TORCH_PROFILER_WITH_FLOPS":
    lambda: bool(os.getenv("FLASHGEN_TORCH_PROFILER_WITH_FLOPS", "0") != "0"),
    # Wait steps per profiling cycle (torch.profiler.schedule wait parameter)
    # Defaults to 2 if not set.
    "FLASHGEN_TORCH_PROFILER_WAIT_STEPS":
    lambda: int(os.getenv("FLASHGEN_TORCH_PROFILER_WAIT_STEPS", "2")),
    # Warmup steps per profiling cycle (torch.profiler.schedule warmup parameter)
    # Defaults to 1 if not set.
    "FLASHGEN_TORCH_PROFILER_WARMUP_STEPS":
    lambda: int(os.getenv("FLASHGEN_TORCH_PROFILER_WARMUP_STEPS", "1")),
    # Active steps per profiling cycle (torch.profiler.schedule active parameter)
    # Defaults to 2 if not set.
    "FLASHGEN_TORCH_PROFILER_ACTIVE_STEPS":
    lambda: int(os.getenv("FLASHGEN_TORCH_PROFILER_ACTIVE_STEPS", "2")),
    "FLASHGEN_TORCH_PROFILE_REGIONS":
    lambda: os.getenv("FLASHGEN_TORCH_PROFILE_REGIONS", ""),

    # Enable activation trace hooks if set.
    "FLASHGEN_TRACE_ACTIVATIONS":
    lambda: bool(os.getenv("FLASHGEN_TRACE_ACTIVATIONS", "0") != "0"),
    # Regex filter for traced module names. Empty means all modules.
    "FLASHGEN_TRACE_LAYERS":
    lambda: os.getenv("FLASHGEN_TRACE_LAYERS", ""),
    # Comma-separated activation stats to dump for each output tensor.
    "FLASHGEN_TRACE_STATS":
    lambda: os.getenv("FLASHGEN_TRACE_STATS", "abs_mean,sum"),
    # JSONL sink path. The literal <pid> is replaced at runtime.
    "FLASHGEN_TRACE_OUTPUT":
    lambda: os.getenv("FLASHGEN_TRACE_OUTPUT", "/tmp/fv_trace_<pid>.jsonl"),
    # Comma-separated denoise step indices. Empty means all steps.
    "FLASHGEN_TRACE_STEPS":
    lambda: os.getenv("FLASHGEN_TRACE_STEPS", ""),

    # If set, flashgen will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "FLASHGEN_SERVER_DEV_MODE":
    lambda: bool(int(os.getenv("FLASHGEN_SERVER_DEV_MODE", "0"))),

    # If set, flashgen will enable stage logging, which will print the time
    # taken for each stage
    "FLASHGEN_STAGE_LOGGING":
    lambda: bool(int(os.getenv("FLASHGEN_STAGE_LOGGING", "0"))),

    # CFG gating fraction for stale-uncond reuse (Adaptive Guidance / LinearAG
    # variant — Castillo et al. 2023, arXiv:2312.12487).  Float in [0, 1].
    # Interpretation: for step index `i < len(timesteps) * X`, run both
    # cond and uncond forwards and refresh delta_cached = cond - uncond.
    # Once `i >= len(timesteps) * X`, skip the uncond forward and reuse
    # the cached delta:  noise_pred = cond + (guidance_scale - 1) * delta.
    #
    # Edge cases:
    #   1.0 (default) : disables gating; identical to baseline two-pass CFG.
    #   0.5           : run uncond for the first half of steps, reuse delta
    #                    for the second half (~25% inference time saved on
    #                    bandwidth-bound SP setups).
    #   0.0           : step 0 still computes uncond fresh (cache is empty
    #                    at start) — all subsequent steps reuse the step-0
    #                    delta.  This is the most aggressive setting; does
    #                    NOT mean "no uncond forward ever."
    #
    # Caveats:
    #   - Algorithmically approximate; not bit-exact vs baseline CFG.
    #     Validate per-pipeline with SSIM / VBench before lowering below 1.0.
    #   - Interaction with `guidance_rescale > 0` is unvalidated; the
    #     denoising stage logs a warning when both are active.
    #   - Wan2.2 high/low-noise expert switch invalidates the cache.
    "FLASHGEN_CFG_GATE_STEP":
    lambda: float(os.getenv("FLASHGEN_CFG_GATE_STEP", "1.0")),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
