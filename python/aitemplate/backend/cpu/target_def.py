# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""
CPU target specialization.

The initial target is intentionally minimal.  It establishes the CPU backend
inside AITemplate without depending on CUDA, ROCm, or TVM.  Operator codegen
and XNNPACK integration are added separately.
"""

import os
import platform
import shutil
from typing import List

from aitemplate.backend import registry
from aitemplate.backend.target import AIT_STATIC_FILES_PATH, Target, TargetType


class CPU(Target):
    """AITemplate CPU target."""

    def __init__(
        self,
        arch=None,
        ait_static_files_path=AIT_STATIC_FILES_PATH,
        **kwargs,
    ):
        super().__init__(ait_static_files_path)

        self._target_type = TargetType.cpu
        self._arch = (arch or platform.machine()).lower()
        self._kwargs = kwargs

        # CPU backend starts without AITemplate's GPU kernel profiler.
        self._operators = {}

        self._compile_options = self._build_compile_options()

    def _build_compile_options(self) -> str:
        options = [
            "-O3",
            "-fPIC",
            "-fvisibility=hidden",
            "-std=c++17",
            "-pthread",
        ]

        if self._ndebug == 1:
            options.append("-DNDEBUG")

        return " ".join(options)

    def _load_profile_cache(self):
        """CPU backend does not use the GPU-style profiler cache initially."""
        self._cache_path = None
        self._profile_cache = None

    def cc(self):
        """Return the host C++ compiler."""
        return os.environ.get("CXX", shutil.which("g++") or "g++")

    def compile_cmd(self, executable=False):
        """Return the command used to compile CPU generated sources."""
        if executable:
            return (
                self.cc()
                + " "
                + self._compile_options
                + " -o {target} {src}"
            )

        return (
            self.cc()
            + " "
            + self._compile_options
            + " -c -o {target} {src}"
        )

    def compile_options(self):
        return self._compile_options

    def src_extension(self):
        return ".cpp"

    def dev_select_flag(self):
        # CPU has no GPU-style device-selection environment variable.
        return ""

    def select_minimal_algo(self, algo_names: List[str]):
        if not algo_names:
            return None
        return min(algo_names)

    def get_include_directories(self) -> List[str]:
        return [
            os.path.join(self.static_files_path, "include"),
        ]

    def copy_headers_and_csrc_to_workdir(self, workdir: str) -> List[str]:
        """Copy shared runtime files needed by the CPU backend."""
        sources = super().copy_headers_and_csrc_to_workdir(workdir)

        # debug_utility.cpp currently contains CUDA kernels and CUDA
        # kernel-launch syntax. CPU debug utilities will be implemented
        # separately.
        debug_source = f"debug_utility{self.src_extension()}"
        return [
            source
            for source in sources
            if os.path.basename(source) != debug_source
        ]

    def get_host_compiler_options(self) -> List[str]:
        options = [
            "-O3",
            "-fPIC",
            "-fvisibility=hidden",
            "-std=c++17",
            "-pthread",
        ]

        if self._ndebug == 1:
            options.append("-DNDEBUG")

        return options

    def get_device_compiler_options(self) -> List[str]:
        # CPU has no separate device compiler.
        return []


@registry.reg("cpu.create_target")
def create_target(arch=None, **kwargs):
    return CPU(arch=arch, **kwargs)
