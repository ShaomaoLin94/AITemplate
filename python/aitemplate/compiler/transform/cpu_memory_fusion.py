# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""
CPU memory-planning hook.

Explicit residual/output aliasing is intentionally disabled.

AITemplate's greedy planner already reuses non-overlapping storage.
Forcing gemm_rcr_bias_add output to alias its residual also prevents the
CPU backend from safely writing GEMM results directly into the final output.
"""

from typing import List

from aitemplate.compiler.base import Tensor


def mark_cpu_memory_fusions(
    sorted_graph: List[Tensor],
) -> List[Tensor]:
    return sorted_graph
