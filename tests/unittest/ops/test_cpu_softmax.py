# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Tests for the XNNPACK-backed CPU softmax operator."""

import unittest

import torch

from aitemplate.backend.cpu.target_def import CPU
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.model import torch_to_ait_data
from aitemplate.frontend import Tensor


class CPUSoftmaxTestCase(unittest.TestCase):
    def test_fp32_last_dim(self):
        shape = [2, 12, 8, 8]

        X = Tensor(
            shape=shape,
            dtype="float32",
            name="X",
            is_input=True,
        )

        Y = ops.softmax()(X, dim=-1)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        with compile_model(
            Y,
            CPU(),
            "./tmp",
            "cpu_softmax_fp32_last_dim",
        ) as module:
            torch.manual_seed(0)

            x = torch.randn(
                shape,
                dtype=torch.float32,
                device="cpu",
            ).contiguous()

            y = torch.empty_like(x)
            y_ref = torch.softmax(x, dim=-1)

            module.run(
                {"X": torch_to_ait_data(x)},
                {"Y": torch_to_ait_data(y)},
            )

            torch.testing.assert_close(
                y,
                y_ref,
                atol=1e-5,
                rtol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
