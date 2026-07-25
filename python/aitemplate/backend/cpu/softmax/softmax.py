# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU softmax codegen backed by XNNPACK."""

from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.dtype import normalize_dtype


FUNC_TEMPLATE = jinja2.Template(
    r"""
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <xnnpack.h>

#include "device_functions-generated.h"

namespace {

inline void {{func_name}}_check_xnn_status(
    xnn_status status,
    const char* step) {
  if (status != xnn_status_success) {
    throw std::runtime_error(
        std::string(step) +
        " failed with XNNPACK status " +
        std::to_string(static_cast<int>(status)));
  }
}

struct {{func_name}}_xnn_context {
  xnn_operator_t op = nullptr;

  {{func_name}}_xnn_context() {
    {{func_name}}_check_xnn_status(
        xnn_initialize(nullptr),
        "xnn_initialize");

    {{func_name}}_check_xnn_status(
        xnn_create_softmax_nc_f32(
            0 /* flags */,
            &op),
        "xnn_create_softmax_nc_f32");
  }

  ~{{func_name}}_xnn_context() {
    if (op != nullptr) {
      xnn_delete_operator(op);
    }
  }

  {{func_name}}_xnn_context(
      const {{func_name}}_xnn_context&) = delete;
  {{func_name}}_xnn_context& operator=(
      const {{func_name}}_xnn_context&) = delete;
};

}  // namespace

{{func_signature}}
{
  (void)stream;

  if (batch_size == 0 || channels == 0) {
    return;
  }

  const size_t num_elements = batch_size * channels;
  const size_t extra_elements =
      (XNN_EXTRA_BYTES + sizeof(float) - 1) / sizeof(float);
  const size_t padded_elements =
      num_elements + extra_elements;

  thread_local {{func_name}}_xnn_context context;
  thread_local std::vector<float> input_scratch;
  thread_local std::vector<float> output_scratch;

  input_scratch.resize(padded_elements);
  output_scratch.resize(padded_elements);

// copy input to scratch buffer and pad with zeros to avoid OOB reads
  std::memcpy(
      input_scratch.data(),
      input,
      num_elements * sizeof(float));

  std::fill(
      input_scratch.begin() + num_elements,
      input_scratch.end(),
      0.0f);
  std::fill(
      output_scratch.begin() + num_elements,
      output_scratch.end(),
      0.0f);

  {{func_name}}_check_xnn_status(
      xnn_reshape_softmax_nc_f32(
          context.op,
          channels,
          channels /* input stride */,
          channels /* output stride */,
          batch_size,
          nullptr /* threadpool */),
      "xnn_reshape_softmax_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_setup_softmax_nc_f32(
          context.op,
          input_scratch.data(),
          output_scratch.data()),
      "xnn_setup_softmax_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_run_operator(
          context.op,
          nullptr /* threadpool */),
      "xnn_run_operator");

  std::memcpy(
      output,
      output_scratch.data(),
      num_elements * sizeof(float));
}
"""
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(
    const void* input,
    void* output,
    size_t batch_size,
    size_t channels,
    ait::StreamType stream)
"""
)


FUNC_DECL = jinja2.Template(
    """
{{func_signature}};
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{input}},
{{indent}}    {{output}},
{{indent}}    {{batch_size}},
{{indent}}    {{channels}},
{{indent}}    stream);
"""
)


def _dim_expr(dim) -> str:
    if isinstance(dim, IntImm):
        return str(dim._attrs["values"][0])
    return dim._attrs["name"]


def _validate(func_attrs: Dict[str, Any]) -> None:
    input_tensor = func_attrs["inputs"][0]
    shape = input_tensor._attrs["shape"]
    dim = func_attrs["dim"]

    if dim != len(shape) - 1:   # XNNPACK softmax only supports the last dimension
        raise NotImplementedError(
            "CPU XNNPACK softmax currently supports only the last dimension; "
            f"got dim={dim}, rank={len(shape)}"
        )

    dtype = normalize_dtype(input_tensor._attrs["dtype"])
    if dtype != "float32":
        raise NotImplementedError(
            "CPU XNNPACK softmax currently supports only float32; "
            f"got dtype={input_tensor._attrs['dtype']}"
        )


@registry.reg("cpu.softmax.gen_function")
def gen_function(func_attrs: Dict[str, Any]) -> str:
    _validate(func_attrs)

    func_name = func_attrs["name"]
    return FUNC_TEMPLATE.render(
        func_name=func_name,
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_name,
        ),
    )


@registry.reg("cpu.softmax.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    _validate(func_attrs)

    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
        )
    ).strip()


@registry.reg("cpu.softmax.func_call")
def gen_function_call(
    func_attrs: Dict[str, Any],
    indent="  ",
) -> str:
    _validate(func_attrs)

    input_tensor = func_attrs["inputs"][0]
    output_tensor = func_attrs["outputs"][0]

    shape = input_tensor._attrs["shape"]
    channels = _dim_expr(shape[-1])

    batch_dims = [_dim_expr(dim) for dim in shape[:-1]]
    batch_size = " * ".join(batch_dims) if batch_dims else "1" # last dimension is channels, so batch size is product of all other dimensions

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input=input_tensor._attrs["name"],
        output=output_tensor._attrs["name"],
        batch_size=batch_size,
        channels=channels,
        indent=indent,
    )
