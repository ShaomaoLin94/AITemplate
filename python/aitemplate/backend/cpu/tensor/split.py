# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU split implementation."""

import jinja2

from aitemplate.backend import registry
from aitemplate.compiler.dtype import normalize_dtype


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(
    void* outputs[],
    int64_t** output_shapes[],
    const bool output_masks[],
    const void* input,
    const int64_t* input_shape,
    int64_t real_num_splits,
    int64_t all_num_splits,
    int64_t split_sizes[],
    int64_t split_dim,
    int64_t rank,
    ait::StreamType stream)
"""
)


FUNC_TEMPLATE = jinja2.Template(
    r"""
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "device_functions-generated.h"

{{func_signature}}
{
  (void)stream;

  if (rank <= 0) {
    throw std::runtime_error(
        "CPU split: rank must be positive");
  }

  if (split_dim < 0 || split_dim >= rank) {
    throw std::runtime_error(
        "CPU split: invalid split dimension");
  }

  if (all_num_splits <= 0) {
    throw std::runtime_error(
        "CPU split: no split sections");
  }

  // If every split output was fused away by AITemplate,
  // there is nothing to materialize.
  if (real_num_splits == 0) {
    return;
  }

  if (input == nullptr) {
    throw std::runtime_error(
        "CPU split: input is null");
  }

  // Validate split sizes.
  int64_t total_split_size = 0;

  for (int64_t i = 0; i < all_num_splits; ++i) {
    if (split_sizes[i] < 0) {
      throw std::runtime_error(
          "CPU split: negative split size");
    }

    total_split_size += split_sizes[i];
  }

  if (total_split_size != input_shape[split_dim]) {
    throw std::runtime_error(
        "CPU split: split sizes do not match input shape");
  }

  // Update logical output shapes.
  int64_t real_idx = 0;

  for (int64_t i = 0; i < all_num_splits; ++i) {
    if (!output_masks[i]) {
      continue;
    }

    if (real_idx >= real_num_splits) {
      throw std::runtime_error(
          "CPU split: inconsistent output mask");
    }

    int64_t** shape = output_shapes[real_idx];

    for (int64_t d = 0; d < rank; ++d) {
      *(shape[d]) = input_shape[d];
    }

    *(shape[split_dim]) = split_sizes[i];

    ++real_idx;
  }

  // Row-major contiguous tensor:
  //
  // [outer, split_dim, inner]
  //
  // Splitting along split_dim means each outer block contains
  // one contiguous region for each output.
  int64_t outer = 1;

  for (int64_t d = 0; d < split_dim; ++d) {
    outer *= input_shape[d];
  }

  int64_t inner = 1;

  for (int64_t d = split_dim + 1; d < rank; ++d) {
    inner *= input_shape[d];
  }

  const char* input_bytes =
      static_cast<const char*>(input);

  constexpr size_t element_size = sizeof(float);

  int64_t split_offset = 0;
  real_idx = 0;

  for (int64_t split_idx = 0;
       split_idx < all_num_splits;
       ++split_idx) {
    const int64_t split_size =
        split_sizes[split_idx];

    if (output_masks[split_idx]) {
      if (outputs[real_idx] == nullptr &&
          split_size != 0) {
        throw std::runtime_error(
            "CPU split: output is null");
      }

      char* output_bytes =
          static_cast<char*>(outputs[real_idx]);

      const size_t copy_elements =
          static_cast<size_t>(split_size) *
          static_cast<size_t>(inner);

      const size_t copy_bytes =
          copy_elements * element_size;

      for (int64_t outer_idx = 0;
           outer_idx < outer;
           ++outer_idx) {
        const int64_t src_element_offset =
            (
                outer_idx * input_shape[split_dim]
                + split_offset
            ) * inner;

        const int64_t dst_element_offset =
            outer_idx * split_size * inner;

        std::memcpy(
            output_bytes
                + static_cast<size_t>(dst_element_offset)
                    * element_size,
            input_bytes
                + static_cast<size_t>(src_element_offset)
                    * element_size,
            copy_bytes);
      }

      ++real_idx;
    }

    split_offset += split_size;
  }
}
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
{{func_signature}};
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}  void* split_outputs[] = {
{{indent}}    {{outputs}}
{{indent}}  };

{{output_shape_defs}}
{{indent}}  int64_t** split_output_shapes[] = {
{{indent}}    {{output_shapes}}
{{indent}}  };

{{indent}}  const int64_t split_input_shape[] = {
{{indent}}    {{input_dims}}
{{indent}}  };

{{indent}}  int64_t split_sizes[] = {
{{indent}}    {{split_sizes}}
{{indent}}  };

{{indent}}  const bool split_output_masks[] = {
{{indent}}    {{output_masks}}
{{indent}}  };

{{indent}}  {{func_name}}(
{{indent}}      split_outputs,
{{indent}}      split_output_shapes,
{{indent}}      split_output_masks,
{{indent}}      {{input}},
{{indent}}      split_input_shape,
{{indent}}      {{real_num_splits}},
{{indent}}      {{all_num_splits}},
{{indent}}      split_sizes,
{{indent}}      {{split_dim}},
{{indent}}      {{rank}},
{{indent}}      stream);
{{indent}}}
"""
)


OUTPUT_SHAPE_TEMPLATE = jinja2.Template(
    """
{{indent}}int64_t* {{name}}[] = {
{{indent}}  {{dims}}
{{indent}}};
"""
)


def _validate(func_attrs):
    x = func_attrs["inputs"][0]

    dtype = normalize_dtype(x._attrs["dtype"])

    if dtype != "float32":
        raise NotImplementedError(
            "CPU split currently supports only float32"
        )

    if len(func_attrs["split_sizes"]) != len(
        func_attrs["output_masks"]
    ):
        raise RuntimeError(
            "CPU split: split_sizes/output_masks mismatch"
        )


@registry.reg("cpu.split.gen_function")
def gen_function(func_attrs):
    _validate(func_attrs)

    func_name = func_attrs["name"]

    return FUNC_TEMPLATE.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_name,
        )
    )


@registry.reg("cpu.split.func_decl")
def gen_function_decl(func_attrs):
    _validate(func_attrs)

    return FUNC_DECL_TEMPLATE.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
        )
    ).strip()


@registry.reg("cpu.split.func_call")
def gen_function_call(func_attrs, indent="  "):
    _validate(func_attrs)

    x = func_attrs["inputs"][0]
    outputs = func_attrs["outputs"]

    output_names = []
    output_shape_defs = []
    output_shape_names = []

    for idx, output in enumerate(outputs):
        output_names.append(
            output._attrs["name"]
        )

        shape_name = (
            f"{output._attrs['name']}_split_shape_{idx}"
        )

        dim_refs = ", ".join(
            "&" + dim._attrs["name"]
            for dim in output._attrs["shape"]
        )

        output_shape_defs.append(
            OUTPUT_SHAPE_TEMPLATE.render(
                indent=indent + "  ",
                name=shape_name,
                dims=dim_refs,
            )
        )

        output_shape_names.append(shape_name)

    # Be robust if all split outputs have been fused away.
    if not output_names:
        output_names = ["nullptr"]
        output_shape_names = ["nullptr"]

    input_dims = ", ".join(
        dim._attrs["name"]
        for dim in x._attrs["shape"]
    )

    split_sizes = ", ".join(
        str(size)
        for size in func_attrs["split_sizes"]
    )

    output_masks = ", ".join(
        "true" if mask else "false"
        for mask in func_attrs["output_masks"]
    )

    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        outputs=",\n{}    ".format(indent).join(
            output_names
        ),
        output_shape_defs="".join(
            output_shape_defs
        ),
        output_shapes=", ".join(
            output_shape_names
        ),
        input_dims=input_dims,
        split_sizes=split_sizes,
        output_masks=output_masks,
        func_name=func_attrs["name"],
        input=x._attrs["name"],
        real_num_splits=len(outputs),
        all_num_splits=len(
            func_attrs["output_masks"]
        ),
        split_dim=func_attrs["split_dim"],
        rank=len(x._attrs["shape"]),
    )
