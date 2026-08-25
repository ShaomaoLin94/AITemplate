# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU BERT embeddings + LayerNorm codegen."""

from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.dtype import normalize_dtype


FUNC_TEMPLATE = jinja2.Template(
    r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "device_functions-generated.h"
#include "cpu_threadpool.h"

namespace {

struct {{func_name}}_embedding_context {
  float* output;
  const {{index_type}}* input_ids;
  const {{index_type}}* token_type_ids;
  const {{index_type}}* position_ids;
  const float* word_embeddings;
  const float* token_type_embeddings;
  const float* position_embeddings;
  const float* gamma;
  const float* beta;
  size_t embedding_dim;
  float eps;
};

void {{func_name}}_embedding_row(
    void* raw_context,
    size_t row) {
  auto* context =
      static_cast<{{func_name}}_embedding_context*>(raw_context);

  const size_t word_index =
      static_cast<size_t>(context->input_ids[row]);
  const size_t token_index =
      static_cast<size_t>(context->token_type_ids[row]);
  const size_t position_index =
      static_cast<size_t>(context->position_ids[row]);

  const float* word_row =
      context->word_embeddings + word_index * context->embedding_dim;
  const float* token_row =
      context->token_type_embeddings + token_index * context->embedding_dim;
  const float* position_row =
      context->position_embeddings + position_index * context->embedding_dim;
  float* output_row =
      context->output + row * context->embedding_dim;

  double sum = 0.0;
  double square_sum = 0.0;

  for (size_t col = 0; col < context->embedding_dim; ++col) {
    const float value =
        word_row[col] + token_row[col] + position_row[col];
    output_row[col] = value;

    const double value_d = static_cast<double>(value);
    sum += value_d;
    square_sum += value_d * value_d;
  }

  const double inv_n =
      1.0 / static_cast<double>(context->embedding_dim);
  const double mean = sum * inv_n;
  double variance = square_sum * inv_n - mean * mean;
  variance = std::max(variance, 0.0);

  const float inv_std = static_cast<float>(
      1.0 / std::sqrt(variance + static_cast<double>(context->eps)));
  const float mean_f = static_cast<float>(mean);

  for (size_t col = 0; col < context->embedding_dim; ++col) {
    output_row[col] =
        (output_row[col] - mean_f) * inv_std * context->gamma[col] +
        context->beta[col];
  }
}

}  // namespace

{{func_signature}}
{
  (void)vocab_size;
  (void)type_vocab_size;
  (void)max_position_embeddings;
  (void)stream;

  if (indices_num <= 0 || embedding_dim <= 0) {
    return;
  }

  const {{index_type}}* input_ids_ptr =
      static_cast<const {{index_type}}*>(input_ids);

  const {{index_type}}* token_type_ids_ptr =
      static_cast<const {{index_type}}*>(token_type_ids);

  const {{index_type}}* position_ids_ptr =
      static_cast<const {{index_type}}*>(position_ids);

  {{func_name}}_embedding_context context{
      static_cast<float*>(output),
      input_ids_ptr,
      token_type_ids_ptr,
      position_ids_ptr,
      static_cast<const float*>(word_embeddings),
      static_cast<const float*>(token_type_embeddings),
      static_cast<const float*>(position_embeddings),
      static_cast<const float*>(gamma),
      static_cast<const float*>(beta),
      static_cast<size_t>(embedding_dim),
      eps};

  ait::parallelize_1d(
      {{func_name}}_embedding_row,
      &context,
      static_cast<size_t>(indices_num));
}
"""
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(
    void* output,
    const void* input_ids,
    const void* token_type_ids,
    const void* position_ids,
    const void* word_embeddings,
    const void* token_type_embeddings,
    const void* position_embeddings,
    const void* gamma,
    const void* beta,
    int64_t indices_num,
    int64_t embedding_dim,
    int64_t vocab_size,
    int64_t type_vocab_size,
    int64_t max_position_embeddings,
    float eps,
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
{{indent}}    {{output}},
{{indent}}    {{input_ids}},
{{indent}}    {{token_type_ids}},
{{indent}}    {{position_ids}},
{{indent}}    {{word_embeddings}},
{{indent}}    {{token_type_embeddings}},
{{indent}}    {{position_embeddings}},
{{indent}}    {{gamma}},
{{indent}}    {{beta}},
{{indent}}    {{indices_num}},
{{indent}}    {{embedding_dim}},
{{indent}}    {{vocab_size}},
{{indent}}    {{type_vocab_size}},
{{indent}}    {{max_position_embeddings}},
{{indent}}    {{eps}},
{{indent}}    stream);
"""
)


def _dim_expr(dim) -> str:
    if isinstance(dim, IntImm):
        return str(dim._attrs["values"][0])
    return dim._attrs["name"]


def _index_type(dtype: str) -> str:
    if dtype == "int64":
        return "int64_t"
    if dtype in ("int", "int32"):
        return "int32_t"
    raise NotImplementedError(
        f"CPU BERT embeddings index dtype {dtype} is not supported"
    )


def _validate(func_attrs: Dict[str, Any]) -> None:
    if len(func_attrs["inputs"]) != 8:
        raise NotImplementedError("CPU BERT embeddings requires 8 inputs")

    inputs = func_attrs["inputs"]
    index_dtype = inputs[0]._attrs["dtype"]

    for tensor in inputs[:3]:
        if tensor._attrs["dtype"] != index_dtype:
            raise NotImplementedError("BERT embedding index dtypes must match")

    _index_type(index_dtype)

    for tensor in inputs[3:]:
        if normalize_dtype(tensor._attrs["dtype"]) != "float32":
            raise NotImplementedError(
                "CPU BERT embeddings currently supports float32 embeddings only"
            )


@registry.reg("cpu.bert_embeddings.gen_function")
def gen_function(func_attrs: Dict[str, Any]) -> str:
    _validate(func_attrs)
    index_type = _index_type(func_attrs["inputs"][0]._attrs["dtype"])
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=index_type,
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], index_type=index_type
        ),
    )


@registry.reg("cpu.bert_embeddings.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    _validate(func_attrs)
    index_type = _index_type(func_attrs["inputs"][0]._attrs["dtype"])
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], index_type=index_type
        )
    ).strip()


@registry.reg("cpu.bert_embeddings.func_call")
def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    _validate(func_attrs)

    (
        input_ids,
        token_type_ids,
        position_ids,
        word_embeddings,
        token_type_embeddings,
        position_embeddings,
        gamma,
        beta,
    ) = func_attrs["inputs"]

    output = func_attrs["outputs"][0]
    indices_num = " * ".join(
        _dim_expr(dim) for dim in input_ids._attrs["shape"]
    )
    embedding_dim = _dim_expr(word_embeddings._attrs["shape"][1])
    vocab_size = _dim_expr(word_embeddings._attrs["shape"][0])
    type_vocab_size = _dim_expr(token_type_embeddings._attrs["shape"][0])
    max_position_embeddings = _dim_expr(position_embeddings._attrs["shape"][0])

    eps = repr(float(func_attrs["eps"]))
    if "." not in eps and "e" not in eps.lower():
        eps += ".0"
    eps += "f"

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output._attrs["name"],
        input_ids=input_ids._attrs["name"],
        token_type_ids=token_type_ids._attrs["name"],
        position_ids=position_ids._attrs["name"],
        word_embeddings=word_embeddings._attrs["name"],
        token_type_embeddings=token_type_embeddings._attrs["name"],
        position_embeddings=position_embeddings._attrs["name"],
        gamma=gamma._attrs["name"],
        beta=beta._attrs["name"],
        indices_num=indices_num,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        eps=eps,
        indent=indent,
    )
