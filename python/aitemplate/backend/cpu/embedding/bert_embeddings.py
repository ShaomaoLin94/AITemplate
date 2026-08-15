# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU BERT embeddings codegen."""

from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.compiler.dtype import normalize_dtype


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


FUNC_TEMPLATE = jinja2.Template(
    r"""
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "device_functions-generated.h"

{{func_signature}}
{
  (void)stream;

  if (indices_num == 0 || embedding_dim == 0) {
    return;
  }

  float* output_ptr =
      static_cast<float*>(output);

  const {{index_type}}* input_ids_ptr =
      static_cast<const {{index_type}}*>(input_ids);

  const {{index_type}}* token_type_ids_ptr =
      static_cast<const {{index_type}}*>(token_type_ids);

  const {{index_type}}* position_ids_ptr =
      static_cast<const {{index_type}}*>(position_ids);

  const float* word_ptr =
      static_cast<const float*>(word_embeddings);

  const float* token_type_ptr =
      static_cast<const float*>(token_type_embeddings);

  const float* position_ptr =
      static_cast<const float*>(position_embeddings);

  const float* gamma_ptr =
      static_cast<const float*>(gamma);

  const float* beta_ptr =
      static_cast<const float*>(beta);

  thread_local std::vector<float> scratch;
  scratch.resize(static_cast<size_t>(embedding_dim));

  for (int64_t index = 0; index < indices_num; ++index) {
    const int64_t word_id =
        static_cast<int64_t>(input_ids_ptr[index]);

    const int64_t token_type_id =
        static_cast<int64_t>(token_type_ids_ptr[index]);

    const int64_t position_id =
        static_cast<int64_t>(position_ids_ptr[index]);

    if (word_id < 0 || word_id >= vocab_size) {
      throw std::runtime_error(
          "CPU bert_embeddings: word id out of range");
    }

    if (
        token_type_id < 0 ||
        token_type_id >= type_vocab_size
    ) {
      throw std::runtime_error(
          "CPU bert_embeddings: token type id out of range");
    }

    if (
        position_id < 0 ||
        position_id >= max_position_embeddings
    ) {
      throw std::runtime_error(
          "CPU bert_embeddings: position id out of range");
    }

    const size_t word_offset =
        static_cast<size_t>(word_id) *
        static_cast<size_t>(embedding_dim);

    const size_t token_type_offset =
        static_cast<size_t>(token_type_id) *
        static_cast<size_t>(embedding_dim);

    const size_t position_offset =
        static_cast<size_t>(position_id) *
        static_cast<size_t>(embedding_dim);

    float sum = 0.0f;

    // word + token_type + position
    for (int64_t c = 0; c < embedding_dim; ++c) {
      const float value =
          word_ptr[word_offset + c] +
          token_type_ptr[token_type_offset + c] +
          position_ptr[position_offset + c];

      scratch[c] = value;
      sum += value;
    }

    const float mean =
        sum / static_cast<float>(embedding_dim);

    float variance_sum = 0.0f;

    for (int64_t c = 0; c < embedding_dim; ++c) {
      const float diff = scratch[c] - mean;
      variance_sum += diff * diff;
    }

    const float variance =
        variance_sum /
        static_cast<float>(embedding_dim);

    const float inv_std =
        1.0f / std::sqrt(variance + eps);

    const size_t output_offset =
        static_cast<size_t>(index) *
        static_cast<size_t>(embedding_dim);

    // LayerNorm + affine gamma/beta.
    for (int64_t c = 0; c < embedding_dim; ++c) {
      output_ptr[output_offset + c] =
          (scratch[c] - mean) *
              inv_std *
              gamma_ptr[c] +
          beta_ptr[c];
    }
  }
}
"""
)


FUNC_DECL = jinja2.Template(
    """
{{func_signature}};
"""
)


FUNC_CALL = jinja2.Template(
    """
{{indent}}{
{{indent}}  int64_t indices_num = 1;
{% for dim in index_dims %}
{{indent}}  indices_num *= {{dim}};
{% endfor %}
{{indent}}  {{func_name}}(
{{indent}}      {{output}},
{{indent}}      {{input_ids}},
{{indent}}      {{token_type_ids}},
{{indent}}      {{position_ids}},
{{indent}}      {{word_embeddings}},
{{indent}}      {{token_type_embeddings}},
{{indent}}      {{position_embeddings}},
{{indent}}      {{gamma}},
{{indent}}      {{beta}},
{{indent}}      indices_num,
{{indent}}      {{embedding_dim}},
{{indent}}      {{vocab_size}},
{{indent}}      {{type_vocab_size}},
{{indent}}      {{max_position_embeddings}},
{{indent}}      {{eps}},
{{indent}}      stream);
{{indent}}}
"""
)


def _static_dim(dim) -> int:
    values = dim._attrs.get("values")

    if values is None or len(values) != 1:
        raise NotImplementedError(
            "CPU bert_embeddings currently requires "
            "static embedding-table dimensions"
        )

    return int(values[0])


def _index_type(dtype: str) -> str:
    if dtype == "int64":
        return "int64_t"

    if dtype in ("int", "int32"):
        return "int32_t"

    raise NotImplementedError(
        f"CPU bert_embeddings unsupported index dtype: {dtype}"
    )


def _float_literal(value) -> str:
    text = repr(float(value))

    if "." not in text and "e" not in text.lower():
        text += ".0"

    return text + "f"


def _validate(func_attrs: Dict[str, Any]) -> None:
    if len(func_attrs["inputs"]) != 8:
        raise RuntimeError(
            "CPU bert_embeddings expects 8 inputs"
        )

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

    index_dtype = input_ids._attrs["dtype"]

    if (
        token_type_ids._attrs["dtype"] != index_dtype
        or position_ids._attrs["dtype"] != index_dtype
    ):
        raise NotImplementedError(
            "CPU bert_embeddings requires all index "
            "tensors to use the same dtype"
        )

    _index_type(index_dtype)

    for name, tensor in (
        ("word_embeddings", word_embeddings),
        ("token_type_embeddings", token_type_embeddings),
        ("position_embeddings", position_embeddings),
        ("gamma", gamma),
        ("beta", beta),
    ):
        dtype = normalize_dtype(tensor._attrs["dtype"])

        if dtype != "float32":
            raise NotImplementedError(
                "CPU bert_embeddings currently supports "
                f"float32 only; {name} has dtype "
                f"{tensor._attrs['dtype']}"
            )

    if len(word_embeddings._attrs["shape"]) != 2:
        raise NotImplementedError(
            "word embeddings must be rank 2"
        )

    if len(token_type_embeddings._attrs["shape"]) != 2:
        raise NotImplementedError(
            "token type embeddings must be rank 2"
        )

    if len(position_embeddings._attrs["shape"]) != 2:
        raise NotImplementedError(
            "position embeddings must be rank 2"
        )

    embedding_dim = _static_dim(
        word_embeddings._attrs["shape"][1]
    )

    if (
        _static_dim(token_type_embeddings._attrs["shape"][1])
        != embedding_dim
        or _static_dim(position_embeddings._attrs["shape"][1])
        != embedding_dim
    ):
        raise NotImplementedError(
            "all embedding tables must have the same "
            "embedding dimension"
        )

    if (
        len(gamma._attrs["shape"]) != 1
        or len(beta._attrs["shape"]) != 1
        or _static_dim(gamma._attrs["shape"][0])
        != embedding_dim
        or _static_dim(beta._attrs["shape"][0])
        != embedding_dim
    ):
        raise NotImplementedError(
            "gamma/beta must match embedding dimension"
        )


@registry.reg("cpu.bert_embeddings.gen_function")
def gen_function(
    func_attrs: Dict[str, Any],
) -> str:
    _validate(func_attrs)

    index_dtype = func_attrs["inputs"][0]._attrs["dtype"]

    return FUNC_TEMPLATE.render(
        index_type=_index_type(index_dtype),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
        ).strip(),
    )


@registry.reg("cpu.bert_embeddings.func_decl")
def gen_function_decl(
    func_attrs: Dict[str, Any],
) -> str:
    _validate(func_attrs)

    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
        ).strip(),
    ).strip()


@registry.reg("cpu.bert_embeddings.func_call")
def gen_function_call(
    func_attrs: Dict[str, Any],
    indent="  ",
) -> str:
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

    word_shape = word_embeddings._attrs["shape"]
    token_type_shape = token_type_embeddings._attrs["shape"]
    position_shape = position_embeddings._attrs["shape"]

    return FUNC_CALL.render(
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
        index_dims=[
            dim._attrs["name"]
            for dim in input_ids._attrs["shape"]
        ],
        embedding_dim=str(_static_dim(word_shape[1])),
        vocab_size=str(_static_dim(word_shape[0])),
        type_vocab_size=str(
            _static_dim(token_type_shape[0])
        ),
        max_position_embeddings=str(
            _static_dim(position_shape[0])
        ),
        eps=_float_literal(func_attrs["eps"]),
        indent=indent,
    )
