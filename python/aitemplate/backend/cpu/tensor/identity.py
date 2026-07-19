# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU identity function codegen."""

from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.dtype import get_dtype_size


FUNC_TEMPLATE = jinja2.Template(
    """
#include <cstddef>
#include <cstring>

#include "device_functions-generated.h"

{{func_signature}}
{
    (void)stream;
{% if is_copy %}
    if (size != 0 && *output != input) {
        std::memcpy(*output, input, size);
    }
{% else %}
    *output = input;
{% endif %}
}
"""
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(
    void** output,
    void* input,
    size_t size,
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
{{indent}}    &{{output}},
{{indent}}    {{input}},
{{indent}}    {{size}},
{{indent}}    stream);
"""
)


@registry.reg("cpu.identity.gen_function")
def gen_function(func_attrs: Dict[str, Any]) -> str:
    is_copy = func_attrs["outputs"][0]._attrs["is_output"]

    return FUNC_TEMPLATE.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"]
        ),
        is_copy=is_copy,
    )


@registry.reg("cpu.identity.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"]
        )
    ).strip()


@registry.reg("cpu.identity.func_call")
def gen_function_call(
    func_attrs: Dict[str, Any],
    indent="  ",
) -> str:
    assert len(func_attrs["inputs"]) == 1
    assert len(func_attrs["outputs"]) == 1

    input_name = func_attrs["inputs"][0]._attrs["name"]

    output = func_attrs["outputs"][0]
    output_name = output._attrs["name"]

    shape = ["1"]
    for dim in output._attrs["shape"]:
        if isinstance(dim, IntImm):
            shape.append(str(dim._attrs["values"][0]))
        else:
            shape.append(dim._attrs["name"])

    size = (
        "*".join(shape)
        + f" * {get_dtype_size(output._attrs['dtype'])}"
    )

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        input=input_name,
        size=size,
        indent=indent,
    )
