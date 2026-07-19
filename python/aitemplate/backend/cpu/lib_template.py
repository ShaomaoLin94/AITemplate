# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Common CPU backend codegen helpers."""

import jinja2

from aitemplate.backend import registry


VAR_TEMPLATE = jinja2.Template(
    """{{indent}} int64_t {{name}} { {{value}} };"""
)

PTR_TEMPLATE = jinja2.Template(
    """{{indent}} void* {{name}} {nullptr};"""
)


@registry.reg("cpu.lib.var_decl")
def var_decl(name, value=0, indent="  "):
    return VAR_TEMPLATE.render(
        name=name,
        value=value,
        indent=indent,
    )


@registry.reg("cpu.lib.void_ptr_decl")
def void_ptr_decl(name, dtype="float32", indent="  "):
    return PTR_TEMPLATE.render(
        name=name,
        indent=indent,
    )


@registry.reg("cpu.lib.dtype_to_backend_type")
def dtype_to_backend_type(dtype):
    mapping = {
        "float": "float",
        "float32": "float",
        "int": "int32_t",
        "int32": "int32_t",
        "int64": "int64_t",
        "bool": "bool",
    }

    if dtype not in mapping:
        raise NotImplementedError(
            f"CPU - Unsupported dtype: {dtype}"
        )

    return mapping[dtype]
