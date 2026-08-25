# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Helpers for persistent CPU XNNPACK static-FC caches."""


def cache_id_from_tensor_name(name: str) -> int:
    """Return a deterministic non-zero 64-bit cache id for a tensor name."""

    # FNV-1a 64-bit.  We embed the result as a literal in generated C++,
    # so Python's randomized hash() must not be used here.
    value = 14695981039346656037

    for byte in name.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    return value or 1
