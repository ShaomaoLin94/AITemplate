# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU Q-scaling fusion for BERT attention.

This overlay intentionally does NOT change scratch ownership.

AITemplate already deduplicates the 24 BERT-large attention layers to one
bmm_softmax_bmm_permute generated kernel (and two QKV variants), so sharing
thread_local scratch across generated layers does not provide the previously
assumed memory saving.

The only optimization here is:

  old:
    QKV GEMM -> physical QKV permute
             -> separate Q * scale pass
             -> QK BMM

  new:
    QKV GEMM -> physical QKV permute, scaling Q while it is copied
             -> QK BMM consumes Q directly

Generic bmm callers keep the original scale fallback.
"""

import math

from aitemplate.backend import registry


_QKV_GEN_KEY = "cpu.gemm_rcr_bias_permute_m2n3.gen_function"
_BMM_GEN_KEY = "cpu.bmm_softmax_bmm_permute.gen_function"
_BMM_CALL_KEY = "cpu.bmm_softmax_bmm_permute.func_call"

_BASE_QKV_GEN = registry.get(_QKV_GEN_KEY)
_BASE_BMM_GEN = registry.get(_BMM_GEN_KEY)
_BASE_BMM_CALL = registry.get(_BMM_CALL_KEY)


_VIEW_OPS = {
    "reshape",
    "split",
}


def _op_name(op):
    return str(op._attrs.get("op", ""))


def _inputs(op):
    return list(op._attrs.get("inputs", []))


def _outputs(op):
    return list(op._attrs.get("outputs", []))


def _src_ops(tensor):
    return list(tensor._attrs.get("src_ops", []))


def _dst_ops(tensor):
    return list(tensor._attrs.get("dst_ops", []))


def _is_view_op(op):
    return _op_name(op) in _VIEW_OPS


def _float_literal(value):
    text = repr(float(value))

    if "." not in text and "e" not in text.lower():
        text += ".0"

    return text + "f"


def _static_int(value):
    if hasattr(value, "value"):
        try:
            return int(value.value())
        except Exception:
            pass

    attrs = getattr(value, "_attrs", {})
    values = attrs.get("values")

    if values is not None and len(values) == 1:
        return int(values[0])

    try:
        return int(value)
    except Exception:
        return None


def _find_downstream_attention_scale(tensor):
    """Find one common attention scale through reshape/split views only."""

    queue = [tensor]
    seen_tensors = set()
    seen_ops = set()
    scales = []

    while queue:
        current = queue.pop()

        if id(current) in seen_tensors:
            continue

        seen_tensors.add(id(current))

        for op in _dst_ops(current):
            if id(op) in seen_ops:
                continue

            seen_ops.add(id(op))
            name = _op_name(op)

            if name.startswith(
                "bmm_softmax_bmm_permute"
            ):
                scale = op._attrs.get("scale")

                if scale is None:
                    return None

                scales.append(float(scale))
                continue

            if not _is_view_op(op):
                return None

            queue.extend(_outputs(op))

    if not scales:
        return None

    first = scales[0]

    if not math.isfinite(first):
        return None

    for value in scales[1:]:
        if not math.isclose(
            value,
            first,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return None

    return first


def _qkv_attention_scale(func_attrs):
    """Return compile-time scale only for a 3-way QKV attention permute."""

    shape = func_attrs.get("shape")

    if shape is None or len(shape) != 3:
        return None

    # m2n3 BERT convention:
    #   shape=(sequence, 3, heads)
    if _static_int(shape[1]) != 3:
        return None

    outputs = func_attrs.get("outputs", [])

    if len(outputs) != 1:
        return None

    return _find_downstream_attention_scale(
        outputs[0]
    )


def _find_upstream_qkv(tensor):
    """Find exactly one QKV m2n3 producer through view-only edges."""

    queue = [tensor]
    seen_tensors = set()
    seen_ops = set()
    qkv_ops = []

    while queue:
        current = queue.pop()

        if id(current) in seen_tensors:
            continue

        seen_tensors.add(id(current))

        view_of = current._attrs.get(
            "is_view_of"
        )

        if view_of is not None:
            queue.append(view_of)

        for op in _src_ops(current):
            if id(op) in seen_ops:
                continue

            seen_ops.add(id(op))
            name = _op_name(op)

            if name == "gemm_rcr_bias_permute_m2n3":
                qkv_ops.append(op)
                continue

            if not _is_view_op(op):
                return None

            queue.extend(_inputs(op))

    unique = []
    seen = set()

    for op in qkv_ops:
        if id(op) not in seen:
            seen.add(id(op))
            unique.append(op)

    if len(unique) != 1:
        return None

    return unique[0]


def _bmm_q_is_pre_scaled(func_attrs):
    inputs = func_attrs.get("inputs", [])

    if len(inputs) != 3:
        return False

    qkv_op = _find_upstream_qkv(
        inputs[0]
    )

    if qkv_op is None:
        return False

    qkv_scale = _qkv_attention_scale(
        qkv_op._attrs
    )

    if qkv_scale is None:
        return False

    bmm_scale = float(
        func_attrs.get(
            "scale",
            1.0,
        )
    )

    return math.isclose(
        qkv_scale,
        bmm_scale,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def _replace_once(
    code,
    old,
    new,
    label,
):
    count = code.count(old)

    if count != 1:
        raise RuntimeError(
            "CPU Q-scale fusion: "
            f"{label} expected once, got {count}"
        )

    return code.replace(
        old,
        new,
        1,
    )


def _patch_qkv(
    code,
    scale,
):
    if scale is None:
        return code

    if math.isclose(
        scale,
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        return code

    code = _replace_once(
        code,
        """    size_t m0_size;
    size_t n2_size;
  };
""",
        """    size_t m0_size;
    size_t n2_size;
    float q_scale;
  };
""",
        "QKV permute context field",
    )

    code = _replace_once(
        code,
        """      t3,
      m0_size,
      n2_size};
""",
        """      t3,
      m0_size,
      n2_size,
      """
        + _float_literal(scale)
        + """};
""",
        "QKV permute context init",
    )

    code = _replace_once(
        code,
        """            std::memcpy(
                task_context->output + dst,
                task_context->input + src,
                task_context->n2_size *
                    sizeof(float));
""",
        """            const float* src_ptr =
                task_context->input + src;

            float* dst_ptr =
                task_context->output + dst;

            if (n0 == 0) {
              // AIT_Q_SCALE_FUSED
              //
              // Q is the n0==0 branch in BERT's
              // [3, B, H, S, D] output.
              // Scale while this data is already being copied.
              for (
                  size_t element = 0;
                  element < task_context->n2_size;
                  ++element
              ) {
                dst_ptr[element] =
                    src_ptr[element] *
                    task_context->q_scale;
              }
            } else {
              std::memcpy(
                  dst_ptr,
                  src_ptr,
                  task_context->n2_size *
                      sizeof(float));
            }
""",
        "QKV Q branch copy",
    )

    return code


def _patch_bmm(code):
    """Skip the standalone Q-scale pass when generated call passes scale=1."""

    scale_start = """  // ------------------------------------------------------------
  // Scale Q.
  // Parallelize over complete [K] rows.
  // ------------------------------------------------------------
"""

    scale_end = """  // QK output and softmax output need XNNPACK tail padding because
"""

    start = code.find(scale_start)

    if start < 0:
        raise RuntimeError(
            "CPU Q-scale fusion: "
            "bmm Q-scale block start not found"
        )

    end = code.find(
        scale_end,
        start,
    )

    if end < 0:
        raise RuntimeError(
            "CPU Q-scale fusion: "
            "bmm Q-scale block end not found"
        )

    replacement = """  // ------------------------------------------------------------
  // Q input selection.
  //
  // If scale==1.0f, Q was already scaled by the preceding QKV
  // physical permute. Consume it directly and skip q_scaled.
  //
  // Generic callers with scale!=1 keep the original fallback.
  // ------------------------------------------------------------
  const float* qk_q_ptr =
      q_ptr;

  thread_local std::vector<float> q_scaled;

  if (scale != 1.0f) {
    q_scaled.resize(
        q_elements + extra_elements);

    struct q_scale_context {
      const float* input;
      float* output;
      size_t width;
      float scale;
    };

    q_scale_context scale_context{
        q_ptr,
        q_scaled.data(),
        k_dim,
        scale};

    auto q_scale_task =
        [](void* raw_context, size_t row) {
          auto* task_context =
              static_cast<q_scale_context*>(
                  raw_context);

          const size_t offset =
              row * task_context->width;

          const float* input_row =
              task_context->input + offset;

          float* output_row =
              task_context->output + offset;

          for (
              size_t col = 0;
              col < task_context->width;
              ++col
          ) {
            output_row[col] =
                input_row[col] *
                task_context->scale;
          }
        };

    ait::parallelize_1d(
        q_scale_task,
        &scale_context,
        batch_heads * m);

    std::fill(
        q_scaled.begin() + q_elements,
        q_scaled.end(),
        0.0f);

    qk_q_ptr =
        q_scaled.data();
  }

"""

    code = (
        code[:start]
        + replacement
        + code[end:]
    )

    code = _replace_once(
        code,
        """          q_scaled.data(),
          k_ptr,
          logits.data()),
""",
        """          qk_q_ptr,
          k_ptr,
          logits.data()),
""",
        "QK input pointer",
    )

    return code


def gen_qkv_function(
    func_attrs,
    exec_cond_template=None,
    dim_info_dict=None,
):
    code = _BASE_QKV_GEN(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
    )

    return _patch_qkv(
        code,
        _qkv_attention_scale(
            func_attrs
        ),
    )


def gen_bmm_function(
    func_attrs,
    exec_cond_template=None,
    dim_info_dict=None,
):
    code = _BASE_BMM_GEN(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
    )

    return _patch_bmm(code)


def gen_bmm_call(
    func_attrs,
    indent="  ",
):
    call = _BASE_BMM_CALL(
        func_attrs,
        indent,
    )

    if not _bmm_q_is_pre_scaled(
        func_attrs
    ):
        return call

    old_scale = _float_literal(
        func_attrs.get(
            "scale",
            1.0,
        )
    )

    count = call.count(old_scale)

    if count != 1:
        raise RuntimeError(
            "CPU Q-scale fusion: expected one "
            f"bmm scale literal {old_scale}, got {count}"
        )

    # Do not put a verification marker here: model-generated.h contains
    # both RunImpl and ProfileImpl, which previously made 24 layers look
    # like 48 kernels. Verification belongs in QKV .cpp instead.
    return call.replace(
        old_scale,
        "1.0f",
        1,
    )


registry.BACKEND_FUNCTIONS[
    _QKV_GEN_KEY
] = gen_qkv_function

registry.BACKEND_FUNCTIONS[
    _BMM_GEN_KEY
] = gen_bmm_function

registry.BACKEND_FUNCTIONS[
    _BMM_CALL_KEY
] = gen_bmm_call
