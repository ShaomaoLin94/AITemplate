# flake8: noqa
from aitemplate.backend.cpu.gemm_universal.gemm_rcr_bias import *
from aitemplate.backend.cpu.gemm_universal.gemm_rcr_bias_add import *
from aitemplate.backend.cpu.gemm_universal.gemm_rcr_bias_fast_gelu import *
from aitemplate.backend.cpu.gemm_universal import bmm_softmax_bmm_permute
from aitemplate.backend.cpu.gemm_universal import gemm_rcr_bias_permute

# CPU fused residual + LayerNorm overlay.
from aitemplate.backend.cpu.gemm_universal import (
    gemm_rcr_bias_add_layernorm_overlay,
)

# CPU fused residual + LayerNorm overlay.
from aitemplate.backend.cpu.gemm_universal import (
    gemm_rcr_bias_add_layernorm_overlay,
)


# CPU attention optimization:
# shared layer scratch + Q scaling folded into the QKV physical permute.
from aitemplate.backend.cpu.gemm_universal import attention_memory_qscale_overlay
