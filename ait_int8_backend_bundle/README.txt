AITemplate CPU INT8 static-FC prototype

Files replaced:
- python/aitemplate/backend/cpu/gemm_universal/static_fc.py
- python/aitemplate/backend/cpu/gemm_universal/gemm_rcr_bias_permute.py
- python/aitemplate/backend/cpu/gemm_universal/gemm_rcr_bias_fast_gelu.py
- python/aitemplate/backend/cpu/gemm_universal/gemm_rcr_bias_add_layernorm_overlay.py

Compile-time toggle:
  AIT_CPU_INT8_FC=1

Default without the variable remains FP32.
