import torch
import torch.nn.functional as F

import aitemplate.backend.cpu
from aitemplate.backend.cpu.target_def import CPU
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor


def main():
    torch.manual_seed(0)

    # BERT-base FFN dimensions:
    # batch * sequence = 1 * 128 = 128
    m = 128
    hidden = 768
    intermediate = 3072
    eps = 1e-5
    dtype = "float32"

    print("===== Build AITemplate FFN graph =====")
    print(f"X: [{m}, {hidden}]")
    print(f"Intermediate: [{m}, {intermediate}]")
    print(f"Output: [{m}, {hidden}]")

    # Runtime input
    X = Tensor(
        shape=[m, hidden],
        dtype=dtype,
        name="X",
        is_input=True,
    )

    # First Linear: 768 -> 3072
    W1 = Tensor(
        shape=[intermediate, hidden],
        dtype=dtype,
        name="W1",
        is_input=True,
    )
    B1 = Tensor(
        shape=[intermediate],
        dtype=dtype,
        name="B1",
        is_input=True,
    )

    # Second Linear: 3072 -> 768
    W2 = Tensor(
        shape=[hidden, intermediate],
        dtype=dtype,
        name="W2",
        is_input=True,
    )
    B2 = Tensor(
        shape=[hidden],
        dtype=dtype,
        name="B2",
        is_input=True,
    )

    # LayerNorm parameters
    gamma = Tensor(
        shape=[hidden],
        dtype=dtype,
        name="gamma",
        is_input=True,
    )
    beta = Tensor(
        shape=[hidden],
        dtype=dtype,
        name="beta",
        is_input=True,
    )

    # FFN:
    #
    # X
    #  -> Linear + bias + FastGELU
    #  -> Linear + bias + residual(X)
    #  -> LayerNorm

    H = ops.gemm_rcr_bias_fast_gelu()(X, W1, B1)

    R = ops.gemm_rcr_bias_add()(H, W2, B2, X)

    Y = ops.layernorm()(
        R,
        gamma,
        beta,
        [hidden],
        eps,
    )

    Y._attrs["name"] = "Y"
    Y._attrs["is_output"] = True

    print("\n===== Compile =====")

    module = compile_model(
        Y,
        CPU(),
        "./tmp",
        "cpu_bert_ffn",
    )

    print("compile: PASS")

    # Print actual operators that survived into the compiled graph.
    op_names = []
    for tensor in module.debug_sorted_graph:
        for op in tensor._attrs["src_ops"]:
            op_names.append(op._attrs["op"])

    op_names = sorted(set(op_names))

    print("\n===== Compiled operators =====")
    for name in op_names:
        print(name)

    expected_ops = {
        "gemm_rcr_bias_fast_gelu",
        "gemm_rcr_bias_add",
        "layernorm",
    }

    missing = expected_ops - set(op_names)
    if missing:
        raise RuntimeError(
            f"Missing expected FFN operators: {sorted(missing)}"
        )

    # -------------------------
    # PyTorch test data
    # -------------------------

    x = torch.randn(m, hidden, dtype=torch.float32)

    w1 = torch.randn(
        intermediate,
        hidden,
        dtype=torch.float32,
    ) * 0.02

    b1 = torch.randn(
        intermediate,
        dtype=torch.float32,
    ) * 0.02

    w2 = torch.randn(
        hidden,
        intermediate,
        dtype=torch.float32,
    ) * 0.02

    b2 = torch.randn(
        hidden,
        dtype=torch.float32,
    ) * 0.02

    gamma_pt = torch.randn(
        hidden,
        dtype=torch.float32,
    ) * 0.02 + 1.0

    beta_pt = torch.randn(
        hidden,
        dtype=torch.float32,
    ) * 0.02

    # -------------------------
    # PyTorch reference
    # -------------------------

    ref = F.linear(x, w1, b1)

    # AITemplate fast_gelu uses the tanh GELU approximation.
    ref = F.gelu(ref, approximate="tanh")

    ref = F.linear(ref, w2, b2)
    ref = ref + x

    ref = F.layer_norm(
        ref,
        (hidden,),
        gamma_pt,
        beta_pt,
        eps,
    )

    # -------------------------
    # AITemplate execution
    # -------------------------

    output = torch.empty(
        m,
        hidden,
        dtype=torch.float32,
    )

    # AITemplate's run_with_tensors() currently assumes CUDA tensors.
    # For the CPU backend, pass the host pointers directly through AITData.
    from aitemplate.compiler.model import torch_to_ait_data

    module.run(
        {
            "X": torch_to_ait_data(x),
            "W1": torch_to_ait_data(w1),
            "B1": torch_to_ait_data(b1),
            "W2": torch_to_ait_data(w2),
            "B2": torch_to_ait_data(b2),
            "gamma": torch_to_ait_data(gamma_pt),
            "beta": torch_to_ait_data(beta_pt),
        },
        {
            "Y": torch_to_ait_data(output),
        },
    )
    # -------------------------
    # Numerical comparison
    # -------------------------

    diff = (output - ref).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    close = torch.allclose(
        output,
        ref,
        atol=3e-3,
        rtol=3e-3,
    )

    print("\n===== Numerical result =====")
    print("max abs diff :", max_diff)
    print("mean abs diff:", mean_diff)
    print("allclose     :", close)

    if not close:
        idx = diff.argmax().item()
        row = idx // hidden
        col = idx % hidden

        print("\nLargest difference:")
        print("index:", (row, col))
        print("AIT :", output[row, col].item())
        print("PT  :", ref[row, col].item())

        raise RuntimeError("CPU BERT FFN numerical test FAILED")

    print("\n========================================")
    print("CPU BERT FFN integration test: PASS")
    print("========================================")


if __name__ == "__main__":
    main()
