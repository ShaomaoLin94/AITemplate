import torch

from aitemplate.backend.cpu import CPU
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.model import torch_to_ait_data
from aitemplate.frontend import Tensor


def main():
    batch = 1
    seq = 128
    hidden = 768
    heads = 12
    head_dim = hidden // heads

    m = batch * seq
    n = hidden * 3

    x = Tensor(
        shape=[m, hidden],
        dtype="float32",
        name="x",
        is_input=True,
    )

    weight = Tensor(
        shape=[n, hidden],
        dtype="float32",
        name="weight",
        is_input=True,
    )

    bias = Tensor(
        shape=[n],
        dtype="float32",
        name="bias",
        is_input=True,
    )

    y = ops.gemm_rcr_bias_permute(
        shape=(seq, 3, heads),
        layout="m2n3",
    )(x, weight, bias)

    y._attrs["name"] = "output"
    y._attrs["is_output"] = True

    module = compile_model(
        y,
        CPU(),
        "./tmp",
        "cpu_gemm_rcr_bias_permute_m2n3",
    )

    torch.manual_seed(0)

    x_pt = torch.randn(
        m,
        hidden,
        dtype=torch.float32,
    )

    weight_pt = torch.randn(
        n,
        hidden,
        dtype=torch.float32,
    )

    bias_pt = torch.randn(
        n,
        dtype=torch.float32,
    )

    output = torch.empty(
        3,
        batch,
        heads,
        seq,
        head_dim,
        dtype=torch.float32,
    )

    module.run(
        {
            "x": torch_to_ait_data(x_pt),
            "weight": torch_to_ait_data(weight_pt),
            "bias": torch_to_ait_data(bias_pt),
        },
        {
            "output": torch_to_ait_data(output),
        },
    )

    reference = torch.nn.functional.linear(
        x_pt,
        weight_pt,
        bias_pt,
    )

    reference = reference.reshape(
        batch,
        seq,
        3,
        heads,
        head_dim,
    )

    reference = reference.permute(
        2,
        0,
        3,
        1,
        4,
    ).contiguous()

    max_abs_diff = (
        output - reference
    ).abs().max().item()

    print()
    print("===== CPU gemm_rcr_bias_permute_m2n3 =====")
    print("batch       :", batch)
    print("sequence    :", seq)
    print("hidden      :", hidden)
    print("heads       :", heads)
    print("head dim    :", head_dim)
    print("output shape:", list(output.shape))
    print("max abs diff:", max_abs_diff)

    torch.testing.assert_close(
        output,
        reference,
        rtol=2e-4,
        atol=2e-4,
    )

    print("PASS")


if __name__ == "__main__":
    main()
