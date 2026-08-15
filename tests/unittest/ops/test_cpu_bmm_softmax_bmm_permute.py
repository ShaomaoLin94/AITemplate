import torch

from aitemplate.backend.cpu import CPU
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.model import torch_to_ait_data
from aitemplate.frontend import Tensor


def main():
    batch = 1
    heads = 12
    seq = 128
    head_dim = 64

    batch_heads = batch * heads
    scale = head_dim ** -0.5

    q = Tensor(
        shape=[batch_heads, seq, head_dim],
        dtype="float32",
        name="q",
        is_input=True,
    )
    k = Tensor(
        shape=[batch_heads, seq, head_dim],
        dtype="float32",
        name="k",
        is_input=True,
    )
    v = Tensor(
        shape=[batch_heads, seq, head_dim],
        dtype="float32",
        name="v",
        is_input=True,
    )

    y = ops.bmm_softmax_bmm_permute(
        shape=(heads,),
        scale=scale,
    )(q, k, v)

    y._attrs["name"] = "output"
    y._attrs["is_output"] = True

    module = compile_model(
        y,
        CPU(),
        "./tmp",
        "cpu_bmm_softmax_bmm_permute",
    )

    torch.manual_seed(0)

    q_pt = torch.randn(
        batch_heads,
        seq,
        head_dim,
        dtype=torch.float32,
    )
    k_pt = torch.randn(
        batch_heads,
        seq,
        head_dim,
        dtype=torch.float32,
    )
    v_pt = torch.randn(
        batch_heads,
        seq,
        head_dim,
        dtype=torch.float32,
    )

    output = torch.empty(
        batch,
        seq,
        heads,
        head_dim,
        dtype=torch.float32,
    )

    module.run(
        {
            "q": torch_to_ait_data(q_pt),
            "k": torch_to_ait_data(k_pt),
            "v": torch_to_ait_data(v_pt),
        },
        {
            "output": torch_to_ait_data(output),
        },
    )

    reference = torch.bmm(
        q_pt,
        k_pt.transpose(1, 2),
    ) * scale

    reference = torch.softmax(reference, dim=-1)

    reference = torch.bmm(
        reference,
        v_pt,
    )

    reference = (
        reference
        .reshape(batch, heads, seq, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    max_abs_diff = (
        output - reference
    ).abs().max().item()

    print()
    print("===== CPU bmm_softmax_bmm_permute =====")
    print("batch       :", batch)
    print("sequence    :", seq)
    print("heads       :", heads)
    print("head dim    :", head_dim)
    print("scale       :", scale)
    print("output shape:", list(output.shape))
    print("max abs diff:", max_abs_diff)

    torch.testing.assert_close(
        output,
        reference,
        rtol=2e-4,
        atol=2e-5,
    )

    print("PASS")


if __name__ == "__main__":
    main()
