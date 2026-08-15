import torch
import torch.nn.functional as F

from aitemplate.backend.cpu import CPU
from aitemplate.compiler import compile_model
from aitemplate.compiler.model import torch_to_ait_data
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.attention import MultiheadAttention
from aitemplate.frontend.nn.linear import Linear


def main():
    batch = 1
    seq = 128
    hidden = 768
    heads = 12
    head_dim = hidden // heads
    scale = head_dim ** -0.5

    # Temporary CPU frontend workaround.
    Linear.USE_CUDA = False
    MultiheadAttention.USE_CUDA = False

    mha = MultiheadAttention(
        dim=hidden,
        batch_size=batch,
        seq_len=seq,
        num_heads=heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        has_residual=True,
        causal=False,
        dtype="float32",
    )

    # Give parameter tensors stable AITemplate names.
    mha.name_parameter_tensor()

    # For this numerical integration test, expose weights as runtime inputs.
    # This avoids mixing attention verification with CPU constant handling.
    parameter_tensors = [
        mha.qkv.weight.tensor(),
        mha.qkv.bias.tensor(),
        mha.proj.weight.tensor(),
        mha.proj.bias.tensor(),
    ]

    for tensor in parameter_tensors:
        tensor._attrs["is_input"] = True

    x = Tensor(
        shape=[batch, seq, hidden],
        dtype="float32",
        name="x",
        is_input=True,
    )

    # Match BERT exactly:
    # self.self(hidden_states, hidden_states)
    y = mha(x, x)

    y._attrs["name"] = "output"
    y._attrs["is_output"] = True

    print("===== MHA graph built =====")
    print("qkv weight :", mha.qkv.weight.tensor()._attrs["name"])
    print("qkv bias   :", mha.qkv.bias.tensor()._attrs["name"])
    print("proj weight:", mha.proj.weight.tensor()._attrs["name"])
    print("proj bias  :", mha.proj.bias.tensor()._attrs["name"])

    module = compile_model(
        y,
        CPU(),
        "./tmp",
        "cpu_multihead_attention",
    )

    torch.manual_seed(0)

    x_pt = torch.randn(
        batch,
        seq,
        hidden,
        dtype=torch.float32,
    )

    # BERT-like initialization scale keeps attention logits reasonable.
    qkv_weight_pt = (
        torch.randn(
            hidden * 3,
            hidden,
            dtype=torch.float32,
        )
        * 0.02
    )

    qkv_bias_pt = (
        torch.randn(
            hidden * 3,
            dtype=torch.float32,
        )
        * 0.02
    )

    proj_weight_pt = (
        torch.randn(
            hidden,
            hidden,
            dtype=torch.float32,
        )
        * 0.02
    )

    proj_bias_pt = (
        torch.randn(
            hidden,
            dtype=torch.float32,
        )
        * 0.02
    )

    output = torch.empty(
        batch,
        seq,
        hidden,
        dtype=torch.float32,
    )

    inputs = {
        "x": torch_to_ait_data(x_pt),
        mha.qkv.weight.tensor()._attrs["name"]:
            torch_to_ait_data(qkv_weight_pt),
        mha.qkv.bias.tensor()._attrs["name"]:
            torch_to_ait_data(qkv_bias_pt),
        mha.proj.weight.tensor()._attrs["name"]:
            torch_to_ait_data(proj_weight_pt),
        mha.proj.bias.tensor()._attrs["name"]:
            torch_to_ait_data(proj_bias_pt),
    }

    module.run(
        inputs,
        {
            "output": torch_to_ait_data(output),
        },
    )

    # ------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------

    x_2d = x_pt.reshape(
        batch * seq,
        hidden,
    )

    qkv = F.linear(
        x_2d,
        qkv_weight_pt,
        qkv_bias_pt,
    )

    # Same m2n3 physical meaning as AITemplate:
    # [B*S, 3*H]
    # -> [B, S, 3, heads, head_dim]
    # -> [3, B, heads, S, head_dim]
    qkv = qkv.reshape(
        batch,
        seq,
        3,
        heads,
        head_dim,
    )

    qkv = qkv.permute(
        2,
        0,
        3,
        1,
        4,
    ).contiguous()

    q = qkv[0].reshape(
        batch * heads,
        seq,
        head_dim,
    )

    k = qkv[1].reshape(
        batch * heads,
        seq,
        head_dim,
    )

    v = qkv[2].reshape(
        batch * heads,
        seq,
        head_dim,
    )

    attention_scores = torch.bmm(
        q,
        k.transpose(1, 2),
    )

    attention_scores *= scale

    attention_probs = torch.softmax(
        attention_scores,
        dim=-1,
    )

    context = torch.bmm(
        attention_probs,
        v,
    )

    context = context.reshape(
        batch,
        heads,
        seq,
        head_dim,
    )

    context = context.permute(
        0,
        2,
        1,
        3,
    ).contiguous()

    context = context.reshape(
        batch * seq,
        hidden,
    )

    reference = F.linear(
        context,
        proj_weight_pt,
        proj_bias_pt,
    )

    # BERT MHA residual.
    reference += x_2d

    reference = reference.reshape(
        batch,
        seq,
        hidden,
    )

    diff = (output - reference).abs()

    print()
    print("===== CPU MultiheadAttention =====")
    print("batch        :", batch)
    print("sequence     :", seq)
    print("hidden       :", hidden)
    print("heads        :", heads)
    print("head dim     :", head_dim)
    print("output shape :", list(output.shape))
    print("max abs diff :", diff.max().item())
    print("mean abs diff:", diff.mean().item())

    torch.testing.assert_close(
        output,
        reference,
        rtol=5e-4,
        atol=5e-4,
    )

    print("PASS")


if __name__ == "__main__":
    main()
