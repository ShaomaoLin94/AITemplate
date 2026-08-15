import torch
import torch.nn.functional as F

from aitemplate.backend.cpu import CPU
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.model import torch_to_ait_data
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.attention import MultiheadAttention
from aitemplate.frontend.nn.layer_norm import LayerNorm
from aitemplate.frontend.nn.linear import Linear
from aitemplate.frontend.nn.module import Module


class CPUEncoderLayer(Module):
    def __init__(
        self,
        batch,
        seq,
        hidden,
        heads,
        intermediate,
        eps,
    ):
        super().__init__()

        self.batch = batch
        self.seq = seq
        self.hidden = hidden

        self.mha = MultiheadAttention(
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

        self.ln1 = LayerNorm(
            hidden,
            eps=eps,
            dtype="float32",
        )

        self.ffn1 = Linear(
            hidden,
            intermediate,
            specialization="fast_gelu",
            dtype="float32",
        )

        self.ffn2 = Linear(
            intermediate,
            hidden,
            specialization="add",
            dtype="float32",
        )

        self.ln2 = LayerNorm(
            hidden,
            eps=eps,
            dtype="float32",
        )

    def forward(self, x):
        # BERT attention:
        # MHA already includes projection + first residual.
        y = self.mha(x, x)

        # Non-CUDA BERT path flattens before LayerNorm.
        if y._rank() != 2:
            y = ops.reshape()(
                y,
                [-1, self.hidden],
            )

        y = self.ln1(y)

        # FFN: dense + fast_gelu.
        z = self.ffn1(y)

        # dense + second residual.
        z = self.ffn2(z, y)

        z = self.ln2(z)

        return ops.reshape()(
            z,
            [
                self.batch,
                self.seq,
                self.hidden,
            ],
        )


def main():
    batch = 1
    seq = 128
    hidden = 768
    heads = 12
    head_dim = hidden // heads
    intermediate = 3072
    eps = 1e-12
    scale = head_dim ** -0.5

    # Temporary frontend workaround.
    Linear.USE_CUDA = False
    MultiheadAttention.USE_CUDA = False

    layer = CPUEncoderLayer(
        batch=batch,
        seq=seq,
        hidden=hidden,
        heads=heads,
        intermediate=intermediate,
        eps=eps,
    )

    layer.name_parameter_tensor()

    print("===== Parameters =====")
    for name, param in layer.named_parameters():
        # cu_length belongs to CUDA FlashAttention.
        # The CPU/non-CUDA MHA path does not use it.
        if name == "mha.cu_length":
            print(name, "-> skipped (unused on CPU)")
            continue

        tensor = param.tensor()
        tensor._attrs["is_input"] = True
        print(
            name,
            "->",
            tensor._attrs["name"],
        )

    x = Tensor(
        shape=[batch, seq, hidden],
        dtype="float32",
        name="x",
        is_input=True,
    )

    y = layer(x)

    y._attrs["name"] = "output"
    y._attrs["is_output"] = True

    print()
    print("===== Compile =====")

    module = compile_model(
        y,
        CPU(),
        "./tmp",
        "cpu_bert_encoder_layer_numerical",
    )

    torch.manual_seed(0)

    x_pt = torch.randn(
        batch,
        seq,
        hidden,
        dtype=torch.float32,
    )

    # Keep weights close to BERT initialization.
    params = {}

    for name, param in layer.named_parameters():
        if name == "mha.cu_length":
            continue

        ait_name = param.tensor()._attrs["name"]

        shape = [
            dim._attrs["values"][0]
            for dim in param.tensor()._attrs["shape"]
        ]

        if "ln" in name and name.endswith("weight"):
            value = torch.ones(
                shape,
                dtype=torch.float32,
            )

        elif "ln" in name and name.endswith("bias"):
            value = torch.zeros(
                shape,
                dtype=torch.float32,
            )

        else:
            value = (
                torch.randn(
                    shape,
                    dtype=torch.float32,
                )
                * 0.02
            )

        params[ait_name] = value

    inputs = {
        "x": torch_to_ait_data(x_pt),
    }

    for name, value in params.items():
        inputs[name] = torch_to_ait_data(value)

    output = torch.empty(
        batch,
        seq,
        hidden,
        dtype=torch.float32,
    )

    module.run(
        inputs,
        {
            "output": torch_to_ait_data(output),
        },
    )

    # ------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------

    x2d = x_pt.reshape(
        batch * seq,
        hidden,
    )

    # QKV projection.
    qkv = F.linear(
        x2d,
        params["mha_qkv_weight"],
        params["mha_qkv_bias"],
    )

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

    scores = torch.bmm(
        q,
        k.transpose(1, 2),
    ) * scale

    probs = torch.softmax(
        scores,
        dim=-1,
    )

    context = torch.bmm(
        probs,
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

    # Attention output projection + residual.
    attention_output = F.linear(
        context,
        params["mha_proj_weight"],
        params["mha_proj_bias"],
    )

    attention_output = (
        attention_output + x2d
    )

    # First BERT LayerNorm.
    attention_output = F.layer_norm(
        attention_output,
        (hidden,),
        params["ln1_weight"],
        params["ln1_bias"],
        eps,
    )

    # FFN first dense + fast GELU.
    intermediate_output = F.linear(
        attention_output,
        params["ffn1_weight"],
        params["ffn1_bias"],
    )

    intermediate_output = F.gelu(
        intermediate_output,
        approximate="tanh",
    )

    # FFN second dense + residual.
    layer_output = F.linear(
        intermediate_output,
        params["ffn2_weight"],
        params["ffn2_bias"],
    )

    layer_output = (
        layer_output + attention_output
    )

    # Final LayerNorm.
    reference = F.layer_norm(
        layer_output,
        (hidden,),
        params["ln2_weight"],
        params["ln2_bias"],
        eps,
    )

    reference = reference.reshape(
        batch,
        seq,
        hidden,
    )

    diff = (
        output - reference
    ).abs()

    print()
    print("===== CPU BERT encoder layer numerical =====")
    print("batch        :", batch)
    print("sequence     :", seq)
    print("hidden       :", hidden)
    print("heads        :", heads)
    print("intermediate :", intermediate)
    print("output shape :", list(output.shape))
    print("max abs diff :", diff.max().item())
    print("mean abs diff:", diff.mean().item())

    torch.testing.assert_close(
        output,
        reference,
        rtol=2e-3,
        atol=2e-3,
    )

    print("PASS")


if __name__ == "__main__":
    main()
