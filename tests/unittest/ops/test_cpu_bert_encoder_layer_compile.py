import aitemplate.backend.cpu

from aitemplate.backend.cpu.target_def import CPU
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor, nn
from aitemplate.frontend.nn.attention import MultiheadAttention
from aitemplate.frontend.nn.linear import Linear


class CPUBertEncoderLayer(nn.Module):
    def __init__(
        self,
        batch_size=1,
        seq_len=128,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        eps=1e-5,
        dtype="float32",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.attention = MultiheadAttention(
            dim=hidden_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            has_residual=True,
            dtype=dtype,
        )

        self.attention_norm = nn.LayerNorm(
            hidden_size,
            eps=eps,
            dtype=dtype,
        )

        self.intermediate = nn.Linear(
            hidden_size,
            intermediate_size,
            specialization="fast_gelu",
            dtype=dtype,
        )

        self.output = nn.Linear(
            intermediate_size,
            hidden_size,
            specialization="add",
            dtype=dtype,
        )

        self.output_norm = nn.LayerNorm(
            hidden_size,
            eps=eps,
            dtype=dtype,
        )

    def forward(self, x):
        # MultiheadAttention performs:
        # QKV projection -> attention -> output projection + residual.
        attn = self.attention(x, x)

        # Non-CUDA Linear works on a 2D representation.
        attn = ops.reshape()(
            attn,
            [
                self.batch_size * self.seq_len,
                self.hidden_size,
            ],
        )

        attn = self.attention_norm(attn)

        hidden = self.intermediate(attn)
        hidden = self.output(hidden, attn)
        hidden = self.output_norm(hidden)

        return ops.reshape()(
            hidden,
            [
                self.batch_size,
                self.seq_len,
                self.hidden_size,
            ],
        )


def main():
    batch_size = 1
    seq_len = 128
    hidden_size = 768

    print("===== CPU BERT encoder layer compile probe =====")
    print("batch       :", batch_size)
    print("sequence    :", seq_len)
    print("hidden      :", hidden_size)
    print("heads       :", 12)
    print("head dim    :", 64)
    print("intermediate:", 3072)

    # AITemplate's frontend currently auto-detects only CUDA/ROCm.
    # Explicitly select the existing non-CUDA frontend paths for this
    # CPU backend compile probe.
    MultiheadAttention.USE_CUDA = False
    Linear.USE_CUDA = False

    model = CPUBertEncoderLayer(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=12,
        intermediate_size=3072,
        dtype="float32",
    )

    model.name_parameter_tensor()

    X = Tensor(
        shape=[batch_size, seq_len, hidden_size],
        dtype="float32",
        name="X",
        is_input=True,
    )

    print("\n===== Build graph =====")

    Y = model(X)

    Y._attrs["name"] = "Y"
    Y._attrs["is_output"] = True

    print("graph construction: PASS")

    print("\n===== Compile =====")

    module = compile_model(
        Y,
        CPU(),
        "./tmp",
        "cpu_bert_encoder_layer",
    )

    print("\n========================================")
    print("CPU BERT encoder layer compile: PASS")
    print("========================================")

    print("\n===== Compiled operators =====")

    op_names = set()
    for tensor in module.debug_sorted_graph:
        for op in tensor._attrs["src_ops"]:
            op_names.add(op._attrs["op"])

    for name in sorted(op_names):
        print(name)


if __name__ == "__main__":
    main()
