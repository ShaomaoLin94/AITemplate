import torch
import torch.nn.functional as F

from aitemplate.backend.cpu import CPU
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.model import torch_to_ait_data
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.attention import MultiheadAttention
from aitemplate.frontend.nn.container import ModuleList
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
        y = self.mha(x, x)

        if y._rank() != 2:
            y = ops.reshape()(
                y,
                [-1, self.hidden],
            )

        y = self.ln1(y)

        z = self.ffn1(y)
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


class CPUEncoder(Module):
    def __init__(
        self,
        num_layers,
        batch,
        seq,
        hidden,
        heads,
        intermediate,
        eps,
    ):
        super().__init__()

        self.layers = ModuleList(
            [
                CPUEncoderLayer(
                    batch=batch,
                    seq=seq,
                    hidden=hidden,
                    heads=heads,
                    intermediate=intermediate,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


def pytorch_encoder_layer(
    x,
    params,
    prefix,
    batch,
    seq,
    hidden,
    heads,
    head_dim,
    eps,
):
    x2d = x.reshape(
        batch * seq,
        hidden,
    )

    qkv = F.linear(
        x2d,
        params[prefix + "mha.qkv.weight"],
        params[prefix + "mha.qkv.bias"],
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
    ) * (head_dim ** -0.5)

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

    attention_output = F.linear(
        context,
        params[prefix + "mha.proj.weight"],
        params[prefix + "mha.proj.bias"],
    )

    attention_output = attention_output + x2d

    attention_output = F.layer_norm(
        attention_output,
        (hidden,),
        params[prefix + "ln1.weight"],
        params[prefix + "ln1.bias"],
        eps,
    )

    intermediate_output = F.linear(
        attention_output,
        params[prefix + "ffn1.weight"],
        params[prefix + "ffn1.bias"],
    )

    intermediate_output = F.gelu(
        intermediate_output,
        approximate="tanh",
    )

    layer_output = F.linear(
        intermediate_output,
        params[prefix + "ffn2.weight"],
        params[prefix + "ffn2.bias"],
    )

    layer_output = layer_output + attention_output

    layer_output = F.layer_norm(
        layer_output,
        (hidden,),
        params[prefix + "ln2.weight"],
        params[prefix + "ln2.bias"],
        eps,
    )

    return layer_output.reshape(
        batch,
        seq,
        hidden,
    )


def main():
    batch = 1
    seq = 128
    hidden = 768
    heads = 12
    head_dim = hidden // heads
    intermediate = 3072
    num_layers = 12

    vocab_size = 30522
    max_position_embeddings = 512
    type_vocab_size = 2

    eps = 1e-12

    # Temporary frontend workaround.
    Linear.USE_CUDA = False
    MultiheadAttention.USE_CUDA = False

    # ------------------------------------------------------------
    # BERT inputs
    # ------------------------------------------------------------

    input_ids = Tensor(
        shape=[batch, seq],
        dtype="int64",
        name="input_ids",
        is_input=True,
    )

    token_type_ids = Tensor(
        shape=[batch, seq],
        dtype="int64",
        name="token_type_ids",
        is_input=True,
    )

    position_ids = Tensor(
        shape=[batch, seq],
        dtype="int64",
        name="position_ids",
        is_input=True,
    )

    # ------------------------------------------------------------
    # Embedding parameters
    # ------------------------------------------------------------

    word_embeddings = Tensor(
        shape=[vocab_size, hidden],
        dtype="float32",
        name="word_embeddings",
    )

    token_type_embeddings = Tensor(
        shape=[type_vocab_size, hidden],
        dtype="float32",
        name="token_type_embeddings",
    )

    position_embeddings = Tensor(
        shape=[max_position_embeddings, hidden],
        dtype="float32",
        name="position_embeddings",
    )

    embedding_gamma = Tensor(
        shape=[hidden],
        dtype="float32",
        name="embedding_gamma",
    )

    embedding_beta = Tensor(
        shape=[hidden],
        dtype="float32",
        name="embedding_beta",
    )

    # ------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------

    x = ops.bert_embeddings()(
        input_ids,
        token_type_ids,
        position_ids,
        word_embeddings,
        token_type_embeddings,
        position_embeddings,
        embedding_gamma,
        embedding_beta,
        eps,
    )

    # ------------------------------------------------------------
    # 12-layer encoder
    # ------------------------------------------------------------

    encoder = CPUEncoder(
        num_layers=num_layers,
        batch=batch,
        seq=seq,
        hidden=hidden,
        heads=heads,
        intermediate=intermediate,
        eps=eps,
    )

    encoder.name_parameter_tensor()

    parameter_count = 0

    for name, param in encoder.named_parameters():
        if name.endswith("mha.cu_length"):
            continue

        # Frontend parameters remain non-input tensors.
        # AITemplate will mark them as unbound constants.
        tensor = param.tensor()

        parameter_count += 1

    print("encoder parameter tensors:", parameter_count)

    output_tensor = encoder(x)

    output_tensor._attrs["name"] = "output"
    output_tensor._attrs["is_output"] = True

    print()
    print("===== Compile full CPU BERT =====")

    module = compile_model(
        output_tensor,
        CPU(),
        "./tmp",
        "cpu_bert_full_numerical",
    )

    # ------------------------------------------------------------
    # Test data
    # ------------------------------------------------------------

    torch.manual_seed(0)

    input_ids_pt = torch.randint(
        0,
        vocab_size,
        (batch, seq),
        dtype=torch.int64,
    )

    token_type_ids_pt = torch.randint(
        0,
        type_vocab_size,
        (batch, seq),
        dtype=torch.int64,
    )

    position_ids_pt = (
        torch.arange(
            seq,
            dtype=torch.int64,
        )
        .reshape(1, seq)
        .expand(batch, -1)
        .contiguous()
    )

    word_embeddings_pt = (
        torch.randn(
            vocab_size,
            hidden,
            dtype=torch.float32,
        )
        * 0.02
    )

    token_type_embeddings_pt = (
        torch.randn(
            type_vocab_size,
            hidden,
            dtype=torch.float32,
        )
        * 0.02
    )

    position_embeddings_pt = (
        torch.randn(
            max_position_embeddings,
            hidden,
            dtype=torch.float32,
        )
        * 0.02
    )

    embedding_gamma_pt = torch.ones(
        hidden,
        dtype=torch.float32,
    )

    embedding_beta_pt = torch.zeros(
        hidden,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------
    # Encoder weights
    # ------------------------------------------------------------

    logical_params = {}

    inputs = {
        "input_ids":
            torch_to_ait_data(input_ids_pt),
        "token_type_ids":
            torch_to_ait_data(token_type_ids_pt),
        "position_ids":
            torch_to_ait_data(position_ids_pt),
    }

    constants = {
        "word_embeddings":
            torch_to_ait_data(word_embeddings_pt),
        "token_type_embeddings":
            torch_to_ait_data(token_type_embeddings_pt),
        "position_embeddings":
            torch_to_ait_data(position_embeddings_pt),
        "embedding_gamma":
            torch_to_ait_data(embedding_gamma_pt),
        "embedding_beta":
            torch_to_ait_data(embedding_beta_pt),
    }

    for name, param in encoder.named_parameters():
        if name.endswith("mha.cu_length"):
            continue

        tensor = param.tensor()
        ait_name = tensor._attrs["name"]

        shape = [
            dim._attrs["values"][0]
            for dim in tensor._attrs["shape"]
        ]

        if (
            ("ln1.weight" in name)
            or ("ln2.weight" in name)
        ):
            value = torch.ones(
                shape,
                dtype=torch.float32,
            )

        elif (
            ("ln1.bias" in name)
            or ("ln2.bias" in name)
        ):
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

        logical_params[name] = value
        constants[ait_name] = torch_to_ait_data(value)

    # ------------------------------------------------------------
    # Run AITemplate
    # ------------------------------------------------------------

    output = torch.empty(
        batch,
        seq,
        hidden,
        dtype=torch.float32,
    )

    print()
    print("===== Run AITemplate full BERT =====")

    module.set_many_constants(constants)

    module.run(
        inputs,
        {
            "output": torch_to_ait_data(output),
        },
    )

    print("AITemplate run complete")

    # ------------------------------------------------------------
    # PyTorch reference: embeddings
    # ------------------------------------------------------------

    print()
    print("===== Run PyTorch reference =====")

    reference = (
        F.embedding(
            input_ids_pt,
            word_embeddings_pt,
        )
        + F.embedding(
            token_type_ids_pt,
            token_type_embeddings_pt,
        )
        + F.embedding(
            position_ids_pt,
            position_embeddings_pt,
        )
    )

    reference = F.layer_norm(
        reference,
        (hidden,),
        embedding_gamma_pt,
        embedding_beta_pt,
        eps,
    )

    # ------------------------------------------------------------
    # PyTorch reference: 12 encoder layers
    # ------------------------------------------------------------

    for layer_idx in range(num_layers):
        prefix = "layers." + str(layer_idx) + "."

        reference = pytorch_encoder_layer(
            reference,
            logical_params,
            prefix,
            batch,
            seq,
            hidden,
            heads,
            head_dim,
            eps,
        )

        print(
            "PyTorch layer",
            layer_idx,
            "complete",
        )

    # ------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------

    diff = (
        output - reference
    ).abs()

    print()
    print("===== Full CPU BERT numerical =====")
    print("layers       :", num_layers)
    print("batch        :", batch)
    print("sequence     :", seq)
    print("hidden       :", hidden)
    print("heads        :", heads)
    print("intermediate :", intermediate)
    print("vocab        :", vocab_size)
    print("output shape :", list(output.shape))
    print("max abs diff :", diff.max().item())
    print("mean abs diff:", diff.mean().item())

    torch.testing.assert_close(
        output,
        reference,
        rtol=5e-3,
        atol=5e-3,
    )

    print("PASS")


if __name__ == "__main__":
    main()
