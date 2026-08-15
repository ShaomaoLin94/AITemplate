import torch
import torch.nn.functional as F

from aitemplate.backend.cpu import CPU
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.model import torch_to_ait_data
from aitemplate.frontend import Tensor


def main():
    batch = 1
    seq = 128
    hidden = 768

    vocab_size = 30522
    max_position_embeddings = 512
    type_vocab_size = 2
    eps = 1e-12

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

    word_embeddings = Tensor(
        shape=[vocab_size, hidden],
        dtype="float32",
        name="word_embeddings",
        is_input=True,
    )

    token_type_embeddings = Tensor(
        shape=[type_vocab_size, hidden],
        dtype="float32",
        name="token_type_embeddings",
        is_input=True,
    )

    position_embeddings = Tensor(
        shape=[max_position_embeddings, hidden],
        dtype="float32",
        name="position_embeddings",
        is_input=True,
    )

    gamma = Tensor(
        shape=[hidden],
        dtype="float32",
        name="gamma",
        is_input=True,
    )

    beta = Tensor(
        shape=[hidden],
        dtype="float32",
        name="beta",
        is_input=True,
    )

    output = ops.bert_embeddings()(
        input_ids,
        token_type_ids,
        position_ids,
        word_embeddings,
        token_type_embeddings,
        position_embeddings,
        gamma,
        beta,
        eps,
    )

    output._attrs["name"] = "output"
    output._attrs["is_output"] = True

    module = compile_model(
        output,
        CPU(),
        "./tmp",
        "cpu_bert_embeddings",
    )

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

    gamma_pt = torch.ones(
        hidden,
        dtype=torch.float32,
    )

    beta_pt = torch.zeros(
        hidden,
        dtype=torch.float32,
    )

    output_pt = torch.empty(
        batch,
        seq,
        hidden,
        dtype=torch.float32,
    )

    module.run(
        {
            "input_ids":
                torch_to_ait_data(input_ids_pt),
            "token_type_ids":
                torch_to_ait_data(token_type_ids_pt),
            "position_ids":
                torch_to_ait_data(position_ids_pt),
            "word_embeddings":
                torch_to_ait_data(word_embeddings_pt),
            "token_type_embeddings":
                torch_to_ait_data(token_type_embeddings_pt),
            "position_embeddings":
                torch_to_ait_data(position_embeddings_pt),
            "gamma":
                torch_to_ait_data(gamma_pt),
            "beta":
                torch_to_ait_data(beta_pt),
        },
        {
            "output":
                torch_to_ait_data(output_pt),
        },
    )

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
        gamma_pt,
        beta_pt,
        eps,
    )

    diff = (
        output_pt - reference
    ).abs()

    print()
    print("===== CPU BERT embeddings =====")
    print("batch        :", batch)
    print("sequence     :", seq)
    print("hidden       :", hidden)
    print("vocab        :", vocab_size)
    print("output shape :", list(output_pt.shape))
    print("max abs diff :", diff.max().item())
    print("mean abs diff:", diff.mean().item())

    torch.testing.assert_close(
        output_pt,
        reference,
        rtol=5e-4,
        atol=5e-4,
    )

    print("PASS")


if __name__ == "__main__":
    main()
