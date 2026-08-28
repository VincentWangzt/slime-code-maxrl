"""Create the compact word-level Qwen2 checkpoint used by the maze task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config

from maze.constants import MAZE_MAX_SEQUENCE_LENGTH, MAZE_VOCAB, MAZE_VOCAB_SIZE


def create_model(*, output_dir: Path, seed: int) -> dict[str, int | str]:
    """Initialize and save a deterministic Hugging Face Qwen2 checkpoint."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty model directory: {output_dir}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_backend = Tokenizer(models.WordLevel(vocab=MAZE_VOCAB, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_backend.add_special_tokens(
        [AddedToken(token, special=True) for token in ("<pad>", "<bos>", "<eos>", "<unk>")]
    )
    tokenizer_backend.add_tokens(
        [
            AddedToken(token, normalized=False, single_word=True)
            for token in MAZE_VOCAB
            if not token.startswith("<")
        ]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=MAZE_MAX_SEQUENCE_LENGTH,
        clean_up_tokenization_spaces=False,
    )

    config = Qwen2Config(
        vocab_size=MAZE_VOCAB_SIZE,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=MAZE_MAX_SEQUENCE_LENGTH,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_bias=True,
        tie_word_embeddings=True,
        bos_token_id=MAZE_VOCAB["<bos>"],
        eos_token_id=MAZE_VOCAB["<eos>"],
        pad_token_id=MAZE_VOCAB["<pad>"],
        use_cache=True,
        torch_dtype="bfloat16",
    )
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    summary: dict[str, int | str] = {
        "output_dir": str(output_dir),
        "vocab_size": MAZE_VOCAB_SIZE,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seed": seed,
    }
    (output_dir / "maze_model_metadata.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/models/maze-qwen2"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(create_model(output_dir=args.output_dir, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
