#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from typing import Iterator, List

from tokenizers import BertWordPieceTokenizer
from tokenizers.trainers import WordPieceTrainer
from transformers import BertTokenizerFast


IR_TYPE_REGEX = r"\b(i8|i16|i32|i64|float|double)\b"

BERT_SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]

IR_SPECIAL_TOKENS = [
    "[REG]",
    "[ADDR]",
    "[NUM]",
    "[TYPE]",
    "[GLOB]",
    "[LBL]",
    "[MD]",
    "[STR]",
]


def preprocess_ir(text: str) -> str:
    text = re.sub(r"%[\w.]+", "[REG]", text)
    text = re.sub(r"0x[0-9A-Fa-f]+", "[ADDR]", text)
    text = re.sub(r"\b\d+\b", "[NUM]", text)
    text = re.sub(IR_TYPE_REGEX, "[TYPE]", text)
    text = re.sub(r"@[\w.]+", "[GLOB]", text)
    text = re.sub(r"\b[\w.]+:", "[LBL]:", text)
    text = re.sub(r"!\d+", "[MD]", text)
    text = re.sub(r'c".*?"', "[STR]", text)
    return text


def natural_sort_key(path: str) -> List[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path)
    ]


def read_corpus(
    file_paths: List[str],
    preprocessing: bool,
) -> Iterator[str]:
    for path in file_paths:
        with open(path, encoding="utf-8") as file:
            text = file.read()

        if preprocessing:
            text = preprocess_ir(text)

        yield text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pattern",
        required=True,
        help="Glob pattern for LLVM IR files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    parser.add_argument(
        "--preprocessing",
        choices=["on", "off"],
        required=True,
    )
    parser.add_argument(
        "--max_files",
        type=int,
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--lowercase",
        choices=["on", "off"],
        required=True,
    )
    parser.add_argument(
        "--strip_accents",
        choices=["on", "off"],
        required=True,
    )

    args = parser.parse_args()

    if args.max_files is not None and args.max_files <= 0:
        parser.error(
            "--max_files must be positive."
        )

    if args.vocab_size <= 0:
        parser.error(
            "--vocab_size must be positive."
        )

    if args.min_frequency <= 0:
        parser.error(
            "--min_frequency must be positive."
        )

    return args


def main() -> None:
    args = parse_args()

    file_paths = sorted(
        (
            path
            for path in glob.glob(
                args.pattern,
                recursive=True,
            )
            if os.path.isfile(path)
        ),
        key=natural_sort_key,
    )

    if args.max_files is not None:
        file_paths = file_paths[:args.max_files]

    if not file_paths:
        raise RuntimeError(
            "No IR files matched the supplied pattern: "
            f"{args.pattern}"
        )

    preprocessing_enabled = (
        args.preprocessing == "on"
    )
    lowercase_enabled = (
        args.lowercase == "on"
    )
    strip_accents_enabled = (
        args.strip_accents == "on"
    )

    special_tokens = BERT_SPECIAL_TOKENS.copy()

    if preprocessing_enabled:
        special_tokens.extend(
            IR_SPECIAL_TOKENS
        )

    tokenizer = BertWordPieceTokenizer(
        lowercase=lowercase_enabled,
        strip_accents=strip_accents_enabled,
    )

    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
    )

    tokenizer.train_from_iterator(
        read_corpus(
            file_paths=file_paths,
            preprocessing=preprocessing_enabled,
        ),
        trainer=trainer,
        length=len(file_paths),
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    tokenizer.save_model(
        args.output_dir
    )

    vocabulary_path = os.path.join(
        args.output_dir,
        "vocab.txt",
    )

    additional_special_tokens = (
        IR_SPECIAL_TOKENS
        if preprocessing_enabled
        else []
    )

    fast_tokenizer = BertTokenizerFast(
        vocab_file=vocabulary_path,
        do_lower_case=lowercase_enabled,
        strip_accents=strip_accents_enabled,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        additional_special_tokens=(
            additional_special_tokens
        ),
    )

    fast_tokenizer.save_pretrained(
        args.output_dir
    )

    config = {
        "preprocessing": args.preprocessing,
        "files_used": len(file_paths),
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": len(
            fast_tokenizer
        ),
        "min_frequency": args.min_frequency,
        "lowercase": lowercase_enabled,
        "strip_accents": strip_accents_enabled,
        "ir_special_tokens": (
            additional_special_tokens
        ),
    }

    config_path = os.path.join(
        args.output_dir,
        "tokenizer_training_config.json",
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Files used: {len(file_paths)}")
    print(
        f"Preprocessing: {args.preprocessing}"
    )
    print(
        f"Vocabulary size: "
        f"{len(fast_tokenizer)}"
    )
    print(
        f"Tokenizer saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
