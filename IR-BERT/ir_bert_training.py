#!/usr/bin/env python3
import argparse
import glob
import random
import re
from typing import Dict, List

from torch.utils.data import Dataset
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


IR_TYPE_REGEX = r"\b(i8|i16|i32|i64|float|double)\b"


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


def read_ir(path: str, preprocessing: bool) -> str:
    with open(path, encoding="utf-8") as file:
        text = file.read()

    if preprocessing:
        text = preprocess_ir(text)

    return text


class IRChunkDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: BertTokenizerFast,
        max_length: int,
        stride: int,
        preprocessing: bool,
    ):
        self.features: List[Dict[str, List[int]]] = []

        for path in file_paths:
            text = read_ir(path, preprocessing)

            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_special_tokens_mask=True,
            )

            for index in range(len(encoded["input_ids"])):
                self.features.append(
                    {
                        "input_ids": encoded["input_ids"][index],
                        "attention_mask": encoded["attention_mask"][index],
                        "special_tokens_mask": encoded[
                            "special_tokens_mask"
                        ][index],
                    }
                )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.features[index]


def split_train_val(
    files: List[str],
    val_ratio: float,
    seed: int,
) -> tuple[List[str], List[str]]:
    shuffled_files = files.copy()
    random.Random(seed).shuffle(shuffled_files)

    if val_ratio == 0:
        return shuffled_files, []

    validation_count = int(
        len(shuffled_files) * val_ratio
    )
    validation_count = max(
        1,
        validation_count,
    )
    validation_count = min(
        validation_count,
        len(shuffled_files) - 1,
    )

    validation_files = shuffled_files[
        :validation_count
    ]
    training_files = shuffled_files[
        validation_count:
    ]

    return training_files, validation_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pattern",
        required=True,
        help="Glob pattern for LLVM IR files",
    )
    parser.add_argument(
        "--tokenizer_path",
        required=True,
    )
    parser.add_argument(
        "--base_model",
        required=True,
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
        "--max_length",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--stride",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--mlm_prob",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
    )

    args = parser.parse_args()

    if args.max_length <= 0:
        parser.error(
            "--max_length must be positive."
        )

    if args.stride < 0:
        parser.error(
            "--stride cannot be negative."
        )

    if args.stride >= args.max_length:
        parser.error(
            "--stride must be smaller than "
            "--max_length."
        )

    if args.epochs <= 0:
        parser.error(
            "--epochs must be positive."
        )

    if args.batch_size <= 0:
        parser.error(
            "--batch_size must be positive."
        )

    if (
        args.eval_batch_size is not None
        and args.eval_batch_size <= 0
    ):
        parser.error(
            "--eval_batch_size must be positive."
        )

    if args.gradient_accumulation_steps <= 0:
        parser.error(
            "--gradient_accumulation_steps "
            "must be positive."
        )

    if not 0.0 < args.mlm_prob < 1.0:
        parser.error(
            "--mlm_prob must be between 0 and 1."
        )

    if not 0.0 <= args.val_ratio < 1.0:
        parser.error(
            "--val_ratio must satisfy "
            "0 <= val_ratio < 1."
        )

    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error(
            "--warmup_ratio must satisfy "
            "0 <= warmup_ratio < 1."
        )

    if args.max_files is not None:
        if args.max_files <= 0:
            parser.error(
                "--max_files must be positive."
            )

    if args.fp16 and args.bf16:
        parser.error(
            "--fp16 and --bf16 cannot be used "
            "at the same time."
        )

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    all_files = sorted(
        glob.glob(
            args.pattern,
            recursive=True,
        )
    )

    if args.max_files is not None:
        all_files = all_files[:args.max_files]

    if not all_files:
        raise RuntimeError(
            "No IR files matched the supplied pattern: "
            f"{args.pattern}"
        )

    train_files, validation_files = split_train_val(
        files=all_files,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Files: {len(all_files)}")
    print(f"Train files: {len(train_files)}")
    print(
        f"Validation files: "
        f"{len(validation_files)}"
    )
    print(
        f"Preprocessing: "
        f"{args.preprocessing}"
    )

    tokenizer = BertTokenizerFast.from_pretrained(
        args.tokenizer_path
    )

    model = BertForMaskedLM.from_pretrained(
        args.base_model
    )
    model.resize_token_embeddings(
        len(tokenizer)
    )

    preprocessing_enabled = (
        args.preprocessing == "on"
    )

    train_dataset = IRChunkDataset(
        file_paths=train_files,
        tokenizer=tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        preprocessing=preprocessing_enabled,
    )

    validation_dataset = None

    if validation_files:
        validation_dataset = IRChunkDataset(
            file_paths=validation_files,
            tokenizer=tokenizer,
            max_length=args.max_length,
            stride=args.stride,
            preprocessing=preprocessing_enabled,
        )

    print(
        f"Train chunks: {len(train_dataset)}"
    )

    if validation_dataset is not None:
        print(
            "Validation chunks: "
            f"{len(validation_dataset)}"
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )

    eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size is not None
        else args.batch_size
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=(
            args.overwrite_output_dir
        ),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=(
            args.batch_size
        ),
        per_device_eval_batch_size=(
            eval_batch_size
        ),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=(
            args.save_total_limit
        ),
        prediction_loss_only=True,
        report_to=[],
        fp16=args.fp16,
        bf16=args.bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    trainer.train()

    if validation_dataset is not None:
        metrics = trainer.evaluate()
        print(
            "Validation loss: "
            f"{metrics.get('eval_loss', float('nan')):.6f}"
        )

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(
        "Training completed. "
        f"Output directory: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
