import argparse
import csv
import glob
import json
import os
import re
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
)


IR_TYPE_REGEX = r"\b(i8|i16|i32|i64|float|double)\b"


def preprocess_ir(text: str) -> str:
    text = re.sub(r"%[\w\.]+", "[REG]", text)
    text = re.sub(r"0x[0-9A-Fa-f]+", "[ADDR]", text)
    text = re.sub(r"\b\d+\b", "[NUM]", text)
    text = re.sub(IR_TYPE_REGEX, "[TYPE]", text)
    text = re.sub(r"@[\w\.]+", "[GLOB]", text)
    text = re.sub(r"\b[\w\.]+:", "[LBL]:", text)
    text = re.sub(r"!\d+", "[MD]", text)
    text = re.sub(r'c".*?"', "[STR]", text)
    return text


def read_ir(path: str, apply_preprocessing: bool) -> str:
    with open(path, encoding="utf-8") as file:
        text = file.read()

    if apply_preprocessing:
        return preprocess_ir(text)

    return text


class IRChunkDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: BertTokenizerFast,
        max_length: int,
        stride: int,
        apply_preprocessing: bool,
    ):
        self.samples: List[Dict[str, List[int]]] = []

        for path in file_paths:
            text = read_ir(path, apply_preprocessing)
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_special_tokens_mask=True,
            )

            for index in range(len(encoded["input_ids"])):
                self.samples.append(
                    {
                        "input_ids": encoded["input_ids"][index],
                        "attention_mask": encoded["attention_mask"][index],
                        "special_tokens_mask": encoded[
                            "special_tokens_mask"
                        ][index],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.samples[index]


def mix_hidden_states(
    outputs,
    layer_strategy: str,
    layer_count: int,
) -> torch.Tensor:
    if layer_strategy == "last":
        return outputs.last_hidden_state

    hidden_states = outputs.hidden_states

    if hidden_states is None:
        raise RuntimeError("The model did not return hidden states.")

    if layer_strategy in {"last_n", "last_n_weighted"}:
        count = min(layer_count, len(hidden_states))
        stacked = torch.stack(hidden_states[-count:], dim=0)

        if layer_strategy == "last_n":
            return stacked.mean(dim=0)

        weights = torch.arange(
            1,
            count + 1,
            device=stacked.device,
            dtype=stacked.dtype,
        )
        weights = weights / weights.sum()

        return (
            stacked * weights.view(-1, 1, 1, 1)
        ).sum(dim=0)

    if layer_strategy == "all_mean":
        return torch.stack(
            hidden_states,
            dim=0,
        ).mean(dim=0)

    raise ValueError(
        f"Unknown layer strategy: {layer_strategy}"
    )


@torch.no_grad()
def embed_one_file(
    path: str,
    tokenizer: BertTokenizerFast,
    model: BertForMaskedLM,
    device: torch.device,
    max_length: int,
    stride: int,
    pooling: str,
    layer_strategy: str,
    layer_count: int,
    chunk_weighting: str,
    normalization: str,
    apply_preprocessing: bool,
) -> np.ndarray:
    text = read_ir(path, apply_preprocessing)

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_attention_mask=True,
    )

    if not encoded["input_ids"]:
        raise RuntimeError(f"Empty encoding: {path}")

    chunk_vectors = []
    chunk_weights = []

    for token_ids, mask in zip(
        encoded["input_ids"],
        encoded["attention_mask"],
    ):
        input_ids = torch.tensor(
            [token_ids],
            device=device,
        )
        attention_mask = torch.tensor(
            [mask],
            device=device,
        )

        outputs = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=layer_strategy != "last",
            return_dict=True,
        )

        hidden = mix_hidden_states(
            outputs,
            layer_strategy,
            layer_count,
        )

        if pooling == "cls":
            vector = hidden[:, 0, :].squeeze(0)

        elif pooling == "token_mean":
            expanded_mask = attention_mask.unsqueeze(-1)
            token_sum = (
                hidden * expanded_mask
            ).sum(dim=1)
            token_count = (
                expanded_mask.sum(dim=1)
                .clamp(min=1)
            )
            vector = (
                token_sum / token_count
            ).squeeze(0)

        else:
            raise ValueError(
                f"Unknown pooling method: {pooling}"
            )

        chunk_vectors.append(vector.cpu())
        chunk_weights.append(
            max(
                float(attention_mask.sum().item()),
                1.0,
            )
        )

    stacked = torch.stack(
        chunk_vectors,
        dim=0,
    )

    if chunk_weighting == "token_count":
        weights = torch.tensor(
            chunk_weights,
            dtype=stacked.dtype,
        )
        weights = weights / weights.sum()

        file_vector = (
            stacked * weights.unsqueeze(1)
        ).sum(dim=0)

    elif chunk_weighting == "uniform":
        file_vector = stacked.mean(dim=0)

    else:
        raise ValueError(
            f"Unknown chunk weighting method: "
            f"{chunk_weighting}"
        )

    if normalization == "l2":
        file_vector = F.normalize(
            file_vector,
            p=2,
            dim=0,
        )

    elif normalization != "none":
        raise ValueError(
            f"Unknown normalization method: "
            f"{normalization}"
        )

    return file_vector.numpy()


@torch.no_grad()
def compute_test_loss(
    file_paths: List[str],
    tokenizer: BertTokenizerFast,
    model: BertForMaskedLM,
    device: torch.device,
    batch_size: int,
    max_length: int,
    stride: int,
    mlm_probability: float,
    apply_preprocessing: bool,
) -> tuple[float, float]:
    dataset = IRChunkDataset(
        file_paths=file_paths,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        apply_preprocessing=apply_preprocessing,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    total_loss = 0.0
    total_masked_tokens = 0

    model.eval().to(device)

    for batch in tqdm(loader, desc="eval"):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        masked_tokens = int(
            (batch["labels"] != -100)
            .sum()
            .item()
        )

        if masked_tokens == 0:
            continue

        loss = model(
            **batch,
            return_dict=True,
        ).loss

        total_loss += (
            float(loss.item()) * masked_tokens
        )
        total_masked_tokens += masked_tokens

    if total_masked_tokens == 0:
        raise RuntimeError(
            "No tokens were masked during evaluation."
        )

    mean_loss = (
        total_loss / total_masked_tokens
    )
    perplexity = float(np.exp(mean_loss))

    return mean_loss, perplexity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_glob",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_path",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        required=True,
    )
    parser.add_argument(
        "--csv_out",
        required=True,
    )
    parser.add_argument(
        "--id_regex",
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
        "--pooling",
        choices=["cls", "token_mean"],
        required=True,
    )
    parser.add_argument(
        "--layer_strategy",
        choices=[
            "last",
            "last_n",
            "last_n_weighted",
            "all_mean",
        ],
        required=True,
    )
    parser.add_argument(
        "--layer_count",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--chunk_weighting",
        choices=["uniform", "token_count"],
        required=True,
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "l2"],
        required=True,
    )
    parser.add_argument(
        "--preprocessing",
        choices=["none", "normalize_ir"],
        required=True,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--eval_loss",
        action="store_true",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
    )
    parser.add_argument(
        "--seed",
        type=int,
    )

    args = parser.parse_args()

    if args.max_length <= 0:
        parser.error(
            "--max_length must be positive."
        )

    if (
        args.stride < 0
        or args.stride >= args.max_length
    ):
        parser.error(
            "--stride must satisfy "
            "0 <= stride < max_length."
        )

    if args.layer_count <= 0:
        parser.error(
            "--layer_count must be positive."
        )

    if args.eval_loss:
        if args.eval_batch_size is None:
            parser.error(
                "--eval_batch_size is required "
                "with --eval_loss."
            )

        if args.mlm_probability is None:
            parser.error(
                "--mlm_probability is required "
                "with --eval_loss."
            )

        if args.eval_batch_size <= 0:
            parser.error(
                "--eval_batch_size must be positive."
            )

        if not 0.0 < args.mlm_probability < 1.0:
            parser.error(
                "--mlm_probability must be "
                "between 0 and 1."
            )

    return args


def resolve_device(
    device_name: str,
) -> torch.device:
    if device_name == "auto":
        device_name = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(device_name)


def extract_id(
    path: str,
    id_regex: Optional[str],
) -> Union[str, int]:
    if id_regex is None:
        parent_name = os.path.basename(
            os.path.dirname(path)
        )
        return parent_name or os.path.basename(path)

    match = re.search(id_regex, path)

    if match is None:
        raise ValueError(
            f"ID pattern did not match: {path}"
        )

    if match.lastindex:
        value = match.group(1)
    else:
        value = match.group(0)

    if value.isdigit():
        return int(value)

    return value


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)

    files = sorted(
        path
        for path in glob.glob(
            args.input_glob,
            recursive=True,
        )
        if os.path.isfile(path)
    )

    if not files:
        raise RuntimeError(
            "No input files matched "
            "the supplied glob pattern."
        )

    print(f"Device: {device}")
    print(f"Files: {len(files)}")

    tokenizer = BertTokenizerFast.from_pretrained(
        args.tokenizer_path
    )
    model = BertForMaskedLM.from_pretrained(
        args.model_path
    )
    model.to(device)
    model.eval()

    apply_preprocessing = (
        args.preprocessing == "normalize_ir"
    )

    if args.eval_loss:
        test_loss, perplexity = compute_test_loss(
            file_paths=files,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            stride=args.stride,
            mlm_probability=args.mlm_probability,
            apply_preprocessing=apply_preprocessing,
        )

        print(
            f"Test MLM loss: {test_loss:.6f}"
        )
        print(
            f"Perplexity: {perplexity:.6f}"
        )

    vectors = []
    ids = []
    latencies = []

    for path in tqdm(files, desc="embed"):
        start_time = time.perf_counter()

        vector = embed_one_file(
            path=path,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
            pooling=args.pooling,
            layer_strategy=args.layer_strategy,
            layer_count=args.layer_count,
            chunk_weighting=args.chunk_weighting,
            normalization=args.normalization,
            apply_preprocessing=apply_preprocessing,
        )

        latency = (
            time.perf_counter() - start_time
        )

        vectors.append(vector)
        latencies.append(latency)
        ids.append(
            extract_id(path, args.id_regex)
        )

    output_dir = os.path.dirname(
        os.path.abspath(args.csv_out)
    )
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    with open(
        args.csv_out,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["id", "embedding"])

        for sample_id, vector in zip(
            ids,
            vectors,
        ):
            writer.writerow(
                [
                    sample_id,
                    json.dumps(vector.tolist()),
                ]
            )

    print(
        f"Saved embeddings: {args.csv_out}"
    )
    print(
        "Average latency per input: "
        f"{np.mean(latencies):.4f} seconds"
    )


if __name__ == "__main__":
    main()
