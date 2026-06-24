#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path

CSV_PATH = Path("ppo_results.csv")
OUTPUT_DIR = Path("sequence")
OPT_PREFIX = "opt "
OPT_SUFFIX = " input.bc -o output.bc"


def normalize_command(command: str) -> list[str]:
    command = command.strip()

    if command.startswith(OPT_PREFIX):
        command = command[len(OPT_PREFIX):]
    elif command == "opt":
        command = ""

    if command.endswith(OPT_SUFFIX):
        command = command[: -len(OPT_SUFFIX)]

    return [token for token in command.split() if token]


def benchmark_name(raw_name: str) -> str:
    return raw_name.rstrip("/").split("/")[-1]


def write_sequences():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    with CSV_PATH.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            command_line = row.get("commandline", "")
            if not command_line:
                continue

            passes = normalize_command(command_line)
            name = benchmark_name(row.get("benchmark", "benchmark"))
            output_path = OUTPUT_DIR / f"{name}.txt"

            with output_path.open("w", encoding="utf-8") as sequence_file:
                sequence_file.write("\n".join(passes))


if __name__ == "__main__":
    write_sequences()
