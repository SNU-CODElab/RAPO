#!/usr/bin/env python3
import argparse
import glob
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


TOTAL_PATTERNS = [
    re.compile(r"Total\s+of\s+(\d+)\s+instructions", re.IGNORECASE),
    re.compile(r"Total\s+(?:Insts|Instructions)\s*:\s*(\d+)", re.IGNORECASE),
    re.compile(r"Total\s+number\s+of\s+insts\s*:\s*(\d+)", re.IGNORECASE),
    re.compile(r"Total\s+inst(?:ruction)?s?\s*[:=]\s*(\d+)", re.IGNORECASE),
]

IR_INSTRUCTION_PATTERN = re.compile(
    r"""
    ^\s*
    (?:%[-\w.$]+\s*=\s*)?
    (?:(?:tail|musttail|notail)\s+)?
    (?:
        add|fadd|sub|fsub|mul|fmul|udiv|sdiv|fdiv|urem|srem|frem|
        shl|lshr|ashr|and|or|xor|freeze|
        alloca|load|store|fence|cmpxchg|atomicrmw|getelementptr|
        trunc|zext|sext|fptrunc|fpext|uitofp|sitofp|fptoui|fptosi|
        inttoptr|ptrtoint|bitcast|addrspacecast|
        icmp|fcmp|phi|select|
        call|callbr|invoke|va_arg|
        extractelement|insertelement|shufflevector|extractvalue|insertvalue|
        landingpad|catchpad|cleanuppad|catchswitch|catchret|cleanupret|resume|
        br|switch|indirectbr|ret|unreachable
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def run_command(
    command: List[str],
    timeout: int,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def ll_to_bc(
    input_ll: Path,
    output_bc: Path,
    llvm_as: str,
    timeout: int,
) -> bool:
    result = run_command(
        [llvm_as, str(input_ll), "-o", str(output_bc)],
        timeout,
    )

    if result.returncode != 0:
        print(
            f"  llvm-as failed for {input_ll.name}: "
            f"{result.stderr.strip()}"
        )
        return False

    return True


def bc_to_ll(
    input_bc: Path,
    output_ll: Path,
    llvm_dis: str,
    timeout: int,
) -> bool:
    result = run_command(
        [llvm_dis, str(input_bc), "-o", str(output_ll)],
        timeout,
    )

    if result.returncode != 0:
        print(
            f"  llvm-dis failed for {input_bc.name}: "
            f"{result.stderr.strip()}"
        )
        return False

    return True


def count_textual_ir(ir_path: Path) -> int:
    count = 0
    inside_function = False

    with ir_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            stripped = line.strip()

            if stripped.startswith("define ") and stripped.endswith("{"):
                inside_function = True
                continue

            if inside_function and stripped == "}":
                inside_function = False
                continue

            if not inside_function:
                continue

            if (
                not stripped
                or stripped.startswith(";")
                or stripped.startswith("!")
                or stripped.endswith(":")
            ):
                continue

            if IR_INSTRUCTION_PATTERN.match(stripped):
                count += 1

    return count


def parse_instcount_output(output: str) -> Optional[int]:
    for pattern in TOTAL_PATTERNS:
        match = pattern.search(output)
        if match:
            return int(match.group(1))

    return None


def count_ir_instructions(
    ir_path: Path,
    opt: str,
    llvm_as: str,
    llvm_dis: str,
    assemble_timeout: int,
    analysis_timeout: int,
) -> Optional[int]:
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)

            if ir_path.suffix == ".ll":
                input_bc = temporary_path / "input.bc"
                if not ll_to_bc(
                    ir_path,
                    input_bc,
                    llvm_as,
                    assemble_timeout,
                ):
                    return None
                textual_ir = ir_path
            else:
                input_bc = ir_path
                textual_ir = temporary_path / "input.ll"

            commands = [
                [
                    opt,
                    "-instcount",
                    "-disable-output",
                    str(input_bc),
                ],
                [
                    opt,
                    "-instcount",
                    "-analyze",
                    str(input_bc),
                ],
                [
                    opt,
                    "-passes=instcount",
                    "-disable-output",
                    str(input_bc),
                ],
            ]

            for command in commands:
                result = run_command(command, analysis_timeout)
                output = (
                    (result.stdout or "")
                    + "\n"
                    + (result.stderr or "")
                )

                if result.returncode == 0:
                    parsed = parse_instcount_output(output)
                    if parsed is not None:
                        return parsed

            if ir_path.suffix != ".ll":
                if not bc_to_ll(
                    input_bc,
                    textual_ir,
                    llvm_dis,
                    assemble_timeout,
                ):
                    return None

            return count_textual_ir(textual_ir)

    except (OSError, subprocess.SubprocessError) as error:
        print(f"  Instruction counting failed: {error}")
        return None


def optimize_with_level(
    input_bc: Path,
    output_bc: Path,
    opt_level: str,
    opt: str,
    timeout: int,
    extra_opt_args: List[str],
) -> bool:
    command = [
        opt,
        *extra_opt_args,
        opt_level,
        str(input_bc),
        "-o",
        str(output_bc),
    ]
    result = run_command(command, timeout)

    if result.returncode != 0:
        print(
            f"  {opt_level} optimization failed: "
            f"{result.stderr.strip()}"
        )
        return False

    return True


def parse_optimization_sequence(sequence: str) -> List[str]:
    tokens = shlex.split(sequence)

    if tokens and Path(tokens[0]).name.startswith("opt"):
        tokens = tokens[1:]

    cleaned_tokens: List[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]

        if token == "-o":
            index += 2
            continue

        if Path(token).name in {"input.bc", "output.bc"}:
            index += 1
            continue

        cleaned_tokens.append(token)
        index += 1

    return cleaned_tokens


def apply_optimization_sequence(
    input_bc: Path,
    output_bc: Path,
    sequence: str,
    opt: str,
    timeout: int,
    extra_opt_args: List[str],
) -> bool:
    try:
        pass_arguments = parse_optimization_sequence(sequence)
    except ValueError as error:
        print(f"  Invalid optimization sequence: {error}")
        return False

    if not pass_arguments:
        print("  Optimization sequence is empty")
        return False

    command = [
        opt,
        *extra_opt_args,
        *pass_arguments,
        str(input_bc),
        "-o",
        str(output_bc),
    ]
    result = run_command(command, timeout)

    if result.returncode != 0:
        print(
            "  Optimization sequence failed: "
            f"{result.stderr.strip()}"
        )
        return False

    return True


def load_name_mapping(
    mapping_path: Optional[Path],
) -> Dict[str, str]:
    if mapping_path is None:
        return {}

    with mapping_path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    if not isinstance(mapping, dict):
        raise ValueError(
            "The name mapping file must contain a JSON object."
        )

    return {
        str(key): str(value)
        for key, value in mapping.items()
    }


def normalize_filename(
    filename: str,
    prefix_to_remove: str,
    name_mapping: Dict[str, str],
) -> str:
    name = Path(filename).stem

    if prefix_to_remove and name.startswith(prefix_to_remove):
        name = name[len(prefix_to_remove):]

    return name_mapping.get(name, name)


def load_benchmark_commands(
    ppo_df: pd.DataFrame,
    benchmark_column: str,
    commandline_column: str,
    benchmark_prefix: str,
    filename_prefix: str,
    name_mapping: Dict[str, str],
) -> Dict[str, str]:
    commands: Dict[str, str] = {}

    for _, row in ppo_df.iterrows():
        benchmark = str(row[benchmark_column])
        commandline = row[commandline_column]

        if pd.isna(commandline):
            continue

        if benchmark_prefix and benchmark_prefix in benchmark:
            benchmark = benchmark.split(benchmark_prefix, 1)[1]

        benchmark = normalize_filename(
            benchmark,
            filename_prefix,
            name_mapping,
        )
        commands[benchmark] = str(commandline)

    return commands


def build_cluster_lookup(
    centroid_df: pd.DataFrame,
    cluster_column: str,
    cluster_files_column: str,
    closest_filename_column: str,
    cluster_separator: str,
    filename_prefix: str,
    name_mapping: Dict[str, str],
) -> Dict[str, Tuple[str, str]]:
    lookup: Dict[str, Tuple[str, str]] = {}

    for _, row in centroid_df.iterrows():
        cluster = str(row[cluster_column])
        closest_filename = normalize_filename(
            str(row[closest_filename_column]),
            filename_prefix,
            name_mapping,
        )

        raw_files = str(row[cluster_files_column])
        cluster_files = [
            item.strip()
            for item in raw_files.split(cluster_separator)
            if item.strip()
        ]

        for filename in cluster_files:
            normalized_name = normalize_filename(
                filename,
                filename_prefix,
                name_mapping,
            )
            lookup[normalized_name] = (
                cluster,
                closest_filename,
            )

    return lookup


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: List[str],
    csv_name: str,
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{csv_name} is missing columns: "
            f"{', '.join(missing_columns)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ir_pattern", required=True)
    parser.add_argument("--centroid_csv", required=True)
    parser.add_argument("--ppo_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--name_mapping_json")
    parser.add_argument("--max_files", type=int)

    parser.add_argument("--opt", default="opt")
    parser.add_argument("--llvm_as", default="llvm-as")
    parser.add_argument("--llvm_dis", default="llvm-dis")
    parser.add_argument("--opt_level", required=True)
    parser.add_argument(
        "--extra_opt_arg",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--assemble_timeout",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--optimization_timeout",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--analysis_timeout",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--benchmark_column",
        default="benchmark",
    )
    parser.add_argument(
        "--commandline_column",
        default="commandline",
    )
    parser.add_argument(
        "--cluster_column",
        default="cluster",
    )
    parser.add_argument(
        "--cluster_files_column",
        default="all_cluster_files",
    )
    parser.add_argument(
        "--closest_filename_column",
        default="closest_filename",
    )
    parser.add_argument(
        "--cluster_separator",
        default=", ",
    )
    parser.add_argument(
        "--benchmark_prefix",
        default="cbench-v1/",
    )
    parser.add_argument(
        "--filename_prefix",
        default="cbench-",
    )

    args = parser.parse_args()

    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max_files must be positive.")

    for argument_name in (
        "assemble_timeout",
        "optimization_timeout",
        "analysis_timeout",
    ):
        if getattr(args, argument_name) <= 0:
            parser.error(
                f"--{argument_name} must be positive."
            )

    return args


def main() -> None:
    args = parse_args()

    centroid_csv = Path(args.centroid_csv)
    ppo_csv = Path(args.ppo_csv)
    output_csv = Path(args.output_csv)
    mapping_path = (
        Path(args.name_mapping_json)
        if args.name_mapping_json
        else None
    )

    for required_path in (
        centroid_csv,
        ppo_csv,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {required_path}"
            )

    if mapping_path is not None and not mapping_path.is_file():
        raise FileNotFoundError(
            f"Name mapping file not found: {mapping_path}"
        )

    ir_files = sorted(
        Path(path)
        for path in glob.glob(
            args.ir_pattern,
            recursive=True,
        )
        if Path(path).is_file()
    )

    if args.max_files is not None:
        ir_files = ir_files[:args.max_files]

    if not ir_files:
        raise RuntimeError(
            "No IR files matched the supplied pattern: "
            f"{args.ir_pattern}"
        )

    centroid_df = pd.read_csv(centroid_csv)
    ppo_df = pd.read_csv(ppo_csv)

    validate_columns(
        centroid_df,
        [
            args.cluster_column,
            args.cluster_files_column,
            args.closest_filename_column,
        ],
        str(centroid_csv),
    )
    validate_columns(
        ppo_df,
        [
            args.benchmark_column,
            args.commandline_column,
        ],
        str(ppo_csv),
    )

    name_mapping = load_name_mapping(mapping_path)

    benchmark_commands = load_benchmark_commands(
        ppo_df=ppo_df,
        benchmark_column=args.benchmark_column,
        commandline_column=args.commandline_column,
        benchmark_prefix=args.benchmark_prefix,
        filename_prefix=args.filename_prefix,
        name_mapping=name_mapping,
    )
    cluster_lookup = build_cluster_lookup(
        centroid_df=centroid_df,
        cluster_column=args.cluster_column,
        cluster_files_column=args.cluster_files_column,
        closest_filename_column=args.closest_filename_column,
        cluster_separator=args.cluster_separator,
        filename_prefix=args.filename_prefix,
        name_mapping=name_mapping,
    )

    print(f"IR files: {len(ir_files)}")
    print(
        f"Optimization sequences: "
        f"{len(benchmark_commands)}"
    )
    print(
        f"Cluster assignments: "
        f"{len(cluster_lookup)}"
    )

    results = []

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)

        for index, ir_file in enumerate(ir_files):
            benchmark_name = ir_file.stem
            normalized_name = normalize_filename(
                benchmark_name,
                args.filename_prefix,
                name_mapping,
            )

            print(
                f"[{index + 1}/{len(ir_files)}] "
                f"{ir_file.name}"
            )

            current_count = count_ir_instructions(
                ir_path=ir_file,
                opt=args.opt,
                llvm_as=args.llvm_as,
                llvm_dis=args.llvm_dis,
                assemble_timeout=args.assemble_timeout,
                analysis_timeout=args.analysis_timeout,
            )

            input_bc = (
                temporary_path
                / f"input_{index}.bc"
            )
            input_ready = ll_to_bc(
                input_ll=ir_file,
                output_bc=input_bc,
                llvm_as=args.llvm_as,
                timeout=args.assemble_timeout,
            )

            oz_count: Optional[int] = None
            centroid_count: Optional[int] = None
            individual_count: Optional[int] = None
            centroid_sequence = ""
            individual_sequence = ""
            cluster_id = ""
            closest_filename = ""

            if input_ready:
                optimized_bc = (
                    temporary_path
                    / f"level_{index}.bc"
                )

                if optimize_with_level(
                    input_bc=input_bc,
                    output_bc=optimized_bc,
                    opt_level=args.opt_level,
                    opt=args.opt,
                    timeout=args.optimization_timeout,
                    extra_opt_args=args.extra_opt_arg,
                ):
                    oz_count = count_ir_instructions(
                        ir_path=optimized_bc,
                        opt=args.opt,
                        llvm_as=args.llvm_as,
                        llvm_dis=args.llvm_dis,
                        assemble_timeout=args.assemble_timeout,
                        analysis_timeout=args.analysis_timeout,
                    )

                cluster_info = cluster_lookup.get(
                    normalized_name
                )

                if cluster_info is not None:
                    cluster_id, closest_filename = (
                        cluster_info
                    )
                    centroid_sequence = (
                        benchmark_commands.get(
                            closest_filename,
                            "",
                        )
                    )

                    if centroid_sequence:
                        centroid_bc = (
                            temporary_path
                            / f"centroid_{index}.bc"
                        )

                        if apply_optimization_sequence(
                            input_bc=input_bc,
                            output_bc=centroid_bc,
                            sequence=centroid_sequence,
                            opt=args.opt,
                            timeout=args.optimization_timeout,
                            extra_opt_args=args.extra_opt_arg,
                        ):
                            centroid_count = (
                                count_ir_instructions(
                                    ir_path=centroid_bc,
                                    opt=args.opt,
                                    llvm_as=args.llvm_as,
                                    llvm_dis=args.llvm_dis,
                                    assemble_timeout=(
                                        args.assemble_timeout
                                    ),
                                    analysis_timeout=(
                                        args.analysis_timeout
                                    ),
                                )
                            )

                individual_sequence = (
                    benchmark_commands.get(
                        normalized_name,
                        "",
                    )
                )

                if individual_sequence:
                    individual_bc = (
                        temporary_path
                        / f"individual_{index}.bc"
                    )

                    if apply_optimization_sequence(
                        input_bc=input_bc,
                        output_bc=individual_bc,
                        sequence=individual_sequence,
                        opt=args.opt,
                        timeout=args.optimization_timeout,
                        extra_opt_args=args.extra_opt_arg,
                    ):
                        individual_count = (
                            count_ir_instructions(
                                ir_path=individual_bc,
                                opt=args.opt,
                                llvm_as=args.llvm_as,
                                llvm_dis=args.llvm_dis,
                                assemble_timeout=(
                                    args.assemble_timeout
                                ),
                                analysis_timeout=(
                                    args.analysis_timeout
                                ),
                            )
                        )

            results.append(
                {
                    "benchmark": benchmark_name,
                    "normalized_name": normalized_name,
                    "cluster": cluster_id,
                    "closest_filename": closest_filename,
                    "current_instcount": current_count,
                    "optimized_instcount": oz_count,
                    "centroid_instcount": centroid_count,
                    "individual_instcount": individual_count,
                    "centroid_sequence": centroid_sequence,
                    "individual_sequence": individual_sequence,
                }
            )

            print(
                "  current="
                f"{current_count}, "
                f"{args.opt_level}={oz_count}, "
                f"centroid={centroid_count}, "
                f"individual={individual_count}"
            )

    results_df = pd.DataFrame(results)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    results_df.to_csv(
        output_csv,
        index=False,
    )

    count_columns = [
        "current_instcount",
        "optimized_instcount",
        "centroid_instcount",
        "individual_instcount",
    ]

    print(f"Results saved to: {output_csv}")
    print(f"Files processed: {len(results_df)}")

    for column in count_columns:
        successful = int(
            results_df[column].notna().sum()
        )
        mean_value = results_df[column].mean()

        if pd.isna(mean_value):
            mean_text = "N/A"
        else:
            mean_text = f"{mean_value:.2f}"

        print(
            f"{column}: "
            f"success={successful}/{len(results_df)}, "
            f"mean={mean_text}"
        )


if __name__ == "__main__":
    main()
