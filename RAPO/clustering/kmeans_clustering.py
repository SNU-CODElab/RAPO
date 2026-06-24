#!/usr/bin/env python3
import argparse
import ast
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


Identifier = Union[int, float, str]


def detect_worker_count(requested_workers: int) -> int:
    if requested_workers > 0:
        return requested_workers

    env_value = os.environ.get("NPROC")
    if env_value:
        try:
            workers = int(env_value)
            if workers > 0:
                return workers
        except ValueError:
            pass

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def parse_embedding(value: object) -> np.ndarray:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
    elif isinstance(value, (list, tuple, np.ndarray)):
        parsed = value
    else:
        raise TypeError(
            f"Unsupported embedding value type: {type(value).__name__}"
        )

    embedding = np.asarray(parsed, dtype=np.float32)

    if embedding.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional embedding, got shape {embedding.shape}"
        )

    if embedding.size == 0:
        raise ValueError("Embedding is empty.")

    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains NaN or infinite values.")

    return embedding


def parse_identifier(value: object) -> Identifier:
    if pd.isna(value):
        raise ValueError("Identifier is missing.")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)

    text = str(value).strip()

    if not text:
        raise ValueError("Identifier is empty.")

    try:
        numeric_value = float(text)
        if numeric_value.is_integer():
            return int(numeric_value)
    except ValueError:
        pass

    return text


def load_embeddings(
    file_path: Path,
    id_column: str,
    embedding_column: Optional[str],
    invalid_row_policy: str,
) -> Tuple[np.ndarray, List[Identifier], str]:
    print(f"Loading embeddings from: {file_path}")
    dataframe = pd.read_csv(file_path)

    if id_column not in dataframe.columns:
        raise ValueError(
            f"ID column '{id_column}' was not found in the input CSV."
        )

    if embedding_column is None:
        for candidate in ("embeddings", "embedding"):
            if candidate in dataframe.columns:
                embedding_column = candidate
                break

    if embedding_column is None or embedding_column not in dataframe.columns:
        raise ValueError(
            "No embedding column was found. Specify --embedding_column "
            "or use a column named 'embedding' or 'embeddings'."
        )

    embeddings: List[np.ndarray] = []
    identifiers: List[Identifier] = []
    expected_dimension: Optional[int] = None
    skipped_rows = 0

    for row_index, row in dataframe.iterrows():
        try:
            embedding = parse_embedding(row[embedding_column])
            identifier = parse_identifier(row[id_column])

            if expected_dimension is None:
                expected_dimension = embedding.size
            elif embedding.size != expected_dimension:
                raise ValueError(
                    f"Expected dimension {expected_dimension}, "
                    f"got {embedding.size}"
                )

            embeddings.append(embedding)
            identifiers.append(identifier)

        except (ValueError, TypeError, SyntaxError) as error:
            if invalid_row_policy == "error":
                raise ValueError(
                    f"Failed to parse CSV row {row_index + 2}: {error}"
                ) from error

            skipped_rows += 1
            print(
                f"Skipping CSV row {row_index + 2}: {error}"
            )

    if not embeddings:
        raise ValueError("No valid embeddings were loaded.")

    embedding_array = np.stack(embeddings, axis=0)

    print(f"Valid embeddings: {len(identifiers)}")
    print(f"Embedding dimension: {embedding_array.shape[1]}")
    print(f"Skipped rows: {skipped_rows}")

    return embedding_array, identifiers, embedding_column


def filter_by_id_range(
    embeddings: np.ndarray,
    identifiers: Sequence[Identifier],
    minimum_id: Optional[int],
    maximum_id: Optional[int],
) -> Tuple[np.ndarray, List[Identifier]]:
    if minimum_id is None and maximum_id is None:
        return embeddings, list(identifiers)

    numeric_ids = []

    for identifier in identifiers:
        try:
            numeric_ids.append(int(identifier))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "ID range filtering requires numeric identifiers. "
                f"Invalid identifier: {identifier}"
            ) from error

    numeric_array = np.asarray(numeric_ids, dtype=np.int64)
    mask = np.ones(len(numeric_array), dtype=bool)

    if minimum_id is not None:
        mask &= numeric_array >= minimum_id

    if maximum_id is not None:
        mask &= numeric_array <= maximum_id

    filtered_embeddings = embeddings[mask]
    filtered_ids = numeric_array[mask].tolist()

    if not filtered_ids:
        raise ValueError(
            "No samples remain after ID range filtering."
        )

    print(f"Samples after ID filtering: {len(filtered_ids)}")

    return filtered_embeddings, filtered_ids


def scale_embeddings(
    embeddings: np.ndarray,
    scaling: str,
) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    if scaling == "none":
        return embeddings, None

    if scaling == "standard":
        scaler = StandardScaler()
        return scaler.fit_transform(embeddings), scaler

    raise ValueError(f"Unsupported scaling method: {scaling}")


def compute_inertia_for_k(
    k: int,
    embeddings: np.ndarray,
    random_state: int,
    n_init: int,
    max_iter: int,
    algorithm: str,
) -> Tuple[int, float]:
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        algorithm=algorithm,
    )
    model.fit(embeddings)
    return k, float(model.inertia_)


def select_elbow_k(
    k_values: Sequence[int],
    inertias: Sequence[float],
) -> int:
    if len(k_values) < 3:
        raise ValueError(
            "At least three K values are required for elbow selection."
        )

    x = np.asarray(k_values, dtype=np.float64)
    y = np.asarray(inertias, dtype=np.float64)

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    if x_range == 0 or y_range == 0:
        return int(k_values[0])

    x_normalized = (x - x.min()) / x_range
    y_normalized = (y - y.min()) / y_range

    start = np.array(
        [x_normalized[0], y_normalized[0]],
        dtype=np.float64,
    )
    end = np.array(
        [x_normalized[-1], y_normalized[-1]],
        dtype=np.float64,
    )
    line = end - start
    denominator = np.linalg.norm(line)

    if denominator == 0:
        return int(k_values[0])

    points = np.column_stack(
        [x_normalized, y_normalized]
    )
    offsets = points - start
    distances = np.abs(
        line[0] * offsets[:, 1]
        - line[1] * offsets[:, 0]
    ) / denominator

    return int(k_values[int(np.argmax(distances))])


def find_optimal_clusters(
    embeddings: np.ndarray,
    k_values: Sequence[int],
    workers: int,
    random_state: int,
    n_init: int,
    max_iter: int,
    algorithm: str,
    output_csv: Path,
    output_plot: Path,
    plot_dpi: int,
    show_plot: bool,
) -> int:
    print(
        f"Running elbow analysis for {len(k_values)} K values "
        f"with {workers} worker(s)."
    )

    if workers == 1:
        scores = [
            compute_inertia_for_k(
                k,
                embeddings,
                random_state,
                n_init,
                max_iter,
                algorithm,
            )
            for k in k_values
        ]
    else:
        with parallel_backend(
            "loky",
            inner_max_num_threads=1,
        ):
            scores = Parallel(n_jobs=workers)(
                delayed(compute_inertia_for_k)(
                    k,
                    embeddings,
                    random_state,
                    n_init,
                    max_iter,
                    algorithm,
                )
                for k in k_values
            )

    scores = sorted(scores, key=lambda item: item[0])
    ordered_k = [item[0] for item in scores]
    inertias = [item[1] for item in scores]

    score_dataframe = pd.DataFrame(
        {
            "k": ordered_k,
            "inertia": inertias,
        }
    )
    score_dataframe.to_csv(output_csv, index=False)

    selected_k = select_elbow_k(
        ordered_k,
        inertias,
    )

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        ordered_k,
        inertias,
        marker="o",
    )
    axis.axvline(
        selected_k,
        linestyle="--",
        label=f"Elbow: K={selected_k}",
    )
    axis.set_xlabel("Number of clusters (K)")
    axis.set_ylabel("Inertia")
    axis.set_title(
        f"Elbow Method (K={ordered_k[0]} to {ordered_k[-1]})"
    )
    axis.legend()
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(
        output_plot,
        dpi=plot_dpi,
        bbox_inches="tight",
    )

    if show_plot:
        plt.show()
    else:
        plt.close(figure)

    print(f"Selected K: {selected_k}")
    print(f"Elbow scores saved to: {output_csv}")
    print(f"Elbow plot saved to: {output_plot}")

    return selected_k


def perform_clustering(
    embeddings: np.ndarray,
    identifiers: Sequence[Identifier],
    n_clusters: int,
    random_state: int,
    n_init: int,
    max_iter: int,
    algorithm: str,
) -> Tuple[pd.DataFrame, KMeans]:
    print(f"Running KMeans with K={n_clusters}.")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        algorithm=algorithm,
    )
    labels = model.fit_predict(embeddings)
    distances = model.transform(embeddings)

    results = pd.DataFrame(
        {
            "id": list(identifiers),
            "cluster": labels,
            "distance_to_centroid": np.min(
                distances,
                axis=1,
            ),
        }
    )

    return results, model


def analyze_clusters(
    results: pd.DataFrame,
    n_clusters: int,
    top_n: int,
) -> pd.DataFrame:
    analysis_rows = []

    for cluster_id in range(n_clusters):
        cluster_data = results[
            results["cluster"] == cluster_id
        ]

        if cluster_data.empty:
            continue

        closest_rows = cluster_data.nsmallest(
            top_n,
            "distance_to_centroid",
        )
        closest_id = closest_rows.iloc[0]["id"]
        closest_ids = closest_rows["id"].tolist()
        all_ids = cluster_data["id"].tolist()

        analysis_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(cluster_data),
                "closest_code_id": closest_id,
                f"top_{top_n}_closest_ids": ", ".join(
                    map(str, closest_ids)
                ),
                "all_code_ids": ", ".join(
                    map(str, all_ids)
                ),
            }
        )

    return pd.DataFrame(analysis_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    parser.add_argument(
        "--id_column",
        default="id",
    )
    parser.add_argument(
        "--embedding_column",
    )
    parser.add_argument(
        "--invalid_row_policy",
        choices=["error", "skip"],
        default="error",
    )

    parser.add_argument(
        "--id_min",
        type=int,
    )
    parser.add_argument(
        "--id_max",
        type=int,
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
    )
    parser.add_argument(
        "--k_min",
        type=int,
    )
    parser.add_argument(
        "--k_max",
        type=int,
    )
    parser.add_argument(
        "--k_step",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--scaling",
        choices=["none", "standard"],
        required=True,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--n_init",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--algorithm",
        choices=["lloyd", "elkan"],
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Use 0 to detect the worker count automatically.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--results_filename",
    )
    parser.add_argument(
        "--analysis_filename",
    )
    parser.add_argument(
        "--elbow_scores_filename",
        default="elbow_scores.csv",
    )
    parser.add_argument(
        "--elbow_plot_filename",
        default="elbow_analysis.png",
    )
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
    )

    args = parser.parse_args()

    if args.id_min is not None and args.id_max is not None:
        if args.id_min > args.id_max:
            parser.error("--id_min cannot exceed --id_max.")

    if args.n_clusters is None:
        if args.k_min is None or args.k_max is None:
            parser.error(
                "Specify --n_clusters or both --k_min and --k_max."
            )
    elif args.k_min is not None or args.k_max is not None:
        parser.error(
            "--n_clusters cannot be combined with --k_min or --k_max."
        )

    positive_arguments = {
        "n_clusters": args.n_clusters,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "k_step": args.k_step,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "top_n": args.top_n,
        "plot_dpi": args.plot_dpi,
    }

    for name, value in positive_arguments.items():
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive.")

    if args.k_min is not None and args.k_max is not None:
        if args.k_min > args.k_max:
            parser.error("--k_min cannot exceed --k_max.")

    if args.workers < 0:
        parser.error("--workers cannot be negative.")

    return args


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    if not input_csv.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings, identifiers, embedding_column = (
        load_embeddings(
            file_path=input_csv,
            id_column=args.id_column,
            embedding_column=args.embedding_column,
            invalid_row_policy=args.invalid_row_policy,
        )
    )

    embeddings, identifiers = filter_by_id_range(
        embeddings=embeddings,
        identifiers=identifiers,
        minimum_id=args.id_min,
        maximum_id=args.id_max,
    )

    processed_embeddings, scaler = scale_embeddings(
        embeddings,
        args.scaling,
    )

    sample_count = processed_embeddings.shape[0]

    if args.n_clusters is not None:
        selected_k = args.n_clusters

        if selected_k > sample_count:
            raise ValueError(
                f"--n_clusters ({selected_k}) cannot exceed "
                f"the sample count ({sample_count})."
            )
    else:
        k_values = list(
            range(
                args.k_min,
                args.k_max + 1,
                args.k_step,
            )
        )

        k_values = [
            k
            for k in k_values
            if k <= sample_count
        ]

        if len(k_values) < 3:
            raise ValueError(
                "The valid K range must contain at least three values "
                "that do not exceed the sample count."
            )

        workers = detect_worker_count(
            args.workers
        )
        workers = min(
            workers,
            len(k_values),
        )

        selected_k = find_optimal_clusters(
            embeddings=processed_embeddings,
            k_values=k_values,
            workers=workers,
            random_state=args.random_state,
            n_init=args.n_init,
            max_iter=args.max_iter,
            algorithm=args.algorithm,
            output_csv=(
                output_dir
                / args.elbow_scores_filename
            ),
            output_plot=(
                output_dir
                / args.elbow_plot_filename
            ),
            plot_dpi=args.plot_dpi,
            show_plot=args.show_plot,
        )

    results, model = perform_clustering(
        embeddings=processed_embeddings,
        identifiers=identifiers,
        n_clusters=selected_k,
        random_state=args.random_state,
        n_init=args.n_init,
        max_iter=args.max_iter,
        algorithm=args.algorithm,
    )

    analysis = analyze_clusters(
        results=results,
        n_clusters=selected_k,
        top_n=args.top_n,
    )

    results_filename = (
        args.results_filename
        or f"clustering_results_{selected_k}_clusters.csv"
    )
    analysis_filename = (
        args.analysis_filename
        or f"cluster_analysis_{selected_k}_clusters.csv"
    )

    results_path = output_dir / results_filename
    analysis_path = output_dir / analysis_filename

    results.to_csv(
        results_path,
        index=False,
    )
    analysis.to_csv(
        analysis_path,
        index=False,
    )

    if args.save_model:
        joblib.dump(
            model,
            output_dir / "kmeans_model.joblib",
        )

        if scaler is not None:
            joblib.dump(
                scaler,
                output_dir / "scaler.joblib",
            )

    print(f"Embedding column: {embedding_column}")
    print(f"Selected clusters: {selected_k}")
    print(f"Clustering results saved to: {results_path}")
    print(f"Cluster analysis saved to: {analysis_path}")

    print("Cluster summary:")

    top_column = f"top_{args.top_n}_closest_ids"

    for _, row in analysis.iterrows():
        print(f"Cluster {row['cluster_id']}: {row['cluster_size']} samples")
        print("  Closest ID: {row['closest_code_id']}")
        print("  Top {args.top_n} IDs: {row[top_column]}")


if __name__ == "__main__":
    main()
