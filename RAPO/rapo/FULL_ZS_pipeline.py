import argparse
import ast
import math
import os
import shlex
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

import ctypes
import ctypes.util

_LIBLLVM_HANDLE = None
_LIBLLVM_LOAD_FAILED = False
_LIBLLVM_WARNED = False


def _load_libllvm():
    global _LIBLLVM_HANDLE, _LIBLLVM_LOAD_FAILED
    if _LIBLLVM_HANDLE or _LIBLLVM_LOAD_FAILED:
        return _LIBLLVM_HANDLE

    candidates = []
    configured_path = os.environ.get("LIBLLVM_PATH")
    if configured_path:
        candidates.append(configured_path)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "lib", "libLLVM-10.so"))
    llvm10 = ctypes.util.find_library("LLVM-10")
    if llvm10:
        candidates.append(llvm10)
    candidates.append("libLLVM-10.so")

    for cand in candidates:
        if not cand:
            continue
        try:
            lib = ctypes.CDLL(cand)
        except OSError:
            continue

        c_void_p = ctypes.c_void_p
        c_char_p = ctypes.c_char_p
        c_int = ctypes.c_int

        lib.LLVMContextCreate.restype = c_void_p
        lib.LLVMContextDispose.argtypes = [c_void_p]
        lib.LLVMContextDispose.restype = None
        lib.LLVMDisposeMessage.argtypes = [c_char_p]
        lib.LLVMDisposeMessage.restype = None

        lib.LLVMCreateMemoryBufferWithContentsOfFile.argtypes = [
            c_char_p,
            ctypes.POINTER(c_void_p),
            ctypes.POINTER(c_char_p),
        ]
        lib.LLVMCreateMemoryBufferWithContentsOfFile.restype = c_int
        lib.LLVMDisposeMemoryBuffer.argtypes = [c_void_p]
        lib.LLVMDisposeMemoryBuffer.restype = None

        lib.LLVMParseBitcodeInContext2.argtypes = [
            c_void_p,
            c_void_p,
            ctypes.POINTER(c_void_p),
        ]
        lib.LLVMParseBitcodeInContext2.restype = c_int
        lib.LLVMDisposeModule.argtypes = [c_void_p]
        lib.LLVMDisposeModule.restype = None

        lib.LLVMGetFirstFunction.argtypes = [c_void_p]
        lib.LLVMGetFirstFunction.restype = c_void_p
        lib.LLVMGetNextFunction.argtypes = [c_void_p]
        lib.LLVMGetNextFunction.restype = c_void_p

        lib.LLVMGetFirstBasicBlock.argtypes = [c_void_p]
        lib.LLVMGetFirstBasicBlock.restype = c_void_p
        lib.LLVMGetNextBasicBlock.argtypes = [c_void_p]
        lib.LLVMGetNextBasicBlock.restype = c_void_p

        lib.LLVMGetFirstInstruction.argtypes = [c_void_p]
        lib.LLVMGetFirstInstruction.restype = c_void_p
        lib.LLVMGetNextInstruction.argtypes = [c_void_p]
        lib.LLVMGetNextInstruction.restype = c_void_p

        _LIBLLVM_HANDLE = lib
        return _LIBLLVM_HANDLE

    _LIBLLVM_LOAD_FAILED = True
    return None


def _count_with_libllvm(bitcode_path: Path):
    global _LIBLLVM_WARNED
    lib = _load_libllvm()
    if not lib:
        if not _LIBLLVM_WARNED:
            print("[instcount] libLLVM shared library not found; skipping direct bitcode count.")
            _LIBLLVM_WARNED = True
        return None

    bt = os.fspath(bitcode_path)
    ctx = lib.LLVMContextCreate()
    if not ctx:
        return None

    buf_ref = ctypes.c_void_p()
    err_ptr = ctypes.c_char_p()
    res = lib.LLVMCreateMemoryBufferWithContentsOfFile(
        bt.encode(),
        ctypes.byref(buf_ref),
        ctypes.byref(err_ptr),
    )
    if res != 0:
        msg = "<unknown>"
        if err_ptr and err_ptr.value:
            try:
                msg = err_ptr.value.decode(errors="ignore")
            except Exception:
                msg = "<unknown>"
        print(f"[instcount] libLLVM buffer load failed: {msg}")
        if err_ptr:
            lib.LLVMDisposeMessage(err_ptr)
        lib.LLVMContextDispose(ctx)
        return None

    module_ref = ctypes.c_void_p()
    parse_res = lib.LLVMParseBitcodeInContext2(ctx, buf_ref, ctypes.byref(module_ref))
    if parse_res != 0 or not module_ref:
        print(f"[instcount] libLLVM could not parse bitcode (code={parse_res}).")
        lib.LLVMDisposeMemoryBuffer(buf_ref)
        lib.LLVMContextDispose(ctx)
        return None

    total = 0
    try:
        fn = lib.LLVMGetFirstFunction(module_ref)
        while fn:
            bb = lib.LLVMGetFirstBasicBlock(fn)
            while bb:
                inst = lib.LLVMGetFirstInstruction(bb)
                while inst:
                    total += 1
                    inst = lib.LLVMGetNextInstruction(inst)
                bb = lib.LLVMGetNextBasicBlock(bb)
            fn = lib.LLVMGetNextFunction(fn)
        return total
    finally:
        lib.LLVMDisposeModule(module_ref)
        lib.LLVMDisposeMemoryBuffer(buf_ref)
        lib.LLVMContextDispose(ctx)


def count_ir_instr(ir_path: Path) -> int:
    temp_bc = None
    try:
        input_bc = ir_path
        if ir_path.suffix == ".ll":
            with tempfile.NamedTemporaryFile(suffix=".bc", delete=False) as tmp:
                temp_bc = Path(tmp.name)

            r = subprocess.run(
                ["llvm-as", str(ir_path), "-o", str(temp_bc)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r.returncode != 0:
                print(f"[instcount] llvm-as failed: {r.stderr}")
                return 0
            input_bc = temp_bc

        direct_cnt = _count_with_libllvm(input_bc)
        if direct_cnt is not None:
            return direct_cnt
    except Exception as e:
        print(f"[instcount] error: {e}")
        return 0
    finally:
        if temp_bc and temp_bc.exists():
            try:
                temp_bc.unlink()
            except Exception:
                pass

    return 0


# CompilerGym (guarded import)
try:
    import gym  # noqa: F401
    from compiler_gym.envs.llvm import make_benchmark
except Exception:  # pragma: no cover
    gym = None
    make_benchmark = None


def apply_optimization_sequence(input_bc: Path, output_bc: Path, sequence: str) -> bool:
    try:
        if gym is None or make_benchmark is None:
            print("[cgym] CompilerGym not available: cannot apply sequence via env")
            return False

        seq = sequence.replace(" input.bc -o output.bc", "").strip()
        if not seq:
            return False

        raw_tokens = shlex.split(seq)
        if raw_tokens and Path(raw_tokens[0]).name.startswith("opt"):
            raw_tokens = raw_tokens[1:]

        def _looks_like_pass_flag(tok: str) -> bool:
            if not tok or not tok.startswith("-"):
                return False
            i = 0
            n = len(tok)
            while i < n and tok[i] == "-":
                i += 1
            if i >= n:
                return False
            return tok[i].isalpha()

        pass_tokens = []
        for t in raw_tokens:
            if _looks_like_pass_flag(t):
                pass_tokens.append(t.lstrip("-"))
        if not pass_tokens:
            return False

        with gym.make("llvm-v0") as env:
            bm = make_benchmark(str(input_bc))
            env.reset(benchmark=bm)

            try:
                names = list(env.action_space.names)  # type: ignore[attr-defined]
            except Exception:
                names = []

            name2idx = {}
            for idx, nm in enumerate(names):
                key = nm.lstrip("-").lower()
                name2idx[key] = idx

            actions = []
            missing = []
            for p in pass_tokens:
                key = p.lstrip("-").lower()
                if key in name2idx:
                    actions.append(name2idx[key])
                elif ("-" + key) in name2idx:
                    actions.append(name2idx["-" + key])
                else:
                    missing.append(p)

            if missing:
                print(f"[cgym] Warning: {len(missing)} passes not found in action space and will be skipped: {missing}")

            for a in actions:
                env.step(a)

            wrote = False
            if hasattr(env, "write_bitcode"):
                try:
                    env.write_bitcode(str(output_bc))  # type: ignore[attr-defined]
                    wrote = Path(output_bc).exists()
                except Exception:
                    wrote = False

            if not wrote and hasattr(env, "_env") and hasattr(env._env, "write_bitcode"):
                try:
                    env._env.write_bitcode(str(output_bc))  # type: ignore[attr-defined]
                    wrote = Path(output_bc).exists()
                except Exception:
                    wrote = False

            if not wrote:
                print("[cgym] Failed to write optimized bitcode:", output_bc)
            return wrote
    except Exception as e:
        print(f"[cgym] Error applying sequence via CompilerGym: {e}")
        return False


SEQ_BASE_DIR = Path("sequence")
SOURCE_ROOT = Path(os.environ.get("RAPO_SOURCE_ROOT", "."))


def _source_dir_for_id(sid: Union[str, int]) -> Path:
    sid = str(sid)
    source_root = Path(
        os.environ.get("RAPO_SOURCE_ROOT", os.fspath(SOURCE_ROOT))
    )
    return source_root / f"Source_{sid}"


def _oz_ll_path(sid: Union[str, int]) -> Path:
    return _source_dir_for_id(sid) / "llvm_OZ.ll"


def _o0_ll_path(sid: Union[str, int]) -> Path:
    return _source_dir_for_id(sid) / "llvm_O0.ll"


def _input_bc_path(sid: Union[str, int]) -> Path:
    return _source_dir_for_id(sid) / "input.bc"


def read_sequence_text_for_code_id(code_id: Union[str, int], base_dir: Union[str, Path] = SEQ_BASE_DIR) -> Optional[str]:
    p = Path(base_dir) / f"{code_id}_seq.txt"
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        try:
            return p.read_text(errors="ignore")
        except Exception:
            return None


def ensure_bitcode_for_sample(sid: Union[str, int]) -> Optional[Path]:
    sid = str(sid)
    bc = _input_bc_path(sid)
    if bc.exists():
        return bc

    ll = _o0_ll_path(sid)
    if ll.exists():
        tmp_bc = ll.with_suffix(".tmp.bc")
        try:
            r = subprocess.run(["llvm-as", str(ll), "-o", str(tmp_bc)], capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and tmp_bc.exists():
                return tmp_bc
        except Exception:
            pass
    return None


def count_oz_for_sample(sid: Union[str, int]) -> Optional[int]:
    ll = _oz_ll_path(sid)
    if not ll.exists():
        return None
    try:
        return count_ir_instr(ll)
    except Exception:
        return None


def count_o0_for_sample(sid: Union[str, int]) -> Optional[int]:
    ll = _o0_ll_path(sid)
    if not ll.exists():
        return None
    try:
        return count_ir_instr(ll)
    except Exception:
        return None


def compile_source_c_with_clang_oz(sid: Union[str, int]) -> Tuple[Optional[int], float]:
    src = _source_dir_for_id(sid) / "Source.c"
    if not src.exists():
        return None, 0.0
    out_ll = src.with_suffix(".oz.tmp.ll")
    t0 = time.perf_counter()
    try:
        if out_ll.exists():
            try:
                out_ll.unlink()
            except Exception:
                pass
        r = subprocess.run(
            ["clang", "-Oz", "-emit-llvm", "-S", str(src), "-o", str(out_ll)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if r.returncode != 0 or not out_ll.exists():
            print(f"[clang -Oz] failed for {sid}: {r.stderr.strip() if r.stderr else r.returncode}")
            return None, elapsed_ms
        cnt = count_ir_instr(out_ll)
        return cnt, elapsed_ms
    finally:
        try:
            if out_ll.exists():
                out_ll.unlink()
        except Exception:
            pass


def compile_o0_with_cgym_oz(sid: Union[str, int]) -> Tuple[Optional[int], float, bool]:
    inp = ensure_bitcode_for_sample(sid)
    if inp is None:
        return None, 0.0, False
    out_bc = inp.with_suffix(".cg_oz.tmp.bc")
    if out_bc.exists():
        try:
            out_bc.unlink()
        except Exception:
            pass

    t0 = time.perf_counter()
    used_cgym = False
    try:
        ok = apply_optimization_sequence(inp, out_bc, "-Oz")
        used_cgym = ok
        if not ok:
            try:
                r = subprocess.run(["opt", "-Oz", str(inp), "-o", str(out_bc)], capture_output=True, text=True, timeout=600)
                ok = r.returncode == 0 and out_bc.exists()
                if not ok:
                    print(f"[opt -Oz] failed for {sid}: {r.stderr.strip() if r.stderr else r.returncode}")
            except Exception as e:
                print(f"[opt -Oz] error for {sid}: {e}")
                ok = False
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not ok or not out_bc.exists():
            return None, elapsed_ms, used_cgym
        cnt = count_ir_instr(out_bc)
        return cnt, elapsed_ms, used_cgym
    finally:
        try:
            if out_bc.exists():
                out_bc.unlink()
        except Exception:
            pass


def geomean_ratio(ratios: Iterable[float]) -> Optional[float]:
    vals = [r for r in ratios if r > 0 and math.isfinite(r)]
    if not vals:
        return None
    s = sum(math.log(v) for v in vals) / len(vals)
    return math.exp(s)



def normalize_id(value) -> str:
    if pd.isna(value):
        raise ValueError("ID is missing.")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def load_embeddings_df(file_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    emb_col = "embeddings" if "embeddings" in df.columns else "embedding"
    if emb_col not in df.columns:
        raise ValueError(
            "The CSV must contain an 'embeddings' or 'embedding' column."
        )
    if "id" not in df.columns:
        raise ValueError("The CSV must contain an 'id' column.")

    df = df.copy()
    df["id"] = df["id"].apply(normalize_id)
    df["embedding"] = df[emb_col].apply(
        lambda x: np.array(ast.literal_eval(x), dtype=float)
    )
    return df[["id", "embedding"]]


def build_centroid_embedding_df(cluster_csv: Union[str, Path], embeddings_df: pd.DataFrame) -> pd.DataFrame:
    cluster_df = pd.read_csv(cluster_csv)
    cluster_col = None
    for cand in ["cluster_id", "cluster", "centroid"]:
        if cand in cluster_df.columns:
            cluster_col = cand
            break
    if cluster_col is None:
        raise ValueError("cluster_analysis csv must have 'cluster_id'/'cluster'/'centroid'")
    if "closest_code_id" not in cluster_df.columns:
        raise ValueError("cluster_analysis csv missing 'closest_code_id'")

    id2emb: Dict[str, np.ndarray] = {row.id: row.embedding for row in embeddings_df.itertuples()}
    rows = []
    missing = []
    for _, r in cluster_df.iterrows():
        cid = int(r[cluster_col])
        code_id = normalize_id(r["closest_code_id"])
        emb = id2emb.get(code_id, None)
        if emb is None:
            missing.append(code_id)
            continue
        rows.append({"cluster_id": cid, "closest_code_id": code_id, "embedding": emb})
    if missing:
        print(f"[warn] centroid embeddings missing for {len(missing)} ids: {missing}")
    return pd.DataFrame(rows, columns=["cluster_id", "closest_code_id", "embedding"])


def fit_scaler_and_kmeans(
    base_embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> Tuple[StandardScaler, KMeans, float]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(base_embeddings)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    t0 = time.perf_counter()
    kmeans.fit(scaled)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return scaler, kmeans, elapsed_ms


def assign_samples_to_centroids(
    sample_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    scaler: StandardScaler,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, np.ndarray]:
    centroid_arr = np.vstack(centroids_df["embedding"].values)
    sample_arr = np.vstack(sample_df["embedding"].values)

    t0 = time.perf_counter()
    centroid_scaled = scaler.transform(centroid_arr)
    sample_scaled = scaler.transform(sample_arr)
    dist_matrix = pairwise_distances(sample_scaled, centroid_scaled, metric="euclidean")
    assigned_positions = dist_matrix.argmin(axis=1)
    cluster_ids = centroids_df["cluster_id"].to_numpy(dtype=int)
    assigned_clusters = cluster_ids[assigned_positions]
    min_dists = dist_matrix[
        np.arange(len(sample_df)),
        assigned_positions,
    ]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    sample_assign = pd.DataFrame(
        {
            "id": sample_df["id"].values,
            "cluster": assigned_clusters,
            "distance": min_dists,
        }
    )
    rows = []
    for _, row in centroids_df.iterrows():
        cid = int(row["cluster_id"])
        samples = sample_assign[sample_assign["cluster"] == cid]["id"].tolist()
        rows.append(
            {
                "cluster_id": cid,
                "closest_code_id": str(row["closest_code_id"]),
                "assigned_samples": ",".join(samples),
            }
        )
    cluster_to_samples = pd.DataFrame(rows, columns=["cluster_id", "closest_code_id", "assigned_samples"])
    return sample_assign, cluster_to_samples, elapsed_ms, dist_matrix


def maybe_write_top5(
    dist_matrix: np.ndarray,
    sample_ids: List[str],
    cluster_ids: List[int],
    out_path: Union[str, Path],
) -> pd.DataFrame:
    idx_sorted = np.argsort(dist_matrix, axis=1)[:, :5]
    dists_sorted = np.take_along_axis(dist_matrix, idx_sorted, axis=1)
    cluster_ids_array = np.asarray(cluster_ids, dtype=int)

    rows = []
    for i, sid in enumerate(sample_ids):
        nearest_clusters = cluster_ids_array[idx_sorted[i]]
        clusters_str = ",".join(
            str(int(c)) for c in nearest_clusters.tolist()
        )
        dists_str = ",".join(
            f"{float(d):.6f}" for d in dists_sorted[i].tolist()
        )
        rows.append(
            {
                "id": sid,
                "top_5_closest_clusters": clusters_str,
                "top_5_distances": dists_str,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "top_5_closest_clusters",
            "top_5_distances",
        ],
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[save] Top-5 nearest clusters -> {out_path}")
    return df


def apply_seq_count(input_bc: Path, seq_txt: str) -> Tuple[Optional[int], float]:
    tmp = tempfile.NamedTemporaryFile(suffix=".best.tmp.bc", delete=False)
    out_bc = Path(tmp.name)
    tmp.close()
    compile_ms: float = 0.0
    try:
        t0 = time.perf_counter()
        ok = apply_optimization_sequence(input_bc, out_bc, seq_txt)
        compile_ms = (time.perf_counter() - t0) * 1000.0
        if not ok or not out_bc.exists():
            return None, compile_ms
        cnt = count_ir_instr(out_bc)
        return cnt, compile_ms
    finally:
        try:
            if out_bc.exists():
                out_bc.unlink()
        except Exception:
            pass


def _apply_single_seq(input_bc: Path, seq_txt: str, target_cluster: int) -> Tuple[int, Optional[float], float]:
    try:
        cnt, comp_ms = apply_seq_count(input_bc, seq_txt)
    except Exception:
        cnt, comp_ms = None, 0.0
    return target_cluster, cnt, comp_ms


def _worker_apply_centroid(
    sid: str,
    cluster_id: int,
    cluster2code: Dict[int, str],
    seq_base_dir: Union[str, Path],
) -> Dict:
    oz = count_oz_for_sample(sid)
    code_id = cluster2code.get(int(cluster_id), None)
    centroid_cnt = None
    compile_ms = 0.0
    if code_id is not None:
        seq_txt = read_sequence_text_for_code_id(code_id, base_dir=seq_base_dir)
        inp = ensure_bitcode_for_sample(sid)
        if seq_txt and inp is not None:
            centroid_cnt, compile_ms = apply_seq_count(inp, seq_txt)

    ratio = None
    reduction_pct = None
    if oz and centroid_cnt is not None and oz > 0:
        ratio = centroid_cnt / float(oz)
        reduction_pct = (1.0 - ratio) * 100.0

    return {
        "id": sid,
        "cluster": int(cluster_id),
        "centroid_code_id": code_id if code_id is not None else "",
        "oz_count": oz if oz is not None else "",
        "centroid_count": centroid_cnt if centroid_cnt is not None else "",
        "reduction_pct": round(reduction_pct, 6) if reduction_pct is not None else "",
        "_ratio": ratio if ratio is not None else None,
        "compile_ms": compile_ms,
    }


def run_centroid_application(
    assignments_df: pd.DataFrame,
    cluster2code: Dict[int, str],
    seq_base_dir: Union[str, Path],
    out_csv: Union[str, Path],
    workers: int = 1,
) -> Tuple[pd.DataFrame, Optional[float], float]:
    rows: List[Dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_worker_apply_centroid, str(row.id), int(row.cluster), cluster2code, seq_base_dir): str(row.id)
            for row in assignments_df.itertuples()
        }
        for fut in as_completed(futs):
            rows.append(fut.result())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    rows.sort(key=lambda d: int(d["id"]))
    df = pd.DataFrame(rows, columns=["id", "cluster", "centroid_code_id", "oz_count", "centroid_count", "reduction_pct", "compile_ms"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    ratios = [r["_ratio"] for r in rows if r.get("_ratio") is not None]
    gm = geomean_ratio(ratios)
    return df, gm, elapsed_ms


def normalize_bool_col(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    v = s.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "yes": True, "y": True, "t": True, "false": False, "0": False, "no": False, "n": False, "f": False}
    return v.map(mapping)


def geomean_reduction_pct_from_counts(cnt_series: pd.Series, oz_series: pd.Series) -> Optional[float]:
    ratios = []
    for c, o in zip(pd.to_numeric(cnt_series, errors="coerce"), pd.to_numeric(oz_series, errors="coerce")):
        if pd.notna(c) and pd.notna(o) and o > 0 and c > 0:
            ratios.append(c / float(o))
    gm = geomean_ratio(ratios)
    if gm is None:
        return None
    return (1.0 - gm) * 100.0


def geomean_from_pct_series(series: pd.Series) -> Optional[float]:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    ratios = 1.0 - (vals / 100.0)
    ratios = ratios[(ratios > 0) & np.isfinite(ratios)]
    if len(ratios) == 0:
        return None
    gm = math.exp(np.log(ratios).mean())
    return (1.0 - gm) * 100.0


def _fallback_worker(
    row: Dict,
    cl2code: Dict[int, str],
    p10_map: Dict[str, float],
    seq_base_dir: Union[str, Path],
    inner_threads: int,
) -> Dict:
    sid = str(row["id"])
    assigned_cluster = int(row["cluster"])
    oz = pd.to_numeric(row.get("oz_count"), errors="coerce")
    base_cnt = pd.to_numeric(row.get("centroid_count"), errors="coerce")
    base_cnt = base_cnt if pd.notna(base_cnt) else None

    if not (pd.notna(oz) and oz > 0):
        oz = count_oz_for_sample(sid)
    top5 = str(row.get("top_5_closest_clusters", "") or "")
    clusters = [c.strip() for c in top5.split(",") if c.strip().isdigit()]
    extras = clusters[1:5] if len(clusters) >= 5 else clusters[1:]

    inp = ensure_bitcode_for_sample(sid)
    extra_results = []
    total_compile_ms = 0.0
    best_cluster = assigned_cluster
    best_cnt = base_cnt

    # Parallelize the extra sequences per sample with a fixed 4-thread pool.
    if inp is not None and len(extras) > 0:
        tasks = []
        with ThreadPoolExecutor(max_workers=min(inner_threads, len(extras))) as tex:
            for c in extras:
                try:
                    cid = int(c)
                except Exception:
                    continue
                code_id = cl2code.get(cid, None)
                if not code_id:
                    extra_results.append((cid, np.nan, 0.0))
                    continue
                seq_txt = read_sequence_text_for_code_id(code_id, base_dir=seq_base_dir)
                if not seq_txt:
                    extra_results.append((cid, np.nan, 0.0))
                    continue
                tasks.append(tex.submit(_apply_single_seq, inp, seq_txt, cid))
            for fut in tasks:
                try:
                    cid, cnt, comp_ms = fut.result()
                except Exception:
                    continue
                total_compile_ms += float(comp_ms)
                cnt = cnt if cnt is not None else np.nan
                extra_results.append((cid, cnt, comp_ms))
                if pd.notna(cnt) and cnt > 0 and (best_cnt is None or cnt < best_cnt):
                    best_cnt = float(cnt)
                    best_cluster = cid

    best_pct = None
    if pd.notna(oz) and oz and best_cnt and best_cnt > 0:
        best_pct = (1.0 - (best_cnt / float(oz))) * 100.0

    thr = p10_map.get(str(best_cluster), np.nan)
    best_below_p10 = bool(pd.notna(best_pct) and pd.notna(thr) and best_pct < thr)

    return {
        "id": sid,
        "best_cluster": best_cluster,
        "best_count": best_cnt if best_cnt is not None else np.nan,
        "best_reduction_pct": best_pct if best_pct is not None else np.nan,
        "best_below_p10": best_below_p10,
        "extras_compile_ms_total": total_compile_ms,
    }


def run_fallback_stage(
    base_results: pd.DataFrame,
    top5_csv: Union[str, Path],
    cluster_p10_csv: Union[str, Path],
    cl2code: Dict[int, str],
    seq_base_dir: Union[str, Path],
    workers: int = 4,
    inner_threads: int = 4,
) -> Tuple[pd.DataFrame, Dict]:
    cluster_p10 = pd.read_csv(cluster_p10_csv)
    cluster_p10["cluster"] = cluster_p10["cluster"].astype(int)
    p10_map: Dict[str, float] = {str(int(r["cluster"])): float(r["delta_p10"]) for _, r in cluster_p10.iterrows() if pd.notna(r["delta_p10"])}

    merged = base_results.merge(cluster_p10, how="left", left_on="cluster", right_on="cluster")
    merged["delta_p10"] = pd.to_numeric(merged["delta_p10"], errors="coerce")
    merged["reduction_pct"] = pd.to_numeric(merged["reduction_pct"], errors="coerce")
    merged["below_p10"] = merged["reduction_pct"] < merged["delta_p10"]
    base_with_flags = merged[["id", "below_p10"]]

    ncs = merged[merged["below_p10"] == True].copy()
    ncs_count = len(ncs)
    gm_ncs = geomean_reduction_pct_from_counts(ncs["centroid_count"], ncs["oz_count"]) if ncs_count > 0 else None

    top5 = pd.read_csv(top5_csv, dtype={"id": str})
    ncs = ncs.merge(top5, on="id", how="left")

    fallback_rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _fallback_worker,
                r._asdict() if hasattr(r, "_asdict") else r.to_dict(),
                cl2code,
                p10_map,
                seq_base_dir,
                inner_threads,
            ): str(r.id)
            for r in ncs.itertuples()
        }
        for fut in as_completed(futs):
            fallback_rows.append(fut.result())
    fallback_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    fallback_df = pd.DataFrame(
        fallback_rows,
        columns=[
            "id",
            "best_cluster",
            "best_count",
            "best_reduction_pct",
            "best_below_p10",
            "extras_compile_ms_total",
        ],
    )
    ncs = ncs.merge(fallback_df, on="id", how="left")
    gm_ncs_best = geomean_reduction_pct_from_counts(ncs["best_count"], ncs["oz_count"]) if ncs_count > 0 else None

    preset_mask = ncs["best_below_p10"] == True
    preset_count = int(preset_mask.sum())
    preset_ids = ncs.loc[preset_mask, "id"].astype(str).tolist()
    preset_data: Dict[str, Dict] = {}
    preset_t0 = time.perf_counter()
    total_clang_ms = 0.0
    total_cgym_ms = 0.0
    if preset_ids:
        print(f"[preset] compiling {len(preset_ids)} preset-fallback samples sequentially (clang -Oz, CompilerGym/opt -Oz)")
    for sid in preset_ids:
        o0_cnt = count_o0_for_sample(sid)
        clang_cnt, clang_ms = compile_source_c_with_clang_oz(sid)
        cg_cnt, cg_ms, used_cgym = compile_o0_with_cgym_oz(sid)
        total_clang_ms += clang_ms
        total_cgym_ms += cg_ms
        oz_candidates = [cnt for cnt in [clang_cnt, cg_cnt, count_oz_for_sample(sid)] if cnt is not None and cnt > 0]
        best_oz = min(oz_candidates) if oz_candidates else None
        preset_data[sid] = {
            "o0_count": o0_cnt,
            "clang_oz_count": clang_cnt,
            "clang_oz_ms": clang_ms,
            "cg_oz_count": cg_cnt,
            "cg_oz_ms": cg_ms,
            "cg_oz_used_cgym": used_cgym,
            "best_oz_count": best_oz,
        }
    preset_elapsed_ms = (time.perf_counter() - preset_t0) * 1000.0
    preset_rows = [
        {
            "id": sid,
            "preset_o0_count": data.get("o0_count", ""),
            "preset_clang_oz_count": data.get("clang_oz_count", ""),
            "preset_clang_oz_ms": data.get("clang_oz_ms", 0.0),
            "preset_cgym_oz_count": data.get("cg_oz_count", ""),
            "preset_cgym_oz_ms": data.get("cg_oz_ms", 0.0),
            "preset_best_oz_count": data.get("best_oz_count", ""),
        }
        for sid, data in preset_data.items()
    ]
    preset_df = pd.DataFrame(preset_rows)

    # Finalize counts for all 5k samples
    final = base_results.copy()
    final = final.merge(
        ncs[["id", "best_cluster", "best_count", "best_reduction_pct", "best_below_p10", "below_p10"]],
        on="id",
        how="left",
    )
    if not preset_df.empty:
        final = final.merge(preset_df, on="id", how="left")
    final = final.merge(base_with_flags, on="id", how="left", suffixes=("", "_baseflag"))
    final["below_p10"] = normalize_bool_col(final["below_p10"].combine_first(final["below_p10_baseflag"])).fillna(False)
    final["best_below_p10"] = normalize_bool_col(final["best_below_p10"]).fillna(False)

    def _choose(row):
        oz = pd.to_numeric(row["oz_count"], errors="coerce")
        oz = oz if pd.notna(oz) else None
        centroid_cnt = pd.to_numeric(row["centroid_count"], errors="coerce")
        centroid_cnt = centroid_cnt if pd.notna(centroid_cnt) else None
        best_cnt = pd.to_numeric(row["best_count"], errors="coerce")
        best_cnt = best_cnt if pd.notna(best_cnt) else None
        base_pct = pd.to_numeric(row["reduction_pct"], errors="coerce")
        base_pct = base_pct if pd.notna(base_pct) else None
        best_pct = pd.to_numeric(row.get("best_reduction_pct"), errors="coerce")
        best_pct = best_pct if pd.notna(best_pct) else None
        below = bool(row.get("below_p10", False))
        best_below = bool(row.get("best_below_p10", False))
        category = "centroid"
        final_cluster = row["cluster"]
        final_count = centroid_cnt
        final_reduction_pct = base_pct

        if below and not best_below:
            category = "ncs_ok"
            final_cluster = row.get("best_cluster", row["cluster"])
            final_count = best_cnt if best_cnt is not None else centroid_cnt
            final_reduction_pct = best_pct if best_pct is not None else base_pct
        elif best_below:
            category = "preset_fallback"
            info = preset_data.get(str(row["id"]), {})
            best_oz = info.get("best_oz_count", None)
            final_cluster = row.get("best_cluster", row["cluster"])
            final_count = best_oz
            # Requirement: preset fallback rows force final_reduction_pct to 0
            final_reduction_pct = 0.0
        return pd.Series(
            {
                "final_cluster": int(final_cluster) if pd.notna(final_cluster) else row["cluster"],
                "final_count": final_count if final_count is not None else "",
                "final_reduction_pct": final_reduction_pct if final_reduction_pct is not None else "",
                "category": category,
            }
        )

    final_extra = final.apply(_choose, axis=1)
    final = pd.concat([final, final_extra], axis=1)

    gm_final = geomean_from_pct_series(final["final_reduction_pct"])

    cat_counts = final["category"].value_counts().to_dict()

    stats = {
        "ncs_count": ncs_count,
        "preset_fallback_count": preset_count,
        "category_counts": cat_counts,
        "gm_ncs": gm_ncs,
        "gm_ncs_best": gm_ncs_best,
        "gm_final_5k": gm_final,
        "fallback_elapsed_ms": fallback_elapsed_ms,
        "preset_clang_oz_ms": total_clang_ms,
        "preset_cgym_oz_ms": total_cgym_ms,
        "preset_compile_elapsed_ms": preset_elapsed_ms,
    }
    return final, stats


def main():
    ap = argparse.ArgumentParser(description="Zero-shot pipeline for clustering + centroid + fallback")
    ap.add_argument("--source_root", type=str, required=True)
    ap.add_argument("--cluster_analysis_csv", type=str, required=True)
    ap.add_argument("--embeddings_20k_csv", type=str, required=True)
    ap.add_argument("--embeddings_5k_csv", type=str, required=True)
    ap.add_argument("--cluster_p10_csv", type=str, required=True)
    ap.add_argument("--seq_base_dir", type=str, required=True)
    ap.add_argument("--cluster_to_samples_out", type=str, default="cluster_to_samples_5k.csv")
    ap.add_argument("--oz_centroid_out", type=str, default="oz_centroid_5k_ppo.csv")
    ap.add_argument("--fallback_out", type=str, default="oz_centroid_5k_ppo_fallback.csv")
    ap.add_argument("--top5_csv", type=str, default="top5_nearest_clusters_5k.csv")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--fallback_workers", type=int, default=None, help="If unset, defaults to --workers")
    ap.add_argument("--fallback_inner_threads", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--libllvm_path", type=str, default=None)
    ap.add_argument("--write_top5_if_missing", action="store_true")
    args = ap.parse_args()

    if args.workers <= 0:
        ap.error("--workers must be positive.")
    if args.fallback_workers is not None and args.fallback_workers <= 0:
        ap.error("--fallback_workers must be positive.")
    if args.fallback_inner_threads <= 0:
        ap.error("--fallback_inner_threads must be positive.")
    if args.n_init <= 0:
        ap.error("--n_init must be positive.")

    global SOURCE_ROOT
    SOURCE_ROOT = Path(args.source_root)
    os.environ["RAPO_SOURCE_ROOT"] = os.fspath(SOURCE_ROOT.resolve())
    if args.libllvm_path:
        os.environ["LIBLLVM_PATH"] = args.libllvm_path

    required_paths = [
        Path(args.source_root),
        Path(args.cluster_analysis_csv),
        Path(args.embeddings_20k_csv),
        Path(args.embeddings_5k_csv),
        Path(args.cluster_p10_csv),
        Path(args.seq_base_dir),
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    if not args.write_top5_if_missing and not Path(args.top5_csv).exists():
        raise FileNotFoundError(
            f"Top-5 CSV not found: {args.top5_csv}. "
            "Use --write_top5_if_missing to create it."
        )

    timings = {}
    fallback_workers = args.fallback_workers if args.fallback_workers is not None else args.workers
    print(
        f"[info] fallback apply uses workers={fallback_workers} "
        f"(outer) with {args.fallback_inner_threads}-thread "
        "per-sample fanout."
    )

    # Step 1: load embeddings + centroid prep + assignment
    emb20_df = load_embeddings_df(args.embeddings_20k_csv)
    centroids_df = build_centroid_embedding_df(args.cluster_analysis_csv, emb20_df)
    if centroids_df.empty:
        raise ValueError("No centroid embeddings were available.")
    if len(centroids_df) > len(emb20_df):
        raise ValueError(
            "The number of centroids cannot exceed the number of base embeddings."
        )

    base_embeddings = np.vstack(emb20_df["embedding"].values)
    scaler, kmeans, kmeans_ms = fit_scaler_and_kmeans(
        base_embeddings,
        n_clusters=len(centroids_df),
        random_state=args.random_state,
        n_init=args.n_init,
    )
    timings["kmeans_fit_ms"] = kmeans_ms
    print(f"[timer] KMeans fit (k={len(centroids_df)}) : {kmeans_ms:.2f} ms")

    emb5_df = load_embeddings_df(args.embeddings_5k_csv)
    assignments_df, cluster_to_samples_df, assign_ms, dist_matrix = assign_samples_to_centroids(emb5_df, centroids_df, scaler)
    timings["assign_ms"] = assign_ms
    Path(args.cluster_to_samples_out).parent.mkdir(parents=True, exist_ok=True)
    cluster_to_samples_df.to_csv(args.cluster_to_samples_out, index=False)
    print(f"[save] cluster_to_samples -> {args.cluster_to_samples_out} (rows={len(cluster_to_samples_df)})")
    print(f"[timer] 5k distance+assignment : {assign_ms:.2f} ms")

    if args.write_top5_if_missing and not Path(args.top5_csv).exists():
        maybe_write_top5(
            dist_matrix,
            assignments_df["id"].tolist(),
            centroids_df["cluster_id"].astype(int).tolist(),
            args.top5_csv,
        )

    # Step 2: apply centroid sequences (count_ir_ZS logic)
    cl2code = {
        int(r.cluster_id): normalize_id(r.closest_code_id)
        for r in centroids_df.itertuples()
    }
    centroid_df, gm_5k_ratio, centroid_ms = run_centroid_application(assignments_df, cl2code, args.seq_base_dir, args.oz_centroid_out, workers=args.workers)
    timings["centroid_ms"] = centroid_ms
    print(f"[save] oz_centroid results -> {args.oz_centroid_out} (rows={len(centroid_df)})")
    if gm_5k_ratio is None:
        print("[geomean] could not compute geomean for 5k after centroid")
    else:
        print(f"(1) Geomean reduction_pct (5k, centroid) : {(1.0 - gm_5k_ratio) * 100.0:.4f}%")

    # Step 3: fallback (analyze_buffer + apply_anns logic)
    fallback_final_df, fb_stats = run_fallback_stage(
        centroid_df,
        top5_csv=args.top5_csv,
        cluster_p10_csv=args.cluster_p10_csv,
        cl2code=cl2code,
        seq_base_dir=args.seq_base_dir,
        workers=fallback_workers,
        inner_threads=args.fallback_inner_threads,
    )
    Path(args.fallback_out).parent.mkdir(parents=True, exist_ok=True)
    fallback_final_df.to_csv(args.fallback_out, index=False)

    # Timers
    timings["fallback_ms"] = fb_stats.get("fallback_elapsed_ms", 0.0)
    timings["preset_clang_oz_ms"] = fb_stats.get("preset_clang_oz_ms", 0.0)
    timings["preset_cgym_oz_ms"] = fb_stats.get("preset_cgym_oz_ms", 0.0)

    print(f"[save] final fallback results -> {args.fallback_out} (rows={len(fallback_final_df)})")
    print(f"(2) Geomean for NCS subset (baseline centroid): {fb_stats.get('gm_ncs'):.4f}%") if fb_stats.get("gm_ncs") is not None else print("(2) Geomean for NCS subset unavailable")
    print(f"(3) Geomean for NCS after top2-5: {fb_stats.get('gm_ncs_best'):.4f}%") if fb_stats.get("gm_ncs_best") is not None else print("(3) Geomean for NCS after top2-5 unavailable")
    print(f"(4) Geomean for 5k final: {fb_stats.get('gm_final_5k'):.4f}%") if fb_stats.get("gm_final_5k") is not None else print("(4) Geomean for 5k final unavailable")
    print(f"Number of samples subject to NCS: {fb_stats.get('ncs_count', 0)}")
    print(f"Number of samples subject to preset fallback: {fb_stats.get('preset_fallback_count', 0)}")
    if fb_stats.get("category_counts"):
        print(f"[categories] {fb_stats['category_counts']}")
    print(f"[preset] clang -Oz total time: {timings['preset_clang_oz_ms']:.2f} ms ; CompilerGym/opt -Oz total time: {timings['preset_cgym_oz_ms']:.2f} ms")

    print("\n[timers] (ms)")
    for k, v in timings.items():
        print(f"  - {k}: {v:.2f} ms")


if __name__ == "__main__":
    main()
