#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

try:
    import anndata as ad  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    ad = None


SIGNATURES = {
    "immune_infiltration": [
        "CD3D", "CD3E", "TRBC1", "TRBC2", "NKG7", "GNLY", "PRF1", "IFNG",
        "CXCL9", "CXCL10", "CXCL13", "MS4A1", "CD79A", "LST1", "FCER1G", "TYROBP",
    ],
    "immune_activation": [
        "IFNG", "STAT1", "IRF1", "CXCL9", "CXCL10", "CXCL11",
        "GZMB", "PRF1", "HLA-A", "HLA-B", "B2M", "TAP1",
    ],
    "immune_penetration": [
        "CXCL9", "CXCL10", "CXCL11", "IFNG", "STAT1", "IRF1", "CCL5", "CXCL13",
    ],
    "stromal_retention": [
        "COL1A1", "COL1A2", "COL3A1", "TAGLN", "ACTA2", "THY1", "DCN", "LUM",
        "TGFB1", "TGFBI", "FN1", "CXCL12",
    ],
    "tumor_epithelial": [
        "EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "ERBB2",
    ],
}


def require_anndata() -> Any:
    if ad is None:
        raise SystemExit(
            "anndata is required for Visium preprocessing. "
            "Install dependencies from requirements-hetero.txt"
        )
    return ad


def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "scripts").is_dir() and (candidate / "models").is_dir():
            return candidate
    raise SystemExit("Could not locate project root via repository structure")


def ensure_descendant(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise SystemExit(f"Path escapes project root: {resolved}") from exc
    return resolved


def normalize_symbol(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for raw in names:
        name = raw if raw else "UNNAMED"
        count = seen.get(name, 0)
        if count == 0:
            unique.append(name)
        else:
            unique.append(f"{name}-{count}")
        seen[name] = count + 1
    return unique


def read_tissue_positions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] == 6:
        frame.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
        return frame
    frame = pd.read_csv(path)
    rename_map = {
        "barcode": "barcode",
        "in_tissue": "in_tissue",
        "array_row": "array_row",
        "array_col": "array_col",
        "pxl_row_in_fullres": "pxl_row_in_fullres",
        "pxl_col_in_fullres": "pxl_col_in_fullres",
    }
    return frame.rename(columns=rename_map)


def rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    return series.rank(method="average", pct=True).to_numpy(dtype=np.float32)


def mean_signature(
    log_expr: np.ndarray,
    gene_lookup: dict[str, list[int]],
    genes: list[str],
) -> np.ndarray:
    indices: list[int] = []
    for gene in genes:
        indices.extend(gene_lookup.get(normalize_symbol(gene), []))
    if not indices:
        return np.zeros((log_expr.shape[0],), dtype=np.float32)
    values = log_expr[:, indices]
    if values.ndim == 1:
        values = values[:, None]
    return np.asarray(values.mean(axis=1), dtype=np.float32).reshape(-1)


def preprocess_matrix(counts: sparse.spmatrix) -> np.ndarray:
    counts = counts.tocsr().astype(np.float32)
    library_size = np.asarray(counts.sum(axis=1)).reshape(-1)
    library_size = np.clip(library_size, a_min=1.0, a_max=None)
    scale = (1e4 / library_size).astype(np.float32)
    normalized = counts.multiply(scale[:, None])
    return np.log1p(normalized.toarray()).astype(np.float32)


def assign_compartment(
    tumor_score: np.ndarray,
    retention_score: np.ndarray,
    infiltration_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tumor_probability = np.clip(0.65 * tumor_score + 0.35 * (1.0 - infiltration_score), 0.0, 1.0)
    immune_probability = np.clip(0.7 * infiltration_score + 0.3 * (1.0 - retention_score), 0.0, 1.0)
    labels = np.full(tumor_probability.shape, "boundary", dtype=object)
    labels[tumor_probability >= 0.6] = "tumor"
    labels[(tumor_probability < 0.45) & (immune_probability < 0.45)] = "stroma"
    return labels.astype(str), tumor_probability.astype(np.float32), immune_probability.astype(np.float32)


def assign_time_labels(
    infiltration_score: np.ndarray,
    penetration_score: np.ndarray,
    retention_score: np.ndarray,
    activation_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q40_inf, q50_inf = np.quantile(infiltration_score, [0.4, 0.5])
    q40_pen, q60_pen = np.quantile(penetration_score, [0.4, 0.6])
    q60_ret, q75_ret = np.quantile(retention_score, [0.6, 0.75])
    q40_act, q50_act = np.quantile(activation_score, [0.4, 0.5])

    labels: list[str] = []
    uncertainty: list[bool] = []
    for inf, pen, ret, act in zip(infiltration_score, penetration_score, retention_score, activation_score):
        hot = inf >= q50_inf and pen >= q60_pen and act >= q50_act and ret < q75_ret
        excluded = inf >= q50_inf and pen <= q40_pen and ret >= q60_ret
        cold = inf <= q40_inf and act <= q40_act
        matched = [name for name, flag in [("Hot", hot), ("Excluded", excluded), ("Cold", cold)] if flag]
        if len(matched) == 1:
            labels.append(matched[0])
            uncertainty.append(False)
            continue

        hot_score = inf + pen + act - ret
        excluded_score = inf + ret - pen
        cold_score = (1.0 - inf) + (1.0 - act)
        ordered = sorted(
            [("Hot", hot_score), ("Excluded", excluded_score), ("Cold", cold_score)],
            key=lambda item: item[1],
            reverse=True,
        )
        labels.append(ordered[0][0])
        uncertainty.append((ordered[0][1] - ordered[1][1]) < 0.15 or len(matched) != 1)
    return np.asarray(labels, dtype=object), np.asarray(uncertainty, dtype=bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw 10x Visium output into heuristic TIME-ready h5ad")
    parser.add_argument("--matrix-dir", type=Path, required=True, help="Directory containing matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz")
    parser.add_argument("--spatial-dir", type=Path, required=True, help="Directory containing tissue_positions_list.csv")
    parser.add_argument("--output-path", type=Path, required=True, help="Target .h5ad path inside the project")
    parser.add_argument("--patient-id", type=str, required=True)
    parser.add_argument("--sample-id", type=str, required=True)
    parser.add_argument("--slide-id", type=str, required=True)
    parser.add_argument("--cohort-id", type=str, default="visium_breast_demo")
    parser.add_argument("--split-id", type=str, default="external")
    parser.add_argument("--label-source", type=str, default="signature_rule")
    parser.add_argument("--keep-background", action="store_true", help="Retain out-of-tissue spots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = discover_project_root(args.output_path)
    matrix_dir = ensure_descendant(args.matrix_dir, project_root)
    spatial_dir = ensure_descendant(args.spatial_dir, project_root)
    output_path = ensure_descendant(args.output_path, project_root)

    features_path = matrix_dir / "features.tsv.gz"
    barcodes_path = matrix_dir / "barcodes.tsv.gz"
    matrix_path = matrix_dir / "matrix.mtx.gz"
    positions_path = spatial_dir / "tissue_positions_list.csv"

    if not all(path.exists() for path in [features_path, barcodes_path, matrix_path, positions_path]):
        missing = [str(path) for path in [features_path, barcodes_path, matrix_path, positions_path] if not path.exists()]
        raise SystemExit(f"Missing required Visium inputs: {missing}")

    features = pd.read_csv(features_path, sep="\t", header=None, names=["gene_id", "gene_symbol", "feature_type"])
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None, names=["barcode"])
    positions = read_tissue_positions(positions_path)

    counts = mmread(matrix_path).tocsr().T
    if counts.shape[0] != len(barcodes) or counts.shape[1] != len(features):
        raise SystemExit(
            f"Visium matrix dimensions do not match features/barcodes: counts={counts.shape}, "
            f"barcodes={len(barcodes)}, features={len(features)}"
        )

    obs = barcodes.merge(positions, on="barcode", how="left")
    obs["in_tissue"] = pd.to_numeric(obs["in_tissue"], errors="coerce").fillna(0).astype(int)
    obs = obs.set_index("barcode", drop=True)

    if not args.keep_background:
        keep_mask = obs["in_tissue"].to_numpy(dtype=bool)
        obs = obs.loc[keep_mask].copy()
        counts = counts[keep_mask, :]

    log_expr = preprocess_matrix(counts)
    gene_symbols = features["gene_symbol"].fillna(features["gene_id"]).astype(str).tolist()
    gene_lookup: dict[str, list[int]] = {}
    for index, gene in enumerate(gene_symbols):
        gene_lookup.setdefault(normalize_symbol(gene), []).append(index)

    infiltration_raw = mean_signature(log_expr, gene_lookup, SIGNATURES["immune_infiltration"])
    activation_raw = mean_signature(log_expr, gene_lookup, SIGNATURES["immune_activation"])
    penetration_raw = mean_signature(log_expr, gene_lookup, SIGNATURES["immune_penetration"])
    retention_raw = mean_signature(log_expr, gene_lookup, SIGNATURES["stromal_retention"])
    tumor_raw = mean_signature(log_expr, gene_lookup, SIGNATURES["tumor_epithelial"])

    infiltration_score = rank01(infiltration_raw)
    activation_score = rank01(activation_raw)
    penetration_score = rank01(penetration_raw)
    retention_score = rank01(retention_raw)
    tumor_score = rank01(tumor_raw)

    time_label, _ = assign_time_labels(
        infiltration_score=infiltration_score,
        penetration_score=penetration_score,
        retention_score=retention_score,
        activation_score=activation_score,
    )
    compartment, tumor_probability, immune_probability = assign_compartment(
        tumor_score=tumor_score,
        retention_score=retention_score,
        infiltration_score=infiltration_score,
    )

    obs["patient_id"] = args.patient_id
    obs["sample_id"] = args.sample_id
    obs["slide_id"] = args.slide_id
    obs["cohort_id"] = args.cohort_id
    obs["split_id"] = args.split_id
    obs["label_source"] = args.label_source
    # Graph-level uncertainty is re-evaluated downstream after regional aggregation.
    # Keeping spot-level flags false avoids collapsing the entire external slide into
    # an unsupervised set because a single ambiguous spot appears inside each tile.
    obs["is_uncertain"] = False
    obs["compartment"] = compartment.astype(str)
    obs["time_label"] = time_label.astype(str)
    obs["infiltration_score"] = infiltration_score.astype(np.float32)
    obs["penetration_score"] = penetration_score.astype(np.float32)
    obs["retention_score"] = retention_score.astype(np.float32)
    obs["activation_score"] = activation_score.astype(np.float32)
    obs["tumor_probability"] = tumor_probability.astype(np.float32)
    obs["immune_probability"] = immune_probability.astype(np.float32)

    var = pd.DataFrame(
        {
            "gene_ids": features["gene_id"].astype(str).to_numpy(),
            "feature_types": features["feature_type"].astype(str).to_numpy(),
            "gene_symbol": features["gene_symbol"].astype(str).to_numpy(),
        },
        index=make_unique(gene_symbols),
    )

    ad_module = require_anndata()
    adata = ad_module.AnnData(X=sparse.csr_matrix(log_expr), obs=obs, var=var)
    adata.obsm["spatial"] = obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    print(
        f"Wrote {output_path} with {adata.n_obs} spots, {adata.n_vars} genes, "
        f"labels={obs['time_label'].value_counts().to_dict()}"
    )


if __name__ == "__main__":
    main()
