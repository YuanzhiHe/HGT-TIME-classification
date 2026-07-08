#!/usr/bin/env python3
"""Preprocess HEST-1k Visium breast cancer samples for the HGT-TIME pipeline.

Reads HEST-1k H5AD files (already log-normalized), computes TIME signature
scores and labels, then writes pipeline-ready H5AD with required obs columns.

Usage:
    python preprocess_hest1k_visium.py \
        --hest-dir data/hest1k_breast/st \
        --metadata data/hest1k_breast/metadata.csv \
        --output-dir datasets/spatial/processed/hest1k_visium \
        --sample-ids TENX68 TENX53 TENX39 TENX24 TENX14 TENX13 NCBI776
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Force pandas to use Python strings instead of Arrow-backed StringDtype.
# Required for compatibility with anndata's H5AD writer.
pd.options.mode.string_storage = "python"
try:
    pd.set_option("future.infer_string", False)
except (KeyError, pd.errors.OptionError):
    pass

import anndata as ad
from scipy import sparse

logger = logging.getLogger(__name__)

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


def normalize_symbol(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    return series.rank(method="average", pct=True).to_numpy(dtype=np.float32)


def dense_expr(adata: ad.AnnData) -> np.ndarray:
    """Get dense expression matrix, handling sparse formats."""
    X = adata.X
    if sparse.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def ensure_log1p(expr: np.ndarray) -> np.ndarray:
    """Ensure expression is log1p-transformed (heuristic check)."""
    max_val = expr.max()
    if max_val > 50:
        # Likely raw counts, apply normalization
        lib_size = np.clip(expr.sum(axis=1, keepdims=True), 1.0, None)
        normalized = expr / lib_size * 1e4
        return np.log1p(normalized).astype(np.float32)
    return expr


def build_gene_lookup(var_names: pd.Index) -> dict[str, list[int]]:
    lookup: dict[str, list[int]] = {}
    for idx, name in enumerate(var_names):
        key = normalize_symbol(name)
        lookup.setdefault(key, []).append(idx)
    return lookup


def mean_signature(
    log_expr: np.ndarray,
    gene_lookup: dict[str, list[int]],
    genes: list[str],
) -> np.ndarray:
    indices: list[int] = []
    for gene in genes:
        indices.extend(gene_lookup.get(normalize_symbol(gene), []))
    if not indices:
        return np.zeros(log_expr.shape[0], dtype=np.float32)
    values = log_expr[:, indices]
    if values.ndim == 1:
        values = values[:, None]
    return np.asarray(values.mean(axis=1), dtype=np.float32).reshape(-1)


def assign_compartment(
    tumor_score: np.ndarray,
    retention_score: np.ndarray,
    infiltration_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tumor_probability = np.clip(
        0.65 * tumor_score + 0.35 * (1.0 - infiltration_score), 0.0, 1.0
    )
    immune_probability = np.clip(
        0.7 * infiltration_score + 0.3 * (1.0 - retention_score), 0.0, 1.0
    )
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
    for inf, pen, ret, act in zip(
        infiltration_score, penetration_score, retention_score, activation_score
    ):
        hot = inf >= q50_inf and pen >= q60_pen and act >= q50_act and ret < q75_ret
        excluded = inf >= q50_inf and pen <= q40_pen and ret >= q60_ret
        cold = inf <= q40_inf and act <= q40_act
        matched = [
            name for name, flag in [("Hot", hot), ("Excluded", excluded), ("Cold", cold)]
            if flag
        ]
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


def process_sample(
    h5ad_path: Path,
    sample_id: str,
    patient_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Process a single HEST-1k sample into pipeline-ready H5AD."""
    logger.info(f"Processing {sample_id} from {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"  Shape: {adata.shape}")

    # Get expression matrix
    expr = dense_expr(adata)
    expr = ensure_log1p(expr)

    # Build gene lookup
    gene_lookup = build_gene_lookup(adata.var_names)

    # Compute signature scores
    infiltration_raw = mean_signature(expr, gene_lookup, SIGNATURES["immune_infiltration"])
    activation_raw = mean_signature(expr, gene_lookup, SIGNATURES["immune_activation"])
    penetration_raw = mean_signature(expr, gene_lookup, SIGNATURES["immune_penetration"])
    retention_raw = mean_signature(expr, gene_lookup, SIGNATURES["stromal_retention"])
    tumor_raw = mean_signature(expr, gene_lookup, SIGNATURES["tumor_epithelial"])

    infiltration_score = rank01(infiltration_raw)
    activation_score = rank01(activation_raw)
    penetration_score = rank01(penetration_raw)
    retention_score = rank01(retention_raw)
    tumor_score = rank01(tumor_raw)

    # Assign TIME labels
    time_label, is_uncertain = assign_time_labels(
        infiltration_score, penetration_score, retention_score, activation_score,
    )

    # Assign compartments
    compartment, tumor_probability, immune_probability = assign_compartment(
        tumor_score, retention_score, infiltration_score,
    )

    # Add required obs columns
    adata.obs["patient_id"] = patient_id
    adata.obs["sample_id"] = sample_id
    adata.obs["slide_id"] = f"slide_{sample_id}"
    adata.obs["cohort_id"] = "visium_breast_hest1k"
    adata.obs["split_id"] = "external"
    adata.obs["label_source"] = "signature_rule"
    # For external validation data, disable spot-level uncertainty so all
    # graph tiles remain usable downstream.
    adata.obs["is_uncertain"] = False
    adata.obs["compartment"] = compartment
    adata.obs["time_label"] = time_label
    adata.obs["infiltration_score"] = infiltration_score
    adata.obs["penetration_score"] = penetration_score
    adata.obs["retention_score"] = retention_score
    adata.obs["activation_score"] = activation_score
    adata.obs["tumor_probability"] = tumor_probability
    adata.obs["immune_probability"] = immune_probability

    # Ensure in_tissue column exists
    if "in_tissue" not in adata.obs.columns:
        adata.obs["in_tissue"] = 1

    # Ensure spatial coordinates are available
    if "spatial" not in adata.obsm:
        if "pxl_col_in_fullres" in adata.obs.columns and "pxl_row_in_fullres" in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[
                ["pxl_col_in_fullres", "pxl_row_in_fullres"]
            ].to_numpy(dtype=np.float32)
        else:
            raise ValueError(f"No spatial coordinates found in {h5ad_path}")

    # Ensure array_row/array_col exist (used as coord_x/coord_y by spatial_adjacency)
    if "array_row" not in adata.obs.columns:
        spatial = adata.obsm["spatial"]
        adata.obs["array_col"] = spatial[:, 0].astype(np.float32)
        adata.obs["array_row"] = spatial[:, 1].astype(np.float32)

    # Store the log1p expression back
    adata.X = sparse.csr_matrix(expr)

    # Convert Arrow/StringDtype to plain object dtype for H5AD compatibility.
    # pandas 2.x uses ArrowStringArray by default which anndata can't serialize.
    def _fix_df_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.Index([str(x) for x in df.index])
        for col in df.columns:
            s = df[col]
            if isinstance(s.dtype, pd.CategoricalDtype):
                # Convert categorical with Arrow-backed categories to plain object
                df[col] = s.astype("object")
            elif isinstance(s.dtype, pd.StringDtype):
                df[col] = s.astype("object")
        return df

    adata.obs = _fix_df_for_h5ad(adata.obs)
    adata.var = _fix_df_for_h5ad(adata.var)

    # Write output
    output_path = output_dir / f"{sample_id}.h5ad"
    output_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)

    label_counts = pd.Series(time_label).value_counts().to_dict()
    logger.info(
        f"  Wrote {output_path}: {adata.n_obs} spots, {adata.n_vars} genes, "
        f"TIME labels: {label_counts}"
    )
    return {
        "sample_id": sample_id,
        "patient_id": patient_id,
        "n_spots": adata.n_obs,
        "n_genes": adata.n_vars,
        "label_counts": label_counts,
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess HEST-1k Visium breast samples for HGT-TIME pipeline"
    )
    parser.add_argument("--hest-dir", type=Path, required=True,
                        help="Directory containing HEST-1k .h5ad files")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="HEST-1k metadata.csv")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for preprocessed H5AD files")
    parser.add_argument("--sample-ids", nargs="+", required=True,
                        help="Sample IDs to process (e.g., TENX68 TENX53)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    meta = pd.read_csv(args.metadata)
    meta = meta.set_index("id")

    results = []
    for sid in args.sample_ids:
        h5ad_path = args.hest_dir / f"{sid}.h5ad"
        if not h5ad_path.exists():
            logger.warning(f"Skipping {sid}: {h5ad_path} not found")
            continue

        # Determine patient_id from metadata (or use sample_id as fallback)
        if sid in meta.index:
            row = meta.loc[sid]
            patient_id = str(row.get("patient", sid))
            if patient_id == "nan" or not patient_id:
                patient_id = f"hest_{sid}"
        else:
            patient_id = f"hest_{sid}"

        result = process_sample(h5ad_path, sid, patient_id, args.output_dir)
        results.append(result)

    logger.info(f"\nProcessed {len(results)} samples:")
    for r in results:
        logger.info(
            f"  {r['sample_id']} (patient={r['patient_id']}): "
            f"{r['n_spots']} spots, {r['n_genes']} genes, {r['label_counts']}"
        )


if __name__ == "__main__":
    main()
