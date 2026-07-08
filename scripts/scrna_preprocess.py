#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def build_default_config() -> dict[str, Any]:
    return {
        "dataset": {
            "dataset_id": "gse161529",
            "input_path": "datasets/gse161529/processed/input.h5ad",
            "input_format": "auto",
            "species": "human",
            "gene_symbol_column": None,
            "sample_id_column": "sample_id",
            "patient_id_column": "patient_id",
            "cohort_id_column": "cohort_id",
            "batch_key": "batch_id",
        },
        "qc": {
            "min_genes_per_cell": 200,
            "max_genes_per_cell": 8000,
            "min_counts_per_cell": 500,
            "max_counts_per_cell": 50000,
            "max_pct_mt": 20.0,
            "min_cells_per_gene": 3,
            "apply_mad_upper_bounds": True,
            "mad_multiplier": 3.0,
            "mitochondrial_prefixes": ["MT-"],
            "ribosomal_prefixes": ["RPS", "RPL"],
            "hemoglobin_prefixes": ["HB"],
        },
        "preprocess": {
            "target_sum": 10000,
            "hvg_flavor": "seurat",
            "n_top_genes": 3000,
            "subset_to_hvg": True,
            "scale_max_value": 10.0,
            "n_pcs": 50,
            "neighbors_k": 15,
            "umap_min_dist": 0.3,
            "leiden_resolution": 0.6,
            "batch_correction": "none",
            "cluster_key": "leiden",
        },
        "annotation": {
            "unknown_label": "Unknown",
            "min_marker_score": 0.15,
            "use_reference_labels": True,
            "reference_label_path": None,
            "reference_label_column": "seurat_cell_type",
            "reference_score_column": "seurat_label_score",
            "reference_priority": True,
            "cell_type_markers": {
                "T_NK": ["CD3D", "CD3E", "TRBC1", "TRBC2", "NKG7", "IL7R"],
                "B_Plasma": ["MS4A1", "CD79A", "CD79B", "JCHAIN", "MZB1"],
                "Myeloid": ["LYZ", "LST1", "FCER1G", "TYROBP", "CTSS"],
                "CAF": ["COL1A1", "COL1A2", "DCN", "LUM", "TAGLN"],
                "Endothelial": ["PECAM1", "VWF", "KDR", "EMCN", "RAMP2"],
                "Epithelial_Malignant": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1"],
            },
            "cell_state_programs": {
                "cytotoxic": ["NKG7", "GNLY", "PRF1", "GZMB", "IFNG"],
                "exhausted": ["PDCD1", "LAG3", "TIGIT", "HAVCR2", "CTLA4"],
                "interferon_response": ["STAT1", "ISG15", "IFIT1", "IFI6", "CXCL10"],
                "cycling": ["MKI67", "TOP2A", "TYMS", "UBE2C", "BIRC5"],
                "stromal_activation": ["COL1A1", "COL3A1", "TAGLN", "ACTA2", "THY1"],
                "epithelial_malignant": ["EPCAM", "KRT8", "KRT18", "KRT19", "ERBB2"],
            },
        },
        "output": {
            "root_dir": "outputs/scrna/gse161529__reference_v1",
            "prefix": "gse161529__reference_v1",
            "graph_feature_space": "pca",
            "graph_feature_dims": 32,
            "export_h5ad": True,
            "export_graph_features": True,
            "export_patient_state_summary": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="scRNA-seq preprocessing and annotation pipeline")
    parser.add_argument("--config", type=Path, help="Path to YAML config")
    parser.add_argument(
        "--write-default-config",
        type=Path,
        help="Write the default config template and exit",
    )
    return parser.parse_args()


def require_scanpy() -> tuple[Any, Any]:
    try:
        import scanpy as sc  # type: ignore
        import scanpy.external as sce  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scanpy is required to run this pipeline. "
            "Install dependencies from requirements-scrna.txt"
        ) from exc
    return sc, sce


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    default_config = build_default_config()
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    config = deep_update(copy.deepcopy(default_config), user_config)
    config["_config_path"] = str(config_path.resolve())
    return config


def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "instance.json").exists():
            return candidate
    raise SystemExit("Could not locate project root via instance.json")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "," if path.suffix == ".csv" else "\t"
    frame.to_csv(path, sep=sep, index=False)


def load_input(sc: Any, dataset_cfg: dict[str, Any], base_dir: Path) -> Any:
    input_path = resolve_path(base_dir, dataset_cfg["input_path"])
    if input_path is None:
        raise SystemExit("dataset.input_path must be set")
    input_format = dataset_cfg.get("input_format", "auto")

    if input_format == "auto":
        if input_path.suffix == ".h5ad":
            input_format = "h5ad"
        elif input_path.suffix in {".h5", ".hdf5"}:
            input_format = "10x_h5"
        elif input_path.is_dir():
            input_format = "10x_mtx"
        else:
            raise SystemExit(f"Cannot infer input format from {input_path}")

    if input_format == "h5ad":
        adata = sc.read_h5ad(input_path)
    elif input_format == "10x_h5":
        adata = sc.read_10x_h5(input_path)
    elif input_format == "10x_mtx":
        adata = sc.read_10x_mtx(input_path, var_names="gene_symbols", make_unique=True)
    else:
        raise SystemExit(f"Unsupported input format: {input_format}")

    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["dataset_id"] = dataset_cfg["dataset_id"]
    return adata


def ensure_obs_columns(adata: Any, dataset_cfg: dict[str, Any]) -> None:
    mappings = {
        "sample_id": dataset_cfg.get("sample_id_column"),
        "patient_id": dataset_cfg.get("patient_id_column"),
        "cohort_id": dataset_cfg.get("cohort_id_column"),
        "batch_id": dataset_cfg.get("batch_key"),
    }
    for output_name, source_name in mappings.items():
        if source_name and source_name in adata.obs.columns:
            adata.obs[output_name] = adata.obs[source_name].astype(str)
        else:
            adata.obs[output_name] = "unknown"


def standardize_gene_metadata(adata: Any, dataset_cfg: dict[str, Any]) -> None:
    gene_symbol_column = dataset_cfg.get("gene_symbol_column")
    if gene_symbol_column and gene_symbol_column in adata.var.columns:
        gene_symbols = adata.var[gene_symbol_column].astype(str)
    else:
        gene_symbols = pd.Index(adata.var_names.astype(str))
    adata.var["gene_id"] = adata.var_names.astype(str)
    adata.var["gene_symbol"] = gene_symbols


def add_qc_flags(adata: Any, qc_cfg: dict[str, Any]) -> list[str]:
    qc_vars: list[str] = []
    prefix_map = {
        "mt": qc_cfg.get("mitochondrial_prefixes", []),
        "ribo": qc_cfg.get("ribosomal_prefixes", []),
        "hb": qc_cfg.get("hemoglobin_prefixes", []),
    }
    gene_symbols = adata.var["gene_symbol"].astype(str).str.upper()
    for key, prefixes in prefix_map.items():
        prefixes = [prefix.upper() for prefix in prefixes]
        adata.var[key] = gene_symbols.str.startswith(tuple(prefixes))
        qc_vars.append(key)
    return qc_vars


def robust_upper_bound(series: pd.Series, hard_cap: float | None, use_mad: bool, mad_multiplier: float) -> float | None:
    candidates: list[float] = []
    if hard_cap is not None:
        candidates.append(float(hard_cap))
    if use_mad:
        median = float(series.median())
        mad = float((series - median).abs().median())
        if mad > 0:
            candidates.append(median + mad_multiplier * mad)
    if not candidates:
        return None
    return min(candidates)


def apply_qc(sc: Any, adata: Any, qc_cfg: dict[str, Any]) -> tuple[Any, pd.DataFrame, dict[str, float | None]]:
    qc_vars = add_qc_flags(adata, qc_cfg)
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, inplace=True, log1p=False)

    max_genes = robust_upper_bound(
        adata.obs["n_genes_by_counts"],
        qc_cfg.get("max_genes_per_cell"),
        qc_cfg.get("apply_mad_upper_bounds", True),
        qc_cfg.get("mad_multiplier", 3.0),
    )
    max_counts = robust_upper_bound(
        adata.obs["total_counts"],
        qc_cfg.get("max_counts_per_cell"),
        qc_cfg.get("apply_mad_upper_bounds", True),
        qc_cfg.get("mad_multiplier", 3.0),
    )

    mask = adata.obs["n_genes_by_counts"] >= qc_cfg["min_genes_per_cell"]
    mask &= adata.obs["total_counts"] >= qc_cfg["min_counts_per_cell"]
    mask &= adata.obs["pct_counts_mt"] <= qc_cfg["max_pct_mt"]
    if max_genes is not None:
        mask &= adata.obs["n_genes_by_counts"] <= max_genes
    if max_counts is not None:
        mask &= adata.obs["total_counts"] <= max_counts

    adata.obs["pass_qc"] = mask.astype(bool)

    qc_metrics = adata.obs[
        [
            "cell_id",
            "dataset_id",
            "sample_id",
            "patient_id",
            "cohort_id",
            "batch_id",
            "n_genes_by_counts",
            "total_counts",
            "pct_counts_mt",
            "pct_counts_ribo",
            "pct_counts_hb",
            "pass_qc",
        ]
    ].copy()

    adata = adata[adata.obs["pass_qc"].to_numpy()].copy()
    sc.pp.filter_genes(adata, min_cells=qc_cfg["min_cells_per_gene"])

    thresholds = {
        "max_genes_per_cell": max_genes,
        "max_counts_per_cell": max_counts,
        "max_pct_mt": qc_cfg["max_pct_mt"],
    }
    return adata, qc_metrics, thresholds


def normalize_hvg(sc: Any, adata: Any, preprocess_cfg: dict[str, Any], batch_key: str | None) -> Any:
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=preprocess_cfg["target_sum"])
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()
    adata.raw = adata.copy()

    hvg_kwargs = {
        "n_top_genes": preprocess_cfg["n_top_genes"],
        "flavor": preprocess_cfg["hvg_flavor"],
    }
    if batch_key and batch_key in adata.obs.columns and adata.obs[batch_key].nunique() > 1:
        hvg_kwargs["batch_key"] = batch_key
    sc.pp.highly_variable_genes(adata, **hvg_kwargs)

    if preprocess_cfg.get("subset_to_hvg", True):
        adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()

    adata.layers["log1p_hvg"] = adata.X.copy()
    return adata


def cluster_embeddings(sc: Any, sce: Any, adata: Any, preprocess_cfg: dict[str, Any], batch_key: str | None) -> tuple[Any, str]:
    sc.pp.scale(adata, max_value=preprocess_cfg["scale_max_value"])
    adata.layers["scaled_hvg"] = adata.X.copy()
    sc.tl.pca(adata, n_comps=preprocess_cfg["n_pcs"], svd_solver="arpack")

    use_rep = "X_pca"
    batch_method = preprocess_cfg.get("batch_correction", "none").lower()
    if batch_key and batch_key in adata.obs.columns and adata.obs[batch_key].nunique() > 1:
        if batch_method == "harmony":
            sce.pp.harmony_integrate(adata, batch_key, basis="X_pca")
            use_rep = "X_pca_harmony"
        elif batch_method == "bbknn":
            sce.pp.bbknn(
                adata,
                batch_key=batch_key,
                n_pcs=preprocess_cfg["n_pcs"],
                neighbors_within_batch=max(3, preprocess_cfg["neighbors_k"] // 2),
            )
        elif batch_method != "none":
            raise SystemExit(f"Unsupported batch_correction: {batch_method}")

    if batch_method != "bbknn":
        sc.pp.neighbors(adata, n_neighbors=preprocess_cfg["neighbors_k"], use_rep=use_rep)
    sc.tl.umap(adata, min_dist=preprocess_cfg["umap_min_dist"])
    sc.tl.leiden(adata, resolution=preprocess_cfg["leiden_resolution"], key_added=preprocess_cfg["cluster_key"])
    return adata, use_rep


def resolve_gene_list(adata: Any, genes: list[str]) -> list[str]:
    symbol_to_var = {
        str(symbol).upper(): str(var_name)
        for symbol, var_name in zip(adata.var["gene_symbol"], adata.var_names)
    }
    resolved = []
    for gene in genes:
        match = symbol_to_var.get(str(gene).upper())
        if match is not None:
            resolved.append(match)
    return sorted(set(resolved))


def score_gene_sets(sc: Any, adata: Any, gene_sets: dict[str, list[str]], prefix: str) -> dict[str, int]:
    coverage: dict[str, int] = {}
    for label, genes in gene_sets.items():
        resolved = resolve_gene_list(adata, genes)
        score_key = f"score__{prefix}__{slugify(label)}"
        if resolved:
            sc.tl.score_genes(adata, resolved, score_name=score_key, use_raw=False)
        else:
            adata.obs[score_key] = 0.0
        coverage[label] = len(resolved)
    return coverage


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def read_reference_labels(path: Path, label_column: str, score_column: str | None) -> pd.DataFrame:
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(path, sep="\t")
    required = {"cell_id", label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Reference label table missing columns: {sorted(missing)}")
    keep_columns = ["cell_id", label_column]
    if score_column and score_column in frame.columns:
        keep_columns.append(score_column)
    return frame[keep_columns].copy()


def assign_top_label(score_frame: pd.DataFrame, labels: list[str], min_score: float, unknown_label: str) -> tuple[pd.Series, pd.Series]:
    if score_frame.empty:
        return (
            pd.Series([unknown_label] * len(score_frame.index), index=score_frame.index),
            pd.Series([0.0] * len(score_frame.index), index=score_frame.index),
        )
    score_values = score_frame.to_numpy()
    best_index = score_values.argmax(axis=1)
    best_score = score_values[np.arange(score_values.shape[0]), best_index]
    assignments = [labels[idx] if best_score[pos] >= min_score else unknown_label for pos, idx in enumerate(best_index)]
    return pd.Series(assignments, index=score_frame.index), pd.Series(best_score, index=score_frame.index)


def annotate_cells(sc: Any, adata: Any, annotation_cfg: dict[str, Any], cluster_key: str, base_dir: Path) -> tuple[Any, dict[str, Any]]:
    marker_sets = annotation_cfg.get("cell_type_markers", {})
    state_sets = annotation_cfg.get("cell_state_programs", {})
    coverage = {
        "cell_type_markers": score_gene_sets(sc, adata, marker_sets, "cell_type"),
        "cell_state_programs": score_gene_sets(sc, adata, state_sets, "cell_state"),
    }

    marker_labels = list(marker_sets.keys())
    marker_score_columns = [f"score__cell_type__{slugify(label)}" for label in marker_labels]
    marker_frame = adata.obs[marker_score_columns].copy()
    marker_assignments, marker_scores = assign_top_label(
        marker_frame,
        marker_labels,
        annotation_cfg["min_marker_score"],
        annotation_cfg["unknown_label"],
    )
    adata.obs["marker_cell_type"] = marker_assignments.astype(str)
    adata.obs["marker_cell_type_score"] = marker_scores.astype(float)

    state_labels = list(state_sets.keys())
    state_score_columns = [f"score__cell_state__{slugify(label)}" for label in state_labels]
    state_frame = adata.obs[state_score_columns].copy()
    state_assignments, state_scores = assign_top_label(
        state_frame,
        state_labels,
        annotation_cfg["min_marker_score"],
        annotation_cfg["unknown_label"],
    )
    adata.obs["cell_state"] = state_assignments.astype(str)
    adata.obs["cell_state_score"] = state_scores.astype(float)

    adata.obs["reference_cell_type"] = pd.Series(annotation_cfg["unknown_label"], index=adata.obs.index)
    reference_path = resolve_path(base_dir, annotation_cfg.get("reference_label_path"))
    if annotation_cfg.get("use_reference_labels") and reference_path is not None and reference_path.exists():
        reference_labels = read_reference_labels(
            reference_path,
            annotation_cfg["reference_label_column"],
            annotation_cfg.get("reference_score_column"),
        )
        reference_labels = reference_labels.set_index("cell_id")
        adata.obs["reference_cell_type"] = (
            adata.obs["cell_id"].map(reference_labels[annotation_cfg["reference_label_column"]]).fillna(annotation_cfg["unknown_label"]).astype(str)
        )
        if annotation_cfg.get("reference_score_column") and annotation_cfg["reference_score_column"] in reference_labels.columns:
            adata.obs["reference_cell_type_score"] = (
                adata.obs["cell_id"].map(reference_labels[annotation_cfg["reference_score_column"]]).fillna(0.0).astype(float)
            )
        else:
            adata.obs["reference_cell_type_score"] = 0.0
    else:
        adata.obs["reference_cell_type_score"] = 0.0

    if annotation_cfg.get("reference_priority", True):
        adata.obs["cell_type"] = np.where(
            adata.obs["reference_cell_type"] != annotation_cfg["unknown_label"],
            adata.obs["reference_cell_type"],
            adata.obs["marker_cell_type"],
        )
    else:
        adata.obs["cell_type"] = adata.obs["marker_cell_type"]

    cluster_majority = (
        adata.obs.groupby(cluster_key)["cell_type"]
        .agg(lambda series: series.value_counts().index[0] if not series.empty else annotation_cfg["unknown_label"])
        .to_dict()
    )
    adata.obs["cluster_cell_type"] = adata.obs[cluster_key].map(cluster_majority).astype(str)

    summary = {
        "marker_gene_coverage": coverage["cell_type_markers"],
        "state_gene_coverage": coverage["cell_state_programs"],
        "cell_type_counts": adata.obs["cell_type"].value_counts().to_dict(),
        "cell_state_counts": adata.obs["cell_state"].value_counts().to_dict(),
    }
    return adata, summary


def build_annotation_table(adata: Any, cluster_key: str) -> pd.DataFrame:
    score_columns = [column for column in adata.obs.columns if column.startswith("score__")]
    columns = [
        "cell_id",
        "dataset_id",
        "sample_id",
        "patient_id",
        "cohort_id",
        "batch_id",
        cluster_key,
        "marker_cell_type",
        "marker_cell_type_score",
        "reference_cell_type",
        "reference_cell_type_score",
        "cluster_cell_type",
        "cell_type",
        "cell_state",
        "cell_state_score",
    ]
    return adata.obs[columns + score_columns].reset_index(drop=True)


def build_patient_state_summary(adata: Any) -> pd.DataFrame:
    summary = (
        adata.obs.groupby(["patient_id", "sample_id", "cell_type", "cell_state"], dropna=False)
        .size()
        .rename("cell_count")
        .reset_index()
    )
    per_sample = summary.groupby(["patient_id", "sample_id"])["cell_count"].transform("sum")
    summary["cell_fraction"] = summary["cell_count"] / per_sample.replace(0, np.nan)
    return summary.sort_values(["patient_id", "sample_id", "cell_count"], ascending=[True, True, False])


def dense_matrix(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def extract_graph_feature_matrix(adata: Any, feature_space: str, feature_dims: int, use_rep: str) -> tuple[np.ndarray, list[str]]:
    feature_space = feature_space.lower()
    if feature_space == "pca":
        matrix = np.asarray(adata.obsm["X_pca"])[:, :feature_dims]
        columns = [f"feature__pca_{index:03d}" for index in range(matrix.shape[1])]
        return matrix, columns
    if feature_space == "harmony":
        basis = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else use_rep
        matrix = np.asarray(adata.obsm[basis])[:, :feature_dims]
        columns = [f"feature__harmony_{index:03d}" for index in range(matrix.shape[1])]
        return matrix, columns
    if feature_space == "scaled_hvg":
        matrix = dense_matrix(adata.layers["scaled_hvg"])[:, :feature_dims]
        columns = [f"feature__hvg_{index:03d}" for index in range(matrix.shape[1])]
        return matrix, columns
    raise SystemExit(f"Unsupported graph_feature_space: {feature_space}")


def build_graph_exports(adata: Any, output_cfg: dict[str, Any], use_rep: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_matrix, feature_columns = extract_graph_feature_matrix(
        adata,
        output_cfg["graph_feature_space"],
        output_cfg["graph_feature_dims"],
        use_rep,
    )
    feature_frame = pd.DataFrame(feature_matrix, columns=feature_columns)
    metadata_columns = [
        "cell_id",
        "dataset_id",
        "sample_id",
        "patient_id",
        "cohort_id",
        "batch_id",
        "cell_type",
        "cell_state",
        "cluster_cell_type",
        "marker_cell_type",
        "reference_cell_type",
        "n_genes_by_counts",
        "total_counts",
        "pct_counts_mt",
    ]
    score_columns = [column for column in adata.obs.columns if column.startswith("score__")]
    metadata = adata.obs[metadata_columns + score_columns].reset_index(drop=True)
    graph_features = pd.concat([metadata.copy(), feature_frame], axis=1)

    gene_columns = ["gene_id", "gene_symbol"]
    extra_gene_columns = [column for column in ["highly_variable", "means", "dispersions", "dispersions_norm"] if column in adata.var.columns]
    gene_panel = adata.var[gene_columns + extra_gene_columns].copy().reset_index(drop=True)
    return graph_features, metadata, gene_panel


def output_paths(root_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "raw_h5ad": root_dir / f"{prefix}__step-01_raw.h5ad",
        "qc_metrics": root_dir / f"{prefix}__step-02_qc_metrics.tsv.gz",
        "qc_filtered_h5ad": root_dir / f"{prefix}__step-03_qc_filtered.h5ad",
        "lognorm_hvg_h5ad": root_dir / f"{prefix}__step-04_lognorm_hvg.h5ad",
        "clustered_h5ad": root_dir / f"{prefix}__step-05_clustered.h5ad",
        "annotations": root_dir / f"{prefix}__step-06_annotations.tsv.gz",
        "patient_state_summary": root_dir / f"{prefix}__step-07_patient_state_summary.tsv.gz",
        "graph_features": root_dir / f"{prefix}__step-08_graph_cell_features.tsv.gz",
        "graph_metadata": root_dir / f"{prefix}__step-08_graph_cell_metadata.tsv.gz",
        "gene_panel": root_dir / f"{prefix}__step-08_gene_panel.tsv",
        "graph_ready_h5ad": root_dir / f"{prefix}__step-08_graph_ready.h5ad",
        "manifest": root_dir / f"{prefix}__step-08_manifest.json",
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.write_default_config:
        write_yaml(args.write_default_config, build_default_config())
        return
    if not args.config:
        raise SystemExit("Either --config or --write-default-config must be provided")

    config_path = args.config.resolve()
    config = load_config(config_path)
    base_dir = discover_project_root(config_path)
    output_root = resolve_path(base_dir, config["output"]["root_dir"])
    if output_root is None:
        raise SystemExit("output.root_dir must be set")
    paths = output_paths(output_root, config["output"]["prefix"])
    output_root.mkdir(parents=True, exist_ok=True)

    sc, sce = require_scanpy()

    adata = load_input(sc, config["dataset"], base_dir)
    ensure_obs_columns(adata, config["dataset"])
    standardize_gene_metadata(adata, config["dataset"])
    if config["output"].get("export_h5ad", True):
        adata.write_h5ad(paths["raw_h5ad"])

    adata, qc_metrics, qc_thresholds = apply_qc(sc, adata, config["qc"])
    write_table(qc_metrics, paths["qc_metrics"])
    if config["output"].get("export_h5ad", True):
        adata.write_h5ad(paths["qc_filtered_h5ad"])

    adata = normalize_hvg(
        sc,
        adata,
        config["preprocess"],
        config["dataset"].get("batch_key"),
    )
    if config["output"].get("export_h5ad", True):
        adata.write_h5ad(paths["lognorm_hvg_h5ad"])

    adata, use_rep = cluster_embeddings(
        sc,
        sce,
        adata,
        config["preprocess"],
        config["dataset"].get("batch_key"),
    )
    if config["output"].get("export_h5ad", True):
        adata.write_h5ad(paths["clustered_h5ad"])

    cluster_key = config["preprocess"]["cluster_key"]
    adata, annotation_summary = annotate_cells(sc, adata, config["annotation"], cluster_key, base_dir)
    annotation_table = build_annotation_table(adata, cluster_key)
    write_table(annotation_table, paths["annotations"])

    if config["output"].get("export_patient_state_summary", True):
        patient_state_summary = build_patient_state_summary(adata)
        write_table(patient_state_summary, paths["patient_state_summary"])

    graph_features, graph_metadata, gene_panel = build_graph_exports(adata, config["output"], use_rep)
    if config["output"].get("export_graph_features", True):
        write_table(graph_features, paths["graph_features"])
        write_table(graph_metadata, paths["graph_metadata"])
        write_table(gene_panel, paths["gene_panel"])
        if config["output"].get("export_h5ad", True):
            adata.write_h5ad(paths["graph_ready_h5ad"])

    manifest = {
        "config_path": config["_config_path"],
        "dataset_id": config["dataset"]["dataset_id"],
        "n_cells_final": int(adata.n_obs),
        "n_genes_final": int(adata.n_vars),
        "qc_thresholds": qc_thresholds,
        "cluster_key": cluster_key,
        "graph_feature_space": config["output"]["graph_feature_space"],
        "graph_feature_dims": config["output"]["graph_feature_dims"],
        "annotation_summary": annotation_summary,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    write_manifest(paths["manifest"], manifest)


if __name__ == "__main__":
    main()
