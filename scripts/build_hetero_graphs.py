#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA

try:
    import anndata as ad  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    ad = None

try:
    from torch_geometric.data import HeteroData  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    HeteroData = None


CELL_TYPE_ORDER = [
    "T_NK",
    "B_Plasma",
    "Myeloid",
    "CAF",
    "Endothelial",
    "Epithelial_Malignant",
]
PROGRAM_ORDER = [
    "cytotoxic",
    "exhausted",
    "interferon_response",
    "cycling",
    "stromal_activation",
    "epithelial_malignant",
]
TOP_CLASS_ORDER = [
    "Cellular Processes",
    "Environmental Information Processing",
    "Genetic Information Processing",
    "Metabolism",
    "Organismal Systems",
]
COMPARTMENT_TO_ID = {
    "tumor": 0,
    "stroma": 1,
    "boundary": 2,
}
LABEL_TO_INDEX = {
    "hot": 0,
    "excluded": 1,
    "cold": 2,
}


def build_default_config() -> dict[str, Any]:
    return {
        "input": {
            "dataset_id": "visium_breast",
            "scrna_manifest_path": "outputs/scrna/gse161529__reference_v1/gse161529__reference_v1__step-08_manifest.json",
            "prior_manifest_path": "outputs/priors/string_kegg_v1/string_kegg_v1__step-07_manifest.json",
            "spatial_manifest_path": "outputs/spatial/visium_breast__spatial_v1/visium_breast__spatial_v1__step-03_manifest.json",
            "spatial_expression_path": None,
            "spatial_expression_gene_symbol_column": "gene_symbol",
            "spatial_node_id_column": None,
            "assume_log1p_input": True,
        },
        "labels": {
            "time_label_column": "time_label",
            "is_uncertain_column": "is_uncertain",
            "graph_label_mode": "constant_obs",
            "uncertainty_margin_robust_sd": 0.25,
            "phenotype_columns": {
                "infiltration_score": "infiltration_score",
                "penetration_score": "penetration_score",
                "retention_score": "retention_score",
                "activation_score": "activation_score",
            },
        },
        "features": {
            "cell_type_column": "cell_type",
            "compartment_column": "compartment",
            "tumor_probability_column": "tumor_probability",
            "immune_probability_column": "immune_probability",
            "program_gene_sets": {
                "cytotoxic": ["NKG7", "GNLY", "PRF1", "GZMB", "IFNG"],
                "exhausted": ["PDCD1", "LAG3", "TIGIT", "HAVCR2", "CTLA4"],
                "interferon_response": ["STAT1", "ISG15", "IFIT1", "IFI6", "CXCL10"],
                "cycling": ["MKI67", "TOP2A", "TYMS", "UBE2C", "BIRC5"],
                "stromal_activation": ["COL1A1", "COL3A1", "TAGLN", "ACTA2", "THY1"],
                "epithelial_malignant": ["EPCAM", "KRT8", "KRT18", "KRT19", "ERBB2"],
            },
            "cell_type_markers": {
                "T_NK": ["CD3D", "CD3E", "TRBC1", "TRBC2", "NKG7", "IL7R"],
                "B_Plasma": ["MS4A1", "CD79A", "CD79B", "JCHAIN", "MZB1"],
                "Myeloid": ["LYZ", "LST1", "FCER1G", "TYROBP", "CTSS"],
                "CAF": ["COL1A1", "COL1A2", "DCN", "LUM", "TAGLN"],
                "Endothelial": ["PECAM1", "VWF", "KDR", "EMCN", "RAMP2"],
                "Epithelial_Malignant": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1"],
            },
            "immune_gene_sets": {
                "immune_genes": [
                    "CD3D",
                    "CD3E",
                    "NKG7",
                    "GNLY",
                    "PRF1",
                    "IFNG",
                    "CXCL9",
                    "CXCL10",
                    "LAG3",
                    "PDCD1",
                    "CTLA4",
                    "MS4A1",
                    "CD79A",
                    "LYZ",
                    "FCER1G",
                    "TYROBP",
                ],
                "tumor_immune_signature_genes": [
                    "EPCAM",
                    "KRT8",
                    "KRT18",
                    "KRT19",
                    "ERBB2",
                    "COL1A1",
                    "COL3A1",
                    "TAGLN",
                    "ACTA2",
                    "THY1",
                    "IFNG",
                    "CXCL9",
                    "CXCL10",
                    "GZMB",
                ],
            },
            "immune_pathway_name_patterns": [
                "immune",
                "cytokine",
                "chemokine",
                "antigen",
                "t cell",
                "b cell",
                "nk cell",
                "jak-stat",
                "pd-1",
                "checkpoint",
            ],
        },
        "schema": {
            "cell_pca_dims": 32,
            "cell_gene_top_k_per_cell": 64,
            "max_gene_nodes_per_graph": 2048,
            "min_genes_per_pathway_in_graph": 3,
            "max_pathway_nodes_per_graph": 256,
        },
        "runtime": {
            "max_graphs": None,
        },
        "targets": {
            "gene_positive_ids": [
                "CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "PDCD1LG2", "VSIR",
                "CD80", "CD86", "IDO1", "SIGLEC15", "IFNG", "STAT1", "IRF1", "CXCL9",
                "CXCL10", "CXCL11", "GBP1", "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1",
                "TAP2", "PSMB9", "CD3D", "CD3E", "CD8A", "CD8B", "GZMA", "GZMB", "PRF1",
                "ICOS", "TGFB1", "TGFB2", "VEGFA", "WNT5A", "CTNNB1", "CCL2", "CCL5",
                "CXCL12", "CXCL13", "CD68", "CD163", "CSF1R", "ARG1", "MRC1",
            ],
            "pathway_positive_ids": [],
            "gene_positive_feature_columns": [
                "immune_gene_flag",
                "tumor_immune_signature_flag",
            ],
            "pathway_positive_feature_columns": [
                "immune_pathway_flag",
            ],
            "base_positive_weight": 1.0,
            "positive_strategy": "union",
        },
        "output": {
            "root_dir": "outputs/hetero_graph/visium_breast__hetero_v1",
            "prefix": "visium_breast__hetero_v1",
        },
        "validation": {
            "min_graphs": None,
            "min_bags": None,
            "fail_below_min": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build patient/slice-level heterogeneous graphs for PyG")
    parser.add_argument("--config", type=Path, help="Path to YAML config")
    parser.add_argument(
        "--write-default-config",
        type=Path,
        help="Write the default config template and exit",
    )
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "scripts").is_dir() and (candidate / "models").is_dir():
            return candidate
    raise SystemExit("Could not locate project root via repository structure")


def resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def ensure_descendant(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise SystemExit(f"Path escapes project root: {resolved}") from exc
    return resolved


def load_config(config_path: Path) -> dict[str, Any]:
    default_config = build_default_config()
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    config = deep_update(copy.deepcopy(default_config), user_config)
    config["_config_path"] = str(config_path.resolve())
    return config


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "," if path.suffix == ".csv" else "\t"
    frame.to_csv(path, sep=sep, index=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required table: {path}")
    if path.suffix in {".csv", ".gz"} and path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if path.suffix in {".csv"}:
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def get_series_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def require_h5ad() -> Any:
    if ad is None:
        raise SystemExit(
            "anndata is required for the hetero graph pipeline. "
            "Install dependencies from requirements-hetero.txt"
        )
    return ad


def require_pyg() -> Any:
    if HeteroData is None:
        raise SystemExit(
            "torch_geometric is required for the hetero graph pipeline. "
            "Install dependencies from requirements-hetero.txt"
        )
    return HeteroData


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_symbol(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    return re.sub(r"\s+", "", text).upper()


def robust_scale(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    median = float(numeric.median())
    mad = float((numeric - median).abs().median())
    if mad > 0:
        return mad / 0.6744897501960817
    return float(numeric.std(ddof=0))


def summarize_graph_metadata(
    *,
    spatial_nodes: pd.DataFrame,
    obs_lookup: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    label_cfg = config["labels"]
    label_column = label_cfg["time_label_column"]
    phenotype_columns = list(label_cfg["phenotype_columns"].values())
    graph_label_mode = str(label_cfg.get("graph_label_mode", "constant_obs")).lower()
    if graph_label_mode not in {"constant_obs", "aggregate_rule", "auto"}:
        raise SystemExit(f"Unsupported labels.graph_label_mode: {graph_label_mode}")

    required_columns = ["__node_id__", label_column, *phenotype_columns]
    missing_columns = [column for column in required_columns if column not in obs_lookup.columns]
    if missing_columns:
        raise SystemExit(f"Spatial expression obs is missing required columns: {missing_columns}")

    obs_subset = obs_lookup.reset_index(drop=True)[required_columns].copy()
    merged = spatial_nodes[
        [
            "graph_id",
            "node_id",
            "patient_id",
            "sample_id",
            "slide_id",
            "split_id",
            "cohort_id",
            "label_source",
            "is_uncertain",
        ]
    ].merge(
        obs_subset,
        left_on="node_id",
        right_on="__node_id__",
        how="left",
    )

    grouped = merged.groupby("graph_id", sort=True)
    graph_frame = grouped.agg(
        patient_id=("patient_id", "first"),
        sample_id=("sample_id", "first"),
        slide_id=("slide_id", "first"),
        split_id=("split_id", "first"),
        cohort_id=("cohort_id", "first"),
        label_source=("label_source", "first"),
        any_spot_uncertain=("is_uncertain", "any"),
        n_spots=("node_id", "size"),
        spot_label_mode=(label_column, lambda s: s.mode().iloc[0] if not s.mode().empty else normalize_text(s.iloc[0])),
        spot_label_nunique=(label_column, "nunique"),
        infiltration_score=(phenotype_columns[0], "mean"),
        penetration_score=(phenotype_columns[1], "mean"),
        retention_score=(phenotype_columns[2], "mean"),
        activation_score=(phenotype_columns[3], "mean"),
    )

    if graph_label_mode == "constant_obs" or (
        graph_label_mode == "auto" and int(graph_frame["spot_label_nunique"].max()) == 1
    ):
        graph_frame["time_label"] = graph_frame["spot_label_mode"].astype(str)
        graph_frame["is_uncertain"] = graph_frame["any_spot_uncertain"].astype(bool)
        return graph_frame

    quantiles = {
        "Q40_inf": float(graph_frame["infiltration_score"].quantile(0.4)),
        "Q50_inf": float(graph_frame["infiltration_score"].quantile(0.5)),
        "Q60_pen": float(graph_frame["penetration_score"].quantile(0.6)),
        "Q40_pen": float(graph_frame["penetration_score"].quantile(0.4)),
        "Q75_ret": float(graph_frame["retention_score"].quantile(0.75)),
        "Q60_ret": float(graph_frame["retention_score"].quantile(0.6)),
        "Q50_act": float(graph_frame["activation_score"].quantile(0.5)),
        "Q40_act": float(graph_frame["activation_score"].quantile(0.4)),
    }
    scales = {
        "infiltration_score": robust_scale(graph_frame["infiltration_score"]),
        "penetration_score": robust_scale(graph_frame["penetration_score"]),
        "retention_score": robust_scale(graph_frame["retention_score"]),
        "activation_score": robust_scale(graph_frame["activation_score"]),
    }
    margin_multiplier = float(label_cfg.get("uncertainty_margin_robust_sd", 0.25))

    derived_labels: list[str] = []
    derived_uncertainty: list[bool] = []
    derived_sources: list[str] = []
    for _, row in graph_frame.iterrows():
        hot = (
            row["infiltration_score"] >= quantiles["Q50_inf"]
            and row["penetration_score"] >= quantiles["Q60_pen"]
            and row["activation_score"] >= quantiles["Q50_act"]
            and row["retention_score"] < quantiles["Q75_ret"]
        )
        excluded = (
            row["infiltration_score"] >= quantiles["Q50_inf"]
            and row["penetration_score"] <= quantiles["Q40_pen"]
            and row["retention_score"] >= quantiles["Q60_ret"]
        )
        cold = (
            row["infiltration_score"] <= quantiles["Q40_inf"]
            and row["activation_score"] <= quantiles["Q40_act"]
        )
        matched = [name for name, flag in [("Hot", hot), ("Excluded", excluded), ("Cold", cold)] if flag]
        near_threshold = False
        if margin_multiplier > 0:
            near_threshold = any(
                abs(float(row[column]) - threshold) < margin_multiplier * scales[column]
                for column, threshold in [
                    ("infiltration_score", quantiles["Q40_inf"]),
                    ("infiltration_score", quantiles["Q50_inf"]),
                    ("penetration_score", quantiles["Q40_pen"]),
                    ("penetration_score", quantiles["Q60_pen"]),
                    ("retention_score", quantiles["Q60_ret"]),
                    ("retention_score", quantiles["Q75_ret"]),
                    ("activation_score", quantiles["Q40_act"]),
                    ("activation_score", quantiles["Q50_act"]),
                ]
                if scales[column] > 0
            )
        is_uncertain = (
            bool(row["any_spot_uncertain"])
            or len(matched) != 1
            or near_threshold
            or any(pd.isna(row[column]) for column in phenotype_columns)
        )
        derived_labels.append(matched[0] if matched else str(row["spot_label_mode"]))
        derived_uncertainty.append(is_uncertain)
        derived_sources.append(str(row["label_source"]) + "+aggregate_rule")

    graph_frame["time_label"] = derived_labels
    graph_frame["is_uncertain"] = derived_uncertainty
    graph_frame["label_source"] = derived_sources
    return graph_frame


def find_single_file(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one file matching '*{suffix}' in {directory}, found {len(matches)}")
    return matches[0]


def validate_export_counts(
    *,
    n_graphs: int,
    n_bags: int,
    validation_cfg: dict[str, Any],
    dataset_id: str,
) -> None:
    failures: list[str] = []
    min_graphs = validation_cfg.get("min_graphs")
    min_bags = validation_cfg.get("min_bags")
    if min_graphs is not None and n_graphs < int(min_graphs):
        failures.append(f"graphs={n_graphs} < min_graphs={int(min_graphs)}")
    if min_bags is not None and n_bags < int(min_bags):
        failures.append(f"bags={n_bags} < min_bags={int(min_bags)}")
    if not failures:
        return
    message = (
        f"{dataset_id}: hetero-graph export does not satisfy the configured evaluation contract "
        f"({', '.join(failures)}). The current dataset is too small for the frozen grouped CV setup."
    )
    if bool(validation_cfg.get("fail_below_min", False)):
        raise SystemExit(message)
    print(f"[WARN] {message}")


def load_reference_artifacts(manifest_path: Path, project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = read_json(manifest_path)
    outputs = manifest.get("outputs", {})
    if not outputs:
        raise SystemExit(f"scRNA manifest does not contain an outputs section: {manifest_path}")
    gene_panel_path = ensure_descendant(Path(outputs["gene_panel"]), project_root)
    gene_panel = read_table(gene_panel_path)
    gene_panel["canonical_gene_symbol"] = gene_panel.get("gene_symbol", gene_panel.get("gene_id", "")).astype(str)
    gene_panel["canonical_gene_symbol_norm"] = gene_panel["canonical_gene_symbol"].map(normalize_symbol)
    gene_panel["is_hvg"] = get_series_or_default(gene_panel, "highly_variable", False).fillna(False).astype(bool)

    graph_ready_path = outputs.get("graph_ready_h5ad")
    if graph_ready_path:
        graph_ready_resolved = ensure_descendant(Path(graph_ready_path), project_root)
        ad_module = require_h5ad()
        adata = ad_module.read_h5ad(graph_ready_resolved)
        matrix = adata.X
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=float)
        gene_symbols = (
            adata.var["gene_symbol"].astype(str).tolist()
            if "gene_symbol" in adata.var.columns
            else adata.var_names.astype(str).tolist()
        )
        detected = (matrix > 0).mean(axis=0)
        mean_expr = matrix.mean(axis=0)
        reference_stats = (
            pd.DataFrame(
                {
                    "canonical_gene_symbol": gene_symbols,
                    "canonical_gene_symbol_norm": [normalize_symbol(value) for value in gene_symbols],
                    "reference_mean_log1p_expr": mean_expr,
                    "reference_detection_rate": detected,
                }
            )
            .groupby("canonical_gene_symbol_norm", as_index=False)
            .agg(
                canonical_gene_symbol=("canonical_gene_symbol", "first"),
                reference_mean_log1p_expr=("reference_mean_log1p_expr", "mean"),
                reference_detection_rate=("reference_detection_rate", "mean"),
            )
        )
    else:
        reference_stats = gene_panel[["canonical_gene_symbol", "canonical_gene_symbol_norm"]].copy()
        reference_stats["reference_mean_log1p_expr"] = pd.to_numeric(gene_panel.get("means", 0.0), errors="coerce").fillna(0.0)
        reference_stats["reference_detection_rate"] = 0.0
    merged = reference_stats.merge(
        gene_panel[["canonical_gene_symbol_norm", "is_hvg"]],
        on="canonical_gene_symbol_norm",
        how="left",
    )
    merged["is_hvg"] = merged["is_hvg"].fillna(False).astype(bool)
    return gene_panel, merged


def load_prior_artifacts(manifest_path: Path, project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = read_json(manifest_path)
    output_dir = ensure_descendant(Path(manifest["output_dir"]), project_root)
    gene_master = read_table(find_single_file(output_dir, "__step-04_gene_master.tsv.gz"))
    gene_gene_edges = read_table(find_single_file(output_dir, "__step-05_gene_gene_edges.tsv.gz"))
    gene_pathway_edges = read_table(find_single_file(output_dir, "__step-06_gene_pathway_edges.tsv.gz"))
    if "gene_symbol" in gene_pathway_edges.columns and "gene_symbol_norm" not in gene_pathway_edges.columns:
        gene_pathway_edges["gene_symbol_norm"] = gene_pathway_edges["gene_symbol"].map(normalize_symbol)
    pathway_catalog = read_table(find_single_file(output_dir, "__step-03_pathway_catalog.tsv.gz"))
    return gene_master, gene_gene_edges, gene_pathway_edges, pathway_catalog


def load_spatial_artifacts(
    manifest_path: Path,
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    manifest = read_json(manifest_path)
    outputs = manifest.get("outputs", {})
    if not outputs:
        raise SystemExit(f"Spatial manifest does not contain an outputs section: {manifest_path}")
    nodes = read_table(ensure_descendant(Path(outputs["nodes"]), project_root))
    edges = read_table(ensure_descendant(Path(outputs["edges_primary"]), project_root))
    input_path = ensure_descendant(Path(manifest["input_path"]), project_root)
    return nodes, edges, input_path


def build_spatial_expression_lookup(
    expression_path: Path,
    gene_symbol_column: str | None,
    assume_log1p_input: bool,
) -> tuple[Any, pd.DataFrame]:
    ad_module = require_h5ad()
    adata = ad_module.read_h5ad(expression_path)
    expression = adata.X
    if hasattr(expression, "toarray"):
        expression = expression.toarray()
    expression = np.asarray(expression, dtype=float)
    if not assume_log1p_input:
        expression = np.log1p(np.clip(expression, a_min=0.0, a_max=None))
    gene_symbols = (
        adata.var[gene_symbol_column].astype(str).tolist()
        if gene_symbol_column and gene_symbol_column in adata.var.columns
        else adata.var_names.astype(str).tolist()
    )
    obs_frame = adata.obs.copy()
    obs_frame.index = obs_frame.index.astype(str)
    obs_frame["__obs_index__"] = obs_frame.index
    gene_frame = pd.DataFrame(
        {
            "gene_name": gene_symbols,
            "gene_name_norm": [normalize_symbol(value) for value in gene_symbols],
            "var_index": np.arange(len(gene_symbols), dtype=int),
        }
    )
    return {"adata": adata, "expression": expression}, gene_frame


def first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def bool_from_series(series: pd.Series, default: bool) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    normalized = (
        series.fillna(default)
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "1": True,
                "true": True,
                "t": True,
                "yes": True,
                "y": True,
                "0": False,
                "false": False,
                "f": False,
                "no": False,
                "n": False,
            }
        )
    )
    return normalized.fillna(default).astype(bool)


def build_reference_lookup(
    reference_stats: pd.DataFrame,
    gene_master: pd.DataFrame,
    gene_gene_edges: pd.DataFrame,
    gene_pathway_edges: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    gene_master = gene_master.copy()
    gene_master["canonical_gene_symbol_norm"] = gene_master["canonical_gene_symbol_norm"].astype(str)
    gene_master["canonical_gene_symbol"] = gene_master["canonical_gene_symbol"].astype(str)
    gene_master["in_gene_universe"] = get_series_or_default(gene_master, "in_gene_universe", False).fillna(False).astype(bool)
    gene_master = gene_master.loc[gene_master["in_gene_universe"]].copy()

    degree_parts = []
    if not gene_gene_edges.empty:
        left = gene_gene_edges.rename(
            columns={
                "source_gene_symbol_norm": "canonical_gene_symbol_norm",
                "edge_weight": "gene_gene_edge_weight",
            }
        )[["canonical_gene_symbol_norm", "gene_gene_edge_weight"]]
        right = gene_gene_edges.rename(
            columns={
                "target_gene_symbol_norm": "canonical_gene_symbol_norm",
                "edge_weight": "gene_gene_edge_weight",
            }
        )[["canonical_gene_symbol_norm", "gene_gene_edge_weight"]]
        degree_parts = [left, right]
        gene_degree = (
            pd.concat(degree_parts, ignore_index=True)
            .groupby("canonical_gene_symbol_norm", as_index=False)
            .agg(
                string_degree=("gene_gene_edge_weight", "count"),
                string_strength=("gene_gene_edge_weight", "sum"),
            )
        )
    else:
        gene_degree = pd.DataFrame(columns=["canonical_gene_symbol_norm", "string_degree", "string_strength"])

    membership = (
        gene_pathway_edges.groupby("gene_symbol_norm", as_index=False)
        .agg(pathway_membership_count=("pathway_id", "nunique"))
        .rename(columns={"gene_symbol_norm": "canonical_gene_symbol_norm"})
    )

    immune_gene_set = {
        normalize_symbol(value)
        for value in config["features"]["immune_gene_sets"]["immune_genes"]
    }
    tumor_immune_signature_set = {
        normalize_symbol(value)
        for value in config["features"]["immune_gene_sets"]["tumor_immune_signature_genes"]
    }

    lookup = (
        gene_master[["canonical_gene_symbol", "canonical_gene_symbol_norm"]]
        .drop_duplicates()
        .merge(reference_stats, on="canonical_gene_symbol_norm", how="left")
        .merge(gene_degree, on="canonical_gene_symbol_norm", how="left")
        .merge(membership, on="canonical_gene_symbol_norm", how="left")
    )
    if "canonical_gene_symbol_x" in lookup.columns or "canonical_gene_symbol_y" in lookup.columns:
        lookup["canonical_gene_symbol"] = (
            get_series_or_default(lookup, "canonical_gene_symbol_x", "")
            .replace("", np.nan)
            .fillna(get_series_or_default(lookup, "canonical_gene_symbol_y", ""))
        )
        lookup = lookup.drop(
            columns=[column for column in ["canonical_gene_symbol_x", "canonical_gene_symbol_y"] if column in lookup.columns]
        )
    lookup["reference_mean_log1p_expr"] = pd.to_numeric(lookup["reference_mean_log1p_expr"], errors="coerce").fillna(0.0)
    lookup["reference_detection_rate"] = pd.to_numeric(lookup["reference_detection_rate"], errors="coerce").fillna(0.0)
    lookup["is_hvg"] = lookup.get("is_hvg", False).fillna(False).astype(bool)
    lookup["string_degree"] = pd.to_numeric(lookup.get("string_degree", 0), errors="coerce").fillna(0.0)
    lookup["string_strength"] = pd.to_numeric(lookup.get("string_strength", 0), errors="coerce").fillna(0.0)
    lookup["pathway_membership_count"] = pd.to_numeric(lookup.get("pathway_membership_count", 0), errors="coerce").fillna(0.0)
    lookup["immune_gene_flag"] = lookup["canonical_gene_symbol_norm"].isin(immune_gene_set).astype(float)
    lookup["tumor_immune_signature_flag"] = lookup["canonical_gene_symbol_norm"].isin(tumor_immune_signature_set).astype(float)

    for column in ["string_degree", "string_strength", "pathway_membership_count"]:
        max_value = float(lookup[column].max()) if not lookup.empty else 0.0
        denom = max(max_value, 1.0)
        lookup[f"{column}_norm"] = lookup[column] / denom
    return lookup


def build_gene_index(gene_frame: pd.DataFrame) -> dict[str, list[int]]:
    grouped = gene_frame.loc[gene_frame["gene_name_norm"] != ""].groupby("gene_name_norm")["var_index"]
    return {gene: values.astype(int).tolist() for gene, values in grouped}


def aggregate_gene_expression(
    expression: np.ndarray,
    gene_indices: dict[str, list[int]],
    genes: list[str],
) -> np.ndarray:
    if expression.size == 0 or not genes:
        return np.zeros((expression.shape[0], 0), dtype=np.float32)
    columns: list[np.ndarray] = []
    for gene in genes:
        indices = gene_indices.get(gene, [])
        if not indices:
            columns.append(np.zeros(expression.shape[0], dtype=np.float32))
            continue
        values = expression[:, indices]
        if values.ndim == 1:
            aggregated = values
        else:
            aggregated = values.sum(axis=1)
        columns.append(np.asarray(aggregated, dtype=np.float32))
    return np.stack(columns, axis=1).astype(np.float32)


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _positive_match_count_from_flags(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if frame.empty:
        return np.zeros((0,), dtype=np.float32)
    match_count = np.zeros((len(frame),), dtype=np.float32)
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        match_count += (values > 0).astype(np.float32)
    return match_count


def build_target_supervision(
    gene_nodes: pd.DataFrame,
    pathway_nodes: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    target_cfg = config.get("targets", {})
    positive_strategy = str(target_cfg.get("positive_strategy", "union")).lower()
    if positive_strategy != "union":
        raise SystemExit(f"Unsupported targets.positive_strategy: {positive_strategy}")

    base_positive_weight = float(target_cfg.get("base_positive_weight", 1.0))

    gene_positive_ids = {
        normalize_symbol(value)
        for value in target_cfg.get("gene_positive_ids", [])
        if normalize_symbol(value)
    }
    pathway_positive_ids = {
        normalize_text(value)
        for value in target_cfg.get("pathway_positive_ids", [])
        if normalize_text(value)
    }

    gene_feature_matches = _positive_match_count_from_flags(
        gene_nodes,
        [str(column) for column in target_cfg.get("gene_positive_feature_columns", [])],
    )
    gene_id_matches = (
        gene_nodes["canonical_gene_symbol_norm"].astype(str).isin(gene_positive_ids).to_numpy(dtype=bool).astype(np.float32)
        if not gene_nodes.empty
        else np.zeros((0,), dtype=np.float32)
    )
    gene_match_count = gene_feature_matches + gene_id_matches
    gene_positive_mask = gene_match_count > 0
    gene_target_weight = np.where(
        gene_positive_mask,
        base_positive_weight + gene_match_count,
        0.0,
    ).astype(np.float32)

    pathway_feature_matches = _positive_match_count_from_flags(
        pathway_nodes,
        [str(column) for column in target_cfg.get("pathway_positive_feature_columns", [])],
    )
    pathway_id_matches = (
        pathway_nodes["pathway_id"].astype(str).isin(pathway_positive_ids).to_numpy(dtype=bool).astype(np.float32)
        if not pathway_nodes.empty
        else np.zeros((0,), dtype=np.float32)
    )
    pathway_match_count = pathway_feature_matches + pathway_id_matches
    pathway_positive_mask = pathway_match_count > 0
    pathway_target_weight = np.where(
        pathway_positive_mask,
        base_positive_weight + pathway_match_count,
        0.0,
    ).astype(np.float32)

    return {
        "gene": {
            "positive_mask": gene_positive_mask.astype(bool),
            "target_weight": gene_target_weight,
            "match_count": gene_match_count,
        },
        "pathway": {
            "positive_mask": pathway_positive_mask.astype(bool),
            "target_weight": pathway_target_weight,
            "match_count": pathway_match_count,
        },
    }


def feature_score_matrix(
    matrix: np.ndarray,
    gene_order: list[str],
    gene_sets: dict[str, list[str]],
) -> pd.DataFrame:
    gene_to_index = {gene: index for index, gene in enumerate(gene_order)}
    scores: dict[str, np.ndarray] = {}
    for name, raw_genes in gene_sets.items():
        indices = [gene_to_index[normalize_symbol(gene)] for gene in raw_genes if normalize_symbol(gene) in gene_to_index]
        if not indices:
            scores[name] = np.zeros(matrix.shape[0], dtype=np.float32)
            continue
        scores[name] = matrix[:, indices].mean(axis=1).astype(np.float32)
    return pd.DataFrame(scores)


def build_cell_type_scores(
    obs: pd.DataFrame,
    expression: np.ndarray,
    gene_order: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    direct_scores = {}
    for name in CELL_TYPE_ORDER:
        candidates = [
            f"type_score__{name}",
            f"score__cell_type__{name}",
            f"score__{name}",
        ]
        column = first_existing_column(obs, candidates)
        if column:
            direct_scores[name] = safe_numeric(obs[column]).to_numpy(dtype=np.float32)
    if len(direct_scores) == len(CELL_TYPE_ORDER):
        return pd.DataFrame(direct_scores)

    derived = feature_score_matrix(
        expression,
        gene_order,
        config["features"]["cell_type_markers"],
    )
    derived = derived.reindex(columns=CELL_TYPE_ORDER, fill_value=0.0)

    cell_type_column = config["features"].get("cell_type_column")
    if cell_type_column and cell_type_column in obs.columns:
        labels = obs[cell_type_column].astype(str).str.strip()
        for name in CELL_TYPE_ORDER:
            if name not in direct_scores:
                derived[name] = np.maximum(
                    derived[name].to_numpy(dtype=np.float32),
                    (labels == name).astype(np.float32).to_numpy(),
                )
    for name in CELL_TYPE_ORDER:
        if name in direct_scores:
            derived[name] = direct_scores[name]
    return derived


def build_cell_features(
    graph_nodes: pd.DataFrame,
    obs: pd.DataFrame,
    cell_expression: np.ndarray,
    gene_order: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str], pd.DataFrame, pd.DataFrame]:
    cell_pca_dims = int(config["schema"]["cell_pca_dims"])
    if cell_expression.shape[0] == 0:
        raise SystemExit("Encountered empty graph during cell feature construction")

    n_components = min(cell_pca_dims, cell_expression.shape[0], max(cell_expression.shape[1], 1))
    if n_components > 0 and cell_expression.shape[1] > 0:
        pca = PCA(n_components=n_components, random_state=0)
        pca_values = pca.fit_transform(cell_expression)
    else:
        pca_values = np.zeros((cell_expression.shape[0], 0), dtype=np.float32)
    if pca_values.shape[1] < cell_pca_dims:
        pca_values = np.pad(
            pca_values,
            ((0, 0), (0, cell_pca_dims - pca_values.shape[1])),
            mode="constant",
        )
    pca_columns = [f"expr_pca_{index:02d}" for index in range(1, cell_pca_dims + 1)]
    pca_frame = pd.DataFrame(pca_values.astype(np.float32), columns=pca_columns)

    program_scores = feature_score_matrix(
        cell_expression,
        gene_order,
        config["features"]["program_gene_sets"],
    ).reindex(columns=PROGRAM_ORDER, fill_value=0.0)
    program_scores.columns = [f"program_score__{name}" for name in PROGRAM_ORDER]

    type_scores = build_cell_type_scores(obs, cell_expression, gene_order, config)
    type_scores = type_scores.reindex(columns=CELL_TYPE_ORDER, fill_value=0.0)
    type_scores.columns = [f"type_score__{name}" for name in CELL_TYPE_ORDER]

    tumor_col = config["features"].get("tumor_probability_column")
    if tumor_col and tumor_col in obs.columns:
        tumor_probability = safe_numeric(obs[tumor_col]).clip(lower=0.0, upper=1.0)
    else:
        tumor_probability = (
            graph_nodes["compartment"].astype(str).str.lower().eq("tumor").astype(float) * 0.8
            + type_scores["type_score__Epithelial_Malignant"] * 0.2
        ).clip(0.0, 1.0)

    immune_col = config["features"].get("immune_probability_column")
    if immune_col and immune_col in obs.columns:
        immune_probability = safe_numeric(obs[immune_col]).clip(lower=0.0, upper=1.0)
    else:
        immune_probability = type_scores[
            ["type_score__T_NK", "type_score__B_Plasma", "type_score__Myeloid"]
        ].max(axis=1).clip(0.0, 1.0)

    compartments = graph_nodes["compartment"].astype(str).str.lower()
    compartment_frame = pd.DataFrame(
        {
            "compartment__tumor": compartments.eq("tumor").astype(np.float32),
            "compartment__stroma": compartments.eq("stroma").astype(np.float32),
            "compartment__boundary": compartments.eq("boundary").astype(np.float32),
        }
    )
    total_counts = np.expm1(cell_expression).sum(axis=1)
    detected_genes = (cell_expression > 0).sum(axis=1)
    count_frame = pd.DataFrame(
        {
            "tumor_probability": tumor_probability.to_numpy(dtype=np.float32),
            "immune_probability": immune_probability.to_numpy(dtype=np.float32),
            "log_total_counts": np.log1p(total_counts).astype(np.float32),
            "log_detected_genes": np.log1p(detected_genes).astype(np.float32),
        }
    )

    cell_features = pd.concat(
        [
            pca_frame,
            program_scores,
            type_scores,
            count_frame[["tumor_probability"]],
            compartment_frame,
            count_frame[["log_total_counts", "log_detected_genes"]],
        ],
        axis=1,
    )
    feature_names = cell_features.columns.tolist()
    return cell_features.to_numpy(dtype=np.float32), feature_names, type_scores, count_frame


def build_cell_gene_edges(
    cell_expression: np.ndarray,
    gene_order: list[str],
    top_k: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if cell_expression.shape[1] == 0:
        return pd.DataFrame(
            columns=[
                "cell_index",
                "gene_symbol_norm",
                "raw_log1p_expr",
                "within_cell_rank",
                "within_cell_rank_norm",
                "edge_weight",
            ]
        )
    for cell_index in range(cell_expression.shape[0]):
        row = np.asarray(cell_expression[cell_index], dtype=float)
        nonzero = np.flatnonzero(row > 0)
        if nonzero.size == 0:
            continue
        if nonzero.size > top_k:
            top = nonzero[np.argpartition(row[nonzero], -top_k)[-top_k:]]
            top = top[np.argsort(row[top])[::-1]]
        else:
            top = nonzero[np.argsort(row[nonzero])[::-1]]
        retained = row[top]
        denom = max(float(retained.sum()), 1.0e-8)
        for rank_index, gene_index in enumerate(top):
            rank_norm = 1.0 - (rank_index / max(len(top) - 1, 1))
            records.append(
                {
                    "cell_index": int(cell_index),
                    "gene_symbol_norm": gene_order[int(gene_index)],
                    "raw_log1p_expr": float(row[gene_index]),
                    "within_cell_rank": int(rank_index),
                    "within_cell_rank_norm": float(rank_norm),
                    "edge_weight": float(row[gene_index] / denom),
                }
            )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "cell_index",
                "gene_symbol_norm",
                "raw_log1p_expr",
                "within_cell_rank",
                "within_cell_rank_norm",
                "edge_weight",
            ]
        )
    return df


def build_gene_nodes(
    cell_gene_edges: pd.DataFrame,
    cell_expression: np.ndarray,
    gene_order: list[str],
    reference_lookup: pd.DataFrame,
    max_gene_nodes: int,
) -> pd.DataFrame:
    if cell_gene_edges.empty:
        return pd.DataFrame(
            columns=[
                "canonical_gene_symbol_norm",
                "canonical_gene_symbol",
                "sample_mean_log1p_expr",
                "sample_detection_rate",
                "reference_mean_log1p_expr",
                "reference_detection_rate",
                "is_hvg",
                "string_degree_norm",
                "string_strength_norm",
                "pathway_membership_count_norm",
                "immune_gene_flag",
                "tumor_immune_signature_flag",
                "n_incident_cells",
                "string_degree",
            ]
        )
    gene_to_index = {gene: index for index, gene in enumerate(gene_order)}
    incident = (
        cell_gene_edges.groupby("gene_symbol_norm", as_index=False)
        .agg(n_incident_cells=("cell_index", "count"))
        .rename(columns={"gene_symbol_norm": "canonical_gene_symbol_norm"})
    )
    stats = []
    for gene_norm in incident["canonical_gene_symbol_norm"].tolist():
        matrix_column = gene_to_index[gene_norm]
        values = cell_expression[:, matrix_column]
        stats.append(
            {
                "canonical_gene_symbol_norm": gene_norm,
                "sample_mean_log1p_expr": float(values.mean()),
                "sample_detection_rate": float((values > 0).mean()),
            }
        )
    gene_stats = pd.DataFrame.from_records(stats)
    gene_nodes = (
        incident.merge(gene_stats, on="canonical_gene_symbol_norm", how="left")
        .merge(reference_lookup, on="canonical_gene_symbol_norm", how="left")
    )
    gene_nodes["canonical_gene_symbol"] = gene_nodes["canonical_gene_symbol"].fillna(gene_nodes["canonical_gene_symbol_norm"])
    gene_nodes["reference_mean_log1p_expr"] = gene_nodes["reference_mean_log1p_expr"].fillna(0.0)
    gene_nodes["reference_detection_rate"] = gene_nodes["reference_detection_rate"].fillna(0.0)
    gene_nodes["is_hvg"] = gene_nodes["is_hvg"].fillna(False).astype(bool)
    gene_nodes["string_degree_norm"] = gene_nodes["string_degree_norm"].fillna(0.0)
    gene_nodes["string_strength_norm"] = gene_nodes["string_strength_norm"].fillna(0.0)
    gene_nodes["pathway_membership_count_norm"] = gene_nodes["pathway_membership_count_norm"].fillna(0.0)
    gene_nodes["immune_gene_flag"] = gene_nodes["immune_gene_flag"].fillna(0.0)
    gene_nodes["tumor_immune_signature_flag"] = gene_nodes["tumor_immune_signature_flag"].fillna(0.0)
    gene_nodes["string_degree"] = gene_nodes.get("string_degree", 0.0).fillna(0.0)
    gene_nodes = gene_nodes.sort_values(
        ["n_incident_cells", "sample_mean_log1p_expr", "string_degree"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    if len(gene_nodes) > max_gene_nodes:
        gene_nodes = gene_nodes.head(max_gene_nodes).copy()
    gene_nodes["in_graph_rank"] = np.arange(len(gene_nodes), dtype=int)
    return gene_nodes


def build_pathway_nodes(
    gene_nodes: pd.DataFrame,
    gene_pathway_edges: pd.DataFrame,
    pathway_catalog: pd.DataFrame,
    min_genes_per_pathway: int,
    max_pathway_nodes: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if gene_nodes.empty:
        empty_pathways = pd.DataFrame(
            columns=[
                "pathway_id",
                "pathway_name",
                "pathway_top_class",
                "total_mapped_genes",
                "mapped_gene_count",
                "member_gene_count_log1p",
                "mapped_gene_count_in_graph_log1p",
                "mapped_gene_coverage_ratio",
                "sample_member_expr_mean",
                "sample_member_expr_std",
                "immune_pathway_flag",
            ]
        )
        return empty_pathways, pd.DataFrame(columns=gene_pathway_edges.columns)

    retained_genes = set(gene_nodes["canonical_gene_symbol_norm"].astype(str))
    pathway_edges = gene_pathway_edges.loc[gene_pathway_edges["gene_symbol_norm"].isin(retained_genes)].copy()
    if pathway_edges.empty:
        empty_pathways = pd.DataFrame(
            columns=[
                "pathway_id",
                "pathway_name",
                "pathway_top_class",
                "total_mapped_genes",
                "mapped_gene_count",
                "member_gene_count_log1p",
                "mapped_gene_count_in_graph_log1p",
                "mapped_gene_coverage_ratio",
                "sample_member_expr_mean",
                "sample_member_expr_std",
                "immune_pathway_flag",
            ]
        )
        return empty_pathways, pathway_edges

    pathway_edges = pathway_edges.merge(
        gene_nodes[["canonical_gene_symbol_norm", "sample_mean_log1p_expr"]],
        left_on="gene_symbol_norm",
        right_on="canonical_gene_symbol_norm",
        how="left",
    )
    pathway_edges["sample_mean_log1p_expr"] = pathway_edges["sample_mean_log1p_expr"].fillna(0.0)
    summary = (
        pathway_edges.groupby("pathway_id", as_index=False)
        .agg(
            mapped_gene_count=("gene_symbol_norm", "nunique"),
            sample_member_expr_mean=("sample_mean_log1p_expr", "mean"),
            sample_member_expr_std=("sample_mean_log1p_expr", "std"),
        )
    )
    catalog = pathway_catalog.copy()
    catalog["selected_for_graph"] = get_series_or_default(catalog, "selected_for_graph", False).fillna(False).astype(bool)
    catalog = catalog.loc[catalog["selected_for_graph"]].copy()
    merged = summary.merge(
        catalog[
            [
                "pathway_id",
                "pathway_name",
                "pathway_top_class",
                "total_mapped_genes",
            ]
        ],
        on="pathway_id",
        how="left",
    )
    merged["total_mapped_genes"] = pd.to_numeric(merged["total_mapped_genes"], errors="coerce").fillna(0.0)
    merged["sample_member_expr_std"] = merged["sample_member_expr_std"].fillna(0.0)
    merged = merged.loc[merged["mapped_gene_count"] >= int(min_genes_per_pathway)].copy()
    if merged.empty:
        return merged, pathway_edges.iloc[0:0].copy()

    merged["mapped_gene_coverage_ratio"] = (
        merged["mapped_gene_count"] / merged["total_mapped_genes"].replace(0, np.nan)
    ).fillna(0.0)
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in config["features"]["immune_pathway_name_patterns"]]
    merged["immune_pathway_flag"] = merged["pathway_name"].fillna("").map(
        lambda name: float(any(pattern.search(str(name)) for pattern in patterns))
    )
    merged["member_gene_count_log1p"] = np.log1p(merged["total_mapped_genes"].clip(lower=0.0))
    merged["mapped_gene_count_in_graph_log1p"] = np.log1p(merged["mapped_gene_count"].clip(lower=0.0))
    merged = merged.sort_values(
        ["mapped_gene_coverage_ratio", "sample_member_expr_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)
    if len(merged) > max_pathway_nodes:
        merged = merged.head(max_pathway_nodes).copy()
    keep_pathways = set(merged["pathway_id"].astype(str))
    pathway_edges = pathway_edges.loc[pathway_edges["pathway_id"].isin(keep_pathways)].copy()
    return merged, pathway_edges


def build_edge_index(frame: pd.DataFrame, source_col: str, target_col: str) -> torch.Tensor:
    if frame.empty:
        return torch.empty((2, 0), dtype=torch.long)
    stacked = np.vstack(
        [
            frame[source_col].to_numpy(dtype=np.int64),
            frame[target_col].to_numpy(dtype=np.int64),
        ]
    )
    return torch.tensor(stacked, dtype=torch.long)


def build_edge_tensor(frame: pd.DataFrame, columns: list[str]) -> torch.Tensor:
    if frame.empty:
        return torch.empty((0, len(columns)), dtype=torch.float32)
    return torch.tensor(frame[columns].to_numpy(dtype=np.float32), dtype=torch.float32)


def duplicate_bidirectional(frame: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    swapped = frame.copy()
    swapped[source_col] = frame[target_col].to_numpy()
    swapped[target_col] = frame[source_col].to_numpy()
    return pd.concat([frame, swapped], ignore_index=True)


def build_graph_object(
    graph_nodes: pd.DataFrame,
    graph_obs: pd.DataFrame,
    graph_meta: pd.Series,
    cell_expression: np.ndarray,
    gene_order: list[str],
    graph_edges: pd.DataFrame,
    reference_lookup: pd.DataFrame,
    gene_gene_edges: pd.DataFrame,
    gene_pathway_edges: pd.DataFrame,
    pathway_catalog: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    require_pyg()
    cell_features, cell_feature_names, type_scores, cell_aux = build_cell_features(
        graph_nodes=graph_nodes,
        obs=graph_obs,
        cell_expression=cell_expression,
        gene_order=gene_order,
        config=config,
    )
    cell_gene_edges = build_cell_gene_edges(
        cell_expression=cell_expression,
        gene_order=gene_order,
        top_k=int(config["schema"]["cell_gene_top_k_per_cell"]),
    )
    gene_nodes = build_gene_nodes(
        cell_gene_edges=cell_gene_edges,
        cell_expression=cell_expression,
        gene_order=gene_order,
        reference_lookup=reference_lookup,
        max_gene_nodes=int(config["schema"]["max_gene_nodes_per_graph"]),
    )
    retained_genes = set(gene_nodes["canonical_gene_symbol_norm"].astype(str))
    cell_gene_edges = cell_gene_edges.loc[cell_gene_edges["gene_symbol_norm"].isin(retained_genes)].copy()

    pathway_nodes, pathway_edges = build_pathway_nodes(
        gene_nodes=gene_nodes,
        gene_pathway_edges=gene_pathway_edges,
        pathway_catalog=pathway_catalog,
        min_genes_per_pathway=int(config["schema"]["min_genes_per_pathway_in_graph"]),
        max_pathway_nodes=int(config["schema"]["max_pathway_nodes_per_graph"]),
        config=config,
    )

    cell_index_lookup = {
        int(local_idx): index
        for index, local_idx in enumerate(graph_nodes["local_node_index"].astype(int).tolist())
    }
    gene_index_lookup = {
        gene: index
        for index, gene in enumerate(gene_nodes["canonical_gene_symbol_norm"].astype(str).tolist())
    }
    pathway_index_lookup = {
        pathway: index
        for index, pathway in enumerate(pathway_nodes["pathway_id"].astype(str).tolist())
    }

    spatial_edges = graph_edges.copy()
    spatial_edges["source_idx"] = spatial_edges["source_local_index"].map(cell_index_lookup)
    spatial_edges["target_idx"] = spatial_edges["target_local_index"].map(cell_index_lookup)
    spatial_edges = spatial_edges.dropna(subset=["source_idx", "target_idx"]).copy()
    spatial_edges["source_idx"] = spatial_edges["source_idx"].astype(int)
    spatial_edges["target_idx"] = spatial_edges["target_idx"].astype(int)
    spatial_edges["same_compartment"] = spatial_edges["same_compartment"].astype(float)
    spatial_edges = duplicate_bidirectional(spatial_edges, "source_idx", "target_idx")

    cell_gene_edges["cell_idx"] = cell_gene_edges["cell_index"].astype(int)
    cell_gene_edges["gene_idx"] = cell_gene_edges["gene_symbol_norm"].map(gene_index_lookup)
    cell_gene_edges = cell_gene_edges.dropna(subset=["gene_idx"]).copy()
    cell_gene_edges["gene_idx"] = cell_gene_edges["gene_idx"].astype(int)

    gene_gene = gene_gene_edges.loc[
        gene_gene_edges["source_gene_symbol_norm"].isin(retained_genes)
        & gene_gene_edges["target_gene_symbol_norm"].isin(retained_genes)
    ].copy()
    gene_gene["source_idx"] = gene_gene["source_gene_symbol_norm"].map(gene_index_lookup)
    gene_gene["target_idx"] = gene_gene["target_gene_symbol_norm"].map(gene_index_lookup)
    gene_gene = gene_gene.dropna(subset=["source_idx", "target_idx"]).copy()
    gene_gene["source_idx"] = gene_gene["source_idx"].astype(int)
    gene_gene["target_idx"] = gene_gene["target_idx"].astype(int)
    gene_gene = duplicate_bidirectional(gene_gene, "source_idx", "target_idx")

    pathway_edges["gene_idx"] = pathway_edges["gene_symbol_norm"].map(gene_index_lookup)
    pathway_edges["pathway_idx"] = pathway_edges["pathway_id"].map(pathway_index_lookup)
    pathway_edges = pathway_edges.dropna(subset=["gene_idx", "pathway_idx"]).copy()
    pathway_edges["gene_idx"] = pathway_edges["gene_idx"].astype(int)
    pathway_edges["pathway_idx"] = pathway_edges["pathway_idx"].astype(int)
    pathway_edges["immune_pathway_flag"] = pathway_edges["pathway_name"].astype(str).map(
        lambda name: float(
            any(
                re.search(pattern, name, flags=re.IGNORECASE)
                for pattern in config["features"]["immune_pathway_name_patterns"]
            )
        )
    )

    data = HeteroData()
    graph_row = graph_nodes.iloc[0]
    label_column = config["labels"]["time_label_column"]
    label_text = normalize_text(graph_meta[label_column]).lower()
    if label_text not in LABEL_TO_INDEX:
        raise SystemExit(f"Unsupported TIME label '{graph_meta[label_column]}' for graph {graph_row['graph_id']}")

    pheno_values = []
    pheno_mask = []
    for _, column_name in config["labels"]["phenotype_columns"].items():
        if column_name in graph_obs.columns:
            value = pd.to_numeric(graph_meta[column_name], errors="coerce")
            pheno_values.append(0.0 if pd.isna(value) else float(value))
            pheno_mask.append(not pd.isna(value))
        else:
            pheno_values.append(0.0)
            pheno_mask.append(False)

    data.graph_id = str(graph_row["graph_id"])
    data.bag_id = f"{graph_row['patient_id']}|{graph_row['sample_id']}"
    data.patient_id = str(graph_row["patient_id"])
    data.sample_id = str(graph_row["sample_id"])
    data.slide_id = str(graph_row["slide_id"])
    data.split_id = str(graph_row["split_id"])
    data.cohort_id = str(graph_row["cohort_id"])
    data.label_source = str(graph_meta["label_source"])
    data.cell_resolution = "spot_proxy"
    data.spatial_variant = "scaled_radius"
    data.y_graph = torch.tensor([LABEL_TO_INDEX[label_text]], dtype=torch.long)
    label_mask = not bool(graph_meta["is_uncertain"])
    data.label_mask = torch.tensor([label_mask], dtype=torch.bool)
    data.y_pheno = torch.tensor(pheno_values, dtype=torch.float32)
    data.pheno_mask = torch.tensor(pheno_mask, dtype=torch.bool)

    data["cell"].x = torch.tensor(cell_features, dtype=torch.float32)
    data["cell"].pos = torch.tensor(
        graph_nodes[["coord_x_scaled", "coord_y_scaled"]].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    data["cell"].node_id = graph_nodes["node_id"].astype(str).tolist()
    data["cell"].local_node_index = torch.tensor(graph_nodes["local_node_index"].to_numpy(dtype=np.int64), dtype=torch.long)
    compartment_ids = graph_nodes["compartment"].astype(str).str.lower().map(COMPARTMENT_TO_ID).fillna(-1)
    data["cell"].compartment_id = torch.tensor(compartment_ids.to_numpy(dtype=np.int64), dtype=torch.long)
    data["cell"].tumor_probability = torch.tensor(cell_aux["tumor_probability"].to_numpy(dtype=np.float32), dtype=torch.float32)
    data["cell"].immune_probability = torch.tensor(cell_aux["immune_probability"].to_numpy(dtype=np.float32), dtype=torch.float32)

    gene_feature_frame = gene_nodes[
        [
            "sample_mean_log1p_expr",
            "sample_detection_rate",
            "reference_mean_log1p_expr",
            "reference_detection_rate",
            "is_hvg",
            "string_degree_norm",
            "string_strength_norm",
            "pathway_membership_count_norm",
            "immune_gene_flag",
            "tumor_immune_signature_flag",
        ]
    ].copy()
    gene_feature_frame["is_hvg"] = gene_feature_frame["is_hvg"].astype(float)
    data["gene"].x = torch.tensor(gene_feature_frame.to_numpy(dtype=np.float32), dtype=torch.float32)
    data["gene"].node_id = gene_nodes["canonical_gene_symbol"].astype(str).tolist()
    data["gene"].n_incident_cells = torch.tensor(gene_nodes["n_incident_cells"].to_numpy(dtype=np.int32), dtype=torch.int32)
    data["gene"].in_graph_rank = torch.tensor(gene_nodes["in_graph_rank"].to_numpy(dtype=np.int32), dtype=torch.int32)

    if pathway_nodes.empty:
        pathway_feature_matrix = np.zeros((0, 11), dtype=np.float32)
        pathway_ids: list[str] = []
        mapped_gene_count = np.zeros((0,), dtype=np.int32)
    else:
        top_class_one_hot = pd.DataFrame(
            {
                f"top_class__{name.replace(' ', '_')}": pathway_nodes["pathway_top_class"].astype(str).eq(name).astype(float)
                for name in TOP_CLASS_ORDER
            }
        )
        pathway_feature_matrix = pd.concat(
            [
                pathway_nodes[
                    [
                        "member_gene_count_log1p",
                        "mapped_gene_count_in_graph_log1p",
                        "mapped_gene_coverage_ratio",
                        "sample_member_expr_mean",
                        "sample_member_expr_std",
                        "immune_pathway_flag",
                    ]
                ],
                top_class_one_hot,
            ],
            axis=1,
        ).to_numpy(dtype=np.float32)
        pathway_ids = pathway_nodes["pathway_id"].astype(str).tolist()
        mapped_gene_count = pathway_nodes["mapped_gene_count"].to_numpy(dtype=np.int32)
    data["pathway"].x = torch.tensor(pathway_feature_matrix, dtype=torch.float32)
    data["pathway"].node_id = pathway_ids
    data["pathway"].mapped_gene_count = torch.tensor(mapped_gene_count, dtype=torch.int32)

    target_supervision = build_target_supervision(gene_nodes=gene_nodes, pathway_nodes=pathway_nodes, config=config)
    data["gene"].target_pos_mask = torch.tensor(
        target_supervision["gene"]["positive_mask"],
        dtype=torch.bool,
    )
    data["gene"].target_weight = torch.tensor(
        target_supervision["gene"]["target_weight"],
        dtype=torch.float32,
    )
    data["pathway"].target_pos_mask = torch.tensor(
        target_supervision["pathway"]["positive_mask"],
        dtype=torch.bool,
    )
    data["pathway"].target_weight = torch.tensor(
        target_supervision["pathway"]["target_weight"],
        dtype=torch.float32,
    )

    data["cell", "spatial", "cell"].edge_index = build_edge_index(spatial_edges, "source_idx", "target_idx")
    data["cell", "spatial", "cell"].edge_attr = build_edge_tensor(
        spatial_edges,
        ["edge_weight", "distance_normalized", "same_compartment"],
    )
    data["cell", "spatial", "cell"].edge_weight = torch.tensor(
        spatial_edges["edge_weight"].to_numpy(dtype=np.float32) if not spatial_edges.empty else np.zeros((0,), dtype=np.float32),
        dtype=torch.float32,
    )

    data["cell", "expresses", "gene"].edge_index = build_edge_index(cell_gene_edges, "cell_idx", "gene_idx")
    data["cell", "expresses", "gene"].edge_attr = build_edge_tensor(
        cell_gene_edges,
        ["raw_log1p_expr", "within_cell_rank_norm"],
    )
    data["cell", "expresses", "gene"].edge_weight = torch.tensor(
        cell_gene_edges["edge_weight"].to_numpy(dtype=np.float32) if not cell_gene_edges.empty else np.zeros((0,), dtype=np.float32),
        dtype=torch.float32,
    )

    data["gene", "rev_expresses", "cell"].edge_index = build_edge_index(
        cell_gene_edges.rename(columns={"cell_idx": "target_idx", "gene_idx": "source_idx"}),
        "source_idx",
        "target_idx",
    )
    data["gene", "rev_expresses", "cell"].edge_weight = data["cell", "expresses", "gene"].edge_weight.clone()

    data["gene", "interacts", "gene"].edge_index = build_edge_index(gene_gene, "source_idx", "target_idx")
    data["gene", "interacts", "gene"].edge_attr = build_edge_tensor(
        gene_gene,
        ["combined_score_mean", "support_edge_count"],
    )
    data["gene", "interacts", "gene"].edge_weight = torch.tensor(
        gene_gene["edge_weight"].to_numpy(dtype=np.float32) if not gene_gene.empty else np.zeros((0,), dtype=np.float32),
        dtype=torch.float32,
    )

    data["gene", "rev_interacts", "gene"].edge_index = build_edge_index(
        gene_gene.rename(columns={"source_idx": "target_idx", "target_idx": "source_idx"}),
        "source_idx",
        "target_idx",
    )
    data["gene", "rev_interacts", "gene"].edge_weight = data["gene", "interacts", "gene"].edge_weight.clone()

    data["gene", "in_pathway", "pathway"].edge_index = build_edge_index(pathway_edges, "gene_idx", "pathway_idx")
    data["gene", "in_pathway", "pathway"].edge_attr = build_edge_tensor(
        pathway_edges,
        ["mapped_gene_count", "immune_pathway_flag"],
    )
    data["gene", "in_pathway", "pathway"].edge_weight = torch.tensor(
        pathway_edges["edge_weight"].to_numpy(dtype=np.float32) if not pathway_edges.empty else np.zeros((0,), dtype=np.float32),
        dtype=torch.float32,
    )

    data["pathway", "rev_in_pathway", "gene"].edge_index = build_edge_index(
        pathway_edges.rename(columns={"gene_idx": "target_idx", "pathway_idx": "source_idx"}),
        "source_idx",
        "target_idx",
    )
    data["pathway", "rev_in_pathway", "gene"].edge_weight = data["gene", "in_pathway", "pathway"].edge_weight.clone()

    graph_summary = {
        "graph_id": str(graph_row["graph_id"]),
        "bag_id": data.bag_id,
        "patient_id": data.patient_id,
        "sample_id": data.sample_id,
        "slide_id": data.slide_id,
        "split_id": data.split_id,
        "cohort_id": data.cohort_id,
        "label_source": data.label_source,
        "time_label": str(graph_meta[label_column]),
        "label_mask": bool(label_mask),
        "n_cell_nodes": int(len(graph_nodes)),
        "n_gene_nodes": int(len(gene_nodes)),
        "n_pathway_nodes": int(len(pathway_nodes)),
        "n_gene_targets": int(target_supervision["gene"]["positive_mask"].sum()),
        "n_pathway_targets": int(target_supervision["pathway"]["positive_mask"].sum()),
        "n_edges_primary": int(spatial_edges.shape[0]),
        "n_edges_expresses": int(cell_gene_edges.shape[0]),
        "n_edges_interacts": int(gene_gene.shape[0]),
        "n_edges_in_pathway": int(pathway_edges.shape[0]),
    }
    feature_schema = {
        "cell_feature_names": cell_feature_names,
        "gene_feature_names": [
            "sample_mean_log1p_expr",
            "sample_detection_rate",
            "reference_mean_log1p_expr",
            "reference_detection_rate",
            "is_hvg",
            "string_degree_norm",
            "string_strength_norm",
            "pathway_membership_count_norm",
            "immune_gene_flag",
            "tumor_immune_signature_flag",
        ],
        "pathway_feature_names": [
            "member_gene_count_log1p",
            "mapped_gene_count_in_graph_log1p",
            "mapped_gene_coverage_ratio",
            "sample_member_expr_mean",
            "sample_member_expr_std",
            "immune_pathway_flag",
            "top_class__Cellular_Processes",
            "top_class__Environmental_Information_Processing",
            "top_class__Genetic_Information_Processing",
            "top_class__Metabolism",
            "top_class__Organismal_Systems",
        ],
        "cell_type_score_names": CELL_TYPE_ORDER,
        "program_score_names": PROGRAM_ORDER,
    }
    return data, {"summary": graph_summary, "feature_schema": feature_schema}


def output_paths(root_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "graphs_dir": root_dir / "graphs",
        "graph_index": root_dir / f"{prefix}__step-01_graph_index.tsv.gz",
        "sample_bags": root_dir / f"{prefix}__step-02_sample_bags.tsv.gz",
        "feature_schema": root_dir / f"{prefix}__step-03_feature_schema.json",
        "manifest": root_dir / f"{prefix}__step-04_manifest.json",
    }


def main() -> None:
    args = parse_args()
    if args.write_default_config:
        write_yaml(args.write_default_config, build_default_config())
        return
    if not args.config:
        raise SystemExit("Use --config or --write-default-config")

    config = load_config(args.config.resolve())
    project_root = discover_project_root(args.config)
    output_root = resolve_path(project_root, config["output"]["root_dir"])
    if output_root is None:
        raise SystemExit("output.root_dir must be set")
    output_root = ensure_descendant(output_root, project_root)
    paths = output_paths(output_root, config["output"]["prefix"])
    paths["graphs_dir"].mkdir(parents=True, exist_ok=True)
    for stale_graph in paths["graphs_dir"].glob("*.pt"):
        stale_graph.unlink()

    scrna_manifest_path = ensure_descendant(resolve_path(project_root, config["input"]["scrna_manifest_path"]), project_root)
    prior_manifest_path = ensure_descendant(resolve_path(project_root, config["input"]["prior_manifest_path"]), project_root)
    spatial_manifest_path = ensure_descendant(resolve_path(project_root, config["input"]["spatial_manifest_path"]), project_root)

    _, reference_stats = load_reference_artifacts(scrna_manifest_path, project_root)
    gene_master, gene_gene_edges, gene_pathway_edges, pathway_catalog = load_prior_artifacts(prior_manifest_path, project_root)
    spatial_nodes, spatial_edges, spatial_input_path = load_spatial_artifacts(spatial_manifest_path, project_root)

    spatial_expression_path = resolve_path(project_root, config["input"].get("spatial_expression_path")) or spatial_input_path
    spatial_expression_path = ensure_descendant(spatial_expression_path, project_root)
    spatial_bundle, spatial_gene_frame = build_spatial_expression_lookup(
        expression_path=spatial_expression_path,
        gene_symbol_column=config["input"].get("spatial_expression_gene_symbol_column"),
        assume_log1p_input=bool(config["input"].get("assume_log1p_input", True)),
    )

    expression = spatial_bundle["expression"]
    obs_frame = spatial_bundle["adata"].obs.copy()
    obs_frame.index = obs_frame.index.astype(str)
    node_id_column = config["input"].get("spatial_node_id_column")
    if node_id_column and node_id_column in obs_frame.columns:
        obs_frame["__node_id__"] = obs_frame[node_id_column].astype(str)
    else:
        obs_frame["__node_id__"] = obs_frame.index.astype(str)
    obs_lookup = obs_frame.reset_index(drop=True).set_index("__node_id__", drop=False)
    gene_index = build_gene_index(spatial_gene_frame)
    graph_meta_frame = summarize_graph_metadata(
        spatial_nodes=spatial_nodes,
        obs_lookup=obs_lookup,
        config=config,
    )

    reference_lookup = build_reference_lookup(
        reference_stats=reference_stats,
        gene_master=gene_master,
        gene_gene_edges=gene_gene_edges,
        gene_pathway_edges=gene_pathway_edges,
        config=config,
    )
    canonical_genes = [
        gene
        for gene in reference_lookup["canonical_gene_symbol_norm"].astype(str).tolist()
        if gene in gene_index
    ]
    if not canonical_genes:
        raise SystemExit("No overlap between the spatial expression matrix and the canonical gene universe")

    graph_index_rows: list[dict[str, Any]] = []
    bag_rows: list[dict[str, Any]] = []
    feature_schema: dict[str, Any] | None = None

    runtime_max_graphs = config["runtime"].get("max_graphs")
    graph_ids = graph_meta_frame.loc[~graph_meta_frame["is_uncertain"]].index.tolist()
    if runtime_max_graphs is not None:
        graph_ids = graph_ids[: int(runtime_max_graphs)]
    bag_pairs = spatial_nodes.loc[spatial_nodes["graph_id"].isin(graph_ids), ["patient_id", "sample_id"]].astype(str)
    bag_ids = (
        bag_pairs.apply("|".join, axis=1).drop_duplicates().tolist()
        if not bag_pairs.empty
        else []
    )
    validate_export_counts(
        n_graphs=len(graph_ids),
        n_bags=len(bag_ids),
        validation_cfg=config.get("validation", {}),
        dataset_id=config["input"]["dataset_id"],
    )

    for graph_id in graph_ids:
        graph_nodes = (
            spatial_nodes.loc[spatial_nodes["graph_id"] == graph_id]
            .sort_values("local_node_index")
            .reset_index(drop=True)
            .copy()
        )
        if graph_nodes.empty:
            continue
        try:
            graph_obs = obs_lookup.loc[graph_nodes["node_id"].astype(str).tolist()].copy()
        except KeyError as exc:
            missing = sorted(set(graph_nodes["node_id"].astype(str)) - set(obs_lookup.index.astype(str)))
            raise SystemExit(f"Spatial expression matrix is missing nodes for graph {graph_id}: {missing[:10]}") from exc
        graph_expression = expression[[obs_lookup.index.get_loc(node_id) for node_id in graph_nodes["node_id"].astype(str)], :]
        cell_expression = aggregate_gene_expression(graph_expression, gene_index, canonical_genes)
        # Skip graphs with insufficient expression (e.g. Xenium tiles where
        # most cells express none of the canonical genes).
        cells_with_expr = int((cell_expression.sum(axis=1) > 0).sum())
        if cells_with_expr < 5:
            print(
                f"  Skipping {graph_id}: only {cells_with_expr}/{cell_expression.shape[0]} "
                "cells have non-zero expression for canonical genes"
            )
            continue
        graph_edges = spatial_edges.loc[spatial_edges["graph_id"] == graph_id].reset_index(drop=True).copy()
        graph_meta = graph_meta_frame.loc[graph_id]
        data, graph_payload = build_graph_object(
            graph_nodes=graph_nodes,
            graph_obs=graph_obs,
            graph_meta=graph_meta,
            cell_expression=cell_expression,
            gene_order=canonical_genes,
            graph_edges=graph_edges,
            reference_lookup=reference_lookup,
            gene_gene_edges=gene_gene_edges,
            gene_pathway_edges=gene_pathway_edges,
            pathway_catalog=pathway_catalog,
            config=config,
        )
        torch.save(data, paths["graphs_dir"] / f"{graph_id}.pt")
        graph_index_rows.append(graph_payload["summary"])
        bag_rows.append(
            {
                "graph_id": data.graph_id,
                "bag_id": data.bag_id,
                "patient_id": data.patient_id,
                "sample_id": data.sample_id,
                "slide_id": data.slide_id,
                "split_id": data.split_id,
                "cohort_id": data.cohort_id,
            }
        )
        if feature_schema is None:
            feature_schema = graph_payload["feature_schema"]

    graph_index_frame = pd.DataFrame.from_records(graph_index_rows)
    if bag_rows:
        bag_frame = pd.DataFrame.from_records(bag_rows).drop_duplicates().sort_values(["bag_id", "graph_id"])
    else:
        bag_frame = pd.DataFrame(columns=["bag_id", "graph_id", "patient_id", "sample_id"])
    write_table(graph_index_frame, paths["graph_index"])
    write_table(bag_frame, paths["sample_bags"])
    write_json(paths["feature_schema"], feature_schema or {})

    manifest = {
        "builder": "build_hetero_graphs.py",
        "config_path": config["_config_path"],
        "dataset_id": config["input"]["dataset_id"],
        "input_manifests": {
            "scrna": str(scrna_manifest_path),
            "prior": str(prior_manifest_path),
            "spatial": str(spatial_manifest_path),
        },
        "input_expression_path": str(spatial_expression_path),
        "graph_label_mode": str(config["labels"].get("graph_label_mode", "constant_obs")),
        "n_graphs": int(len(graph_index_frame)),
        "n_bags": int(bag_frame["bag_id"].nunique()) if not bag_frame.empty else 0,
        "n_candidate_graphs": int(graph_meta_frame.shape[0]),
        "n_uncertain_graphs_dropped": int(graph_meta_frame["is_uncertain"].sum()),
        "label_counts": graph_index_frame["time_label"].value_counts().to_dict() if not graph_index_frame.empty else {},
        "schema": config["schema"],
        "feature_dims": {
            "cell": 50,
            "gene": 10,
            "pathway": 11,
        },
        "outputs": {
            "graphs_dir": str(paths["graphs_dir"]),
            "graph_index": str(paths["graph_index"]),
            "sample_bags": str(paths["sample_bags"]),
            "feature_schema": str(paths["feature_schema"]),
            "manifest": str(paths["manifest"]),
        },
    }
    write_json(paths["manifest"], manifest)


if __name__ == "__main__":
    main()
