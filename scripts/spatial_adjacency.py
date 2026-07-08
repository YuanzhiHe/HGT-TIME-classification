#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors

try:
    from scipy.spatial import Delaunay, QhullError
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    Delaunay = None
    QhullError = RuntimeError


def build_default_config() -> dict[str, Any]:
    return {
        "dataset": {
            "dataset_id": "visium_breast",
            "input_path": "datasets/spatial/processed/input.h5ad",
            "input_format": "auto",
            "node_id_column": None,
            "patient_id_column": "patient_id",
            "sample_id_column": "sample_id",
            "cohort_id_column": "cohort_id",
            "slide_id_column": "slide_id",
            "split_id_column": "split_id",
            "label_source_column": "label_source",
            "is_uncertain_column": "is_uncertain",
            "in_tissue_column": "in_tissue",
            "compartment_column": "compartment",
            "coord_x_column": "array_col",
            "coord_y_column": "array_row",
            "spatial_obsm_key": "spatial",
            "graph_group_fields": ["patient_id", "sample_id", "slide_id"],
            "graph_partition": {
                "mode": "none",
                "row_bins": 1,
                "col_bins": 1,
            },
        },
        "filters": {
            "keep_in_tissue_only": True,
            "drop_uncertain": False,
            "min_nodes_per_graph": 8,
        },
        "graph": {
            "primary_strategy": "scaled_radius",
            "distance_metric": "euclidean",
            "symmetrize_mode": "union",
            "radius": {
                "anchor_k": 6,
                "scale_factor": 1.15,
                "min_radius": None,
                "max_radius": None,
                "min_degree": 2,
                "max_neighbors_per_source": 8,
            },
            "knn": {
                "k": 6,
            },
            "delaunay": {
                "max_edge_length_scale": 1.6,
                "min_degree": 2,
            },
            "weights": {
                "rule": "gaussian",
                "sigma_scale": 1.0,
                "min_weight": 1.0e-6,
            },
            "permutation": {
                "enabled": True,
                "strategy": "shuffle_coordinates",
                "random_seed": 17,
            },
            "variants_to_export": ["scaled_radius", "knn", "delaunay", "permuted_primary"],
        },
        "output": {
            "root_dir": "outputs/spatial/visium_breast__spatial_v1",
            "prefix": "visium_breast__spatial_v1",
        },
        "validation": {
            "min_graphs": None,
            "fail_below_min_graphs": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reproducible spatial adjacency edge lists")
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
        if (candidate / "instance.json").exists():
            return candidate
    raise SystemExit("Could not locate project root via instance.json")


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


def require_anndata() -> Any:
    try:
        import anndata as ad  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
        raise SystemExit(
            "anndata is required for .h5ad input. Install requirements-spatial.txt"
        ) from exc
    return ad


def load_tabular_input(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if name.endswith(".tsv") or name.endswith(".tsv.gz") or name.endswith(".txt") or name.endswith(".txt.gz"):
        return pd.read_csv(path, sep="\t")
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported tabular input format: {path}")


def load_input(dataset_cfg: dict[str, Any], base_dir: Path) -> tuple[pd.DataFrame, str]:
    input_path = resolve_path(base_dir, dataset_cfg["input_path"])
    if input_path is None or not input_path.exists():
        raise SystemExit(f"Missing dataset input: {dataset_cfg['input_path']}")

    input_format = dataset_cfg.get("input_format", "auto").lower()
    if input_format == "auto":
        if input_path.suffix == ".h5ad":
            input_format = "h5ad"
        else:
            input_format = "table"

    if input_format == "h5ad":
        ad = require_anndata()
        adata = ad.read_h5ad(input_path)
        frame = adata.obs.copy()
        frame.index = frame.index.astype(str)
        frame["__obs_index__"] = frame.index
        x_column = dataset_cfg.get("coord_x_column")
        y_column = dataset_cfg.get("coord_y_column")
        if x_column in frame.columns and y_column in frame.columns:
            return frame.reset_index(drop=True), str(input_path)
        spatial_key = dataset_cfg.get("spatial_obsm_key", "spatial")
        if spatial_key not in adata.obsm:
            raise SystemExit(
                f"Could not find coordinate columns '{x_column}', '{y_column}' "
                f"or adata.obsm['{spatial_key}'] in {input_path}"
            )
        coords = np.asarray(adata.obsm[spatial_key])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise SystemExit(f"adata.obsm['{spatial_key}'] must have shape [n, >=2]")
        frame[x_column] = coords[:, 0]
        frame[y_column] = coords[:, 1]
        return frame.reset_index(drop=True), str(input_path)

    return load_tabular_input(input_path), str(input_path)


def coerce_bool_series(series: pd.Series, default: bool) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype=bool)
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


def get_column(frame: pd.DataFrame, column_name: str | None, default: str) -> pd.Series:
    if column_name and column_name in frame.columns:
        return frame[column_name]
    return pd.Series([default] * len(frame), index=frame.index)


def standardize_nodes(frame: pd.DataFrame, dataset_cfg: dict[str, Any], filter_cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    node_id_column = dataset_cfg.get("node_id_column")
    if node_id_column and node_id_column in frame.columns:
        node_id = frame[node_id_column].astype(str)
    elif "__obs_index__" in frame.columns:
        node_id = frame["__obs_index__"].astype(str)
    else:
        node_id = pd.Series([f"node_{index:07d}" for index in range(len(frame))], index=frame.index)

    coord_x_column = dataset_cfg["coord_x_column"]
    coord_y_column = dataset_cfg["coord_y_column"]
    if coord_x_column not in frame.columns or coord_y_column not in frame.columns:
        raise SystemExit(
            f"Input table must contain coordinate columns '{coord_x_column}' and '{coord_y_column}'"
        )

    nodes = pd.DataFrame(
        {
            "node_id": node_id.astype(str),
            "patient_id": get_column(frame, dataset_cfg.get("patient_id_column"), "unknown").astype(str),
            "sample_id": get_column(frame, dataset_cfg.get("sample_id_column"), "unknown").astype(str),
            "cohort_id": get_column(frame, dataset_cfg.get("cohort_id_column"), "unknown").astype(str),
            "slide_id": get_column(frame, dataset_cfg.get("slide_id_column"), "unknown").astype(str),
            "split_id": get_column(frame, dataset_cfg.get("split_id_column"), "unknown").astype(str),
            "label_source": get_column(frame, dataset_cfg.get("label_source_column"), "unknown").astype(str),
            "compartment": get_column(frame, dataset_cfg.get("compartment_column"), "unknown").astype(str),
            "coord_x": pd.to_numeric(frame[coord_x_column], errors="coerce"),
            "coord_y": pd.to_numeric(frame[coord_y_column], errors="coerce"),
        }
    )
    is_uncertain = get_column(frame, dataset_cfg.get("is_uncertain_column"), "false")
    in_tissue = get_column(frame, dataset_cfg.get("in_tissue_column"), "true")
    nodes["is_uncertain"] = coerce_bool_series(is_uncertain, default=False)
    nodes["in_tissue"] = coerce_bool_series(in_tissue, default=True)

    counts = {
        "n_input_nodes": int(len(nodes)),
    }

    missing_coords = nodes["coord_x"].isna() | nodes["coord_y"].isna()
    counts["dropped_missing_coordinates"] = int(missing_coords.sum())
    nodes = nodes.loc[~missing_coords].copy()

    if filter_cfg.get("keep_in_tissue_only", True):
        mask = nodes["in_tissue"]
        counts["dropped_out_of_tissue"] = int((~mask).sum())
        nodes = nodes.loc[mask].copy()
    else:
        counts["dropped_out_of_tissue"] = 0

    if filter_cfg.get("drop_uncertain", False):
        mask = ~nodes["is_uncertain"]
        counts["dropped_uncertain"] = int((~mask).sum())
        nodes = nodes.loc[mask].copy()
    else:
        counts["dropped_uncertain"] = 0

    graph_fields = dataset_cfg.get("graph_group_fields", ["patient_id", "sample_id", "slide_id"])
    missing_fields = [field for field in graph_fields if field not in nodes.columns]
    if missing_fields:
        raise SystemExit(f"graph_group_fields reference missing standardized columns: {missing_fields}")
    nodes["graph_id"] = (
        nodes[graph_fields]
        .astype(str)
        .agg(lambda row: "__".join(value for value in row if value and value != "unknown"), axis=1)
        .replace("", "graph_unknown")
    )
    nodes["base_graph_id"] = nodes["graph_id"]

    partition_cfg = dataset_cfg.get("graph_partition", {}) or {}
    partition_mode = str(partition_cfg.get("mode", "none")).lower()
    if partition_mode not in {"none", "grid"}:
        raise SystemExit(f"Unsupported graph_partition.mode: {partition_mode}")
    if partition_mode == "grid":
        row_bins = max(int(partition_cfg.get("row_bins", 1)), 1)
        col_bins = max(int(partition_cfg.get("col_bins", 1)), 1)
        tile_frames: list[pd.DataFrame] = []
        for base_graph_id, graph_nodes in nodes.groupby("base_graph_id", sort=True):
            graph_nodes = graph_nodes.copy()
            row_bin = pd.cut(graph_nodes["coord_y"], bins=row_bins, labels=False, include_lowest=True)
            col_bin = pd.cut(graph_nodes["coord_x"], bins=col_bins, labels=False, include_lowest=True)
            graph_nodes["tile_row_bin"] = row_bin.astype(int)
            graph_nodes["tile_col_bin"] = col_bin.astype(int)
            graph_nodes["tile_id"] = graph_nodes.apply(
                lambda row: f"tile_r{int(row['tile_row_bin']):02d}_c{int(row['tile_col_bin']):02d}",
                axis=1,
            )
            graph_nodes["graph_id"] = graph_nodes["base_graph_id"] + "__" + graph_nodes["tile_id"]
            tile_frames.append(graph_nodes)
        nodes = pd.concat(tile_frames, ignore_index=True) if tile_frames else nodes
    else:
        nodes["tile_row_bin"] = -1
        nodes["tile_col_bin"] = -1
        nodes["tile_id"] = "full_graph"

    duplicate_mask = nodes.duplicated(subset=["graph_id", "node_id"], keep=False)
    if duplicate_mask.any():
        duplicated = nodes.loc[duplicate_mask, ["graph_id", "node_id"]].head(10).to_dict(orient="records")
        raise SystemExit(f"Duplicate node_id within graph_id detected: {duplicated}")

    graph_sizes = nodes.groupby("graph_id").size()
    min_nodes = int(filter_cfg.get("min_nodes_per_graph", 1))
    keep_graphs = graph_sizes[graph_sizes >= min_nodes].index
    counts["dropped_small_graph_nodes"] = int((~nodes["graph_id"].isin(keep_graphs)).sum())
    nodes = nodes.loc[nodes["graph_id"].isin(keep_graphs)].copy()

    nodes = nodes.sort_values(["graph_id", "node_id"]).reset_index(drop=True)
    nodes["global_node_index"] = np.arange(len(nodes), dtype=int)
    nodes["local_node_index"] = nodes.groupby("graph_id").cumcount().astype(int)
    counts["n_output_nodes"] = int(len(nodes))
    counts["n_output_graphs"] = int(nodes["graph_id"].nunique())
    return nodes, counts


def make_neighbors(coords: np.ndarray, max_neighbors: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if len(coords) <= 1:
        return np.zeros((len(coords), 1), dtype=float), np.zeros((len(coords), 1), dtype=int)
    n_neighbors = min(len(coords), max_neighbors + 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    return distances, indices


def estimate_scale_unit(distances: np.ndarray) -> float:
    if distances.shape[1] <= 1:
        return 1.0
    anchor = distances[:, -1]
    positive = anchor[anchor > 0]
    if positive.size:
        return float(np.median(positive))
    fallback = distances[:, 1:].reshape(-1)
    fallback = fallback[fallback > 0]
    if fallback.size:
        return float(np.median(fallback))
    return 1.0


def gaussian_weight(distance: float, scale_unit: float, weight_cfg: dict[str, Any]) -> float:
    sigma = max(scale_unit * float(weight_cfg.get("sigma_scale", 1.0)), 1.0e-8)
    rule = str(weight_cfg.get("rule", "gaussian")).lower()
    if rule == "gaussian":
        weight = math.exp(-0.5 * (distance / sigma) ** 2)
    elif rule == "inverse_distance":
        weight = 1.0 / (1.0 + (distance / sigma))
    else:
        raise SystemExit(f"Unsupported weight rule: {rule}")
    return max(float(weight_cfg.get("min_weight", 0.0)), float(weight))


def insert_edge(edge_map: dict[tuple[int, int], float], source: int, target: int, distance: float) -> None:
    if source == target:
        return
    pair = (source, target) if source < target else (target, source)
    current = edge_map.get(pair)
    if current is None or distance < current:
        edge_map[pair] = float(distance)


def degree_from_edges(edge_map: dict[tuple[int, int], float], n_nodes: int) -> np.ndarray:
    degree = np.zeros(n_nodes, dtype=int)
    for source, target in edge_map:
        degree[source] += 1
        degree[target] += 1
    return degree


def supplement_min_degree(
    edge_map: dict[tuple[int, int], float],
    indices: np.ndarray,
    distances: np.ndarray,
    min_degree: int,
) -> None:
    if min_degree <= 0:
        return
    n_nodes = len(indices)
    degree = degree_from_edges(edge_map, n_nodes)
    for source in range(n_nodes):
        if degree[source] >= min_degree:
            continue
        for neighbor_pos in range(1, indices.shape[1]):
            target = int(indices[source, neighbor_pos])
            if source == target:
                continue
            insert_edge(edge_map, source, target, float(distances[source, neighbor_pos]))
            degree = degree_from_edges(edge_map, n_nodes)
            if degree[source] >= min_degree:
                break


def radius_neighbors(
    coords: np.ndarray,
    radius: float,
    metric: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(coords) <= 1:
        return [np.array([], dtype=float) for _ in range(len(coords))], [np.array([], dtype=int) for _ in range(len(coords))]
    nbrs = NearestNeighbors(radius=radius, metric=metric)
    nbrs.fit(coords)
    distances, indices = nbrs.radius_neighbors(coords, sort_results=True)
    return distances, indices


def build_scaled_radius_edges(
    coords: np.ndarray,
    radius_cfg: dict[str, Any],
    metric: str,
    indices: np.ndarray,
    distances: np.ndarray,
    scale_unit: float,
) -> tuple[dict[tuple[int, int], float], dict[str, float]]:
    radius = scale_unit * float(radius_cfg.get("scale_factor", 1.0))
    min_radius = radius_cfg.get("min_radius")
    max_radius = radius_cfg.get("max_radius")
    if min_radius is not None:
        radius = max(radius, float(min_radius))
    if max_radius is not None:
        radius = min(radius, float(max_radius))

    edge_map: dict[tuple[int, int], float] = {}
    radius_distances, radius_indices = radius_neighbors(coords, radius, metric)
    max_neighbors = radius_cfg.get("max_neighbors_per_source")
    for source, (distance_row, index_row) in enumerate(zip(radius_distances, radius_indices)):
        pairs = [
            (int(target), float(distance))
            for distance, target in zip(distance_row, index_row)
            if int(target) != source
        ]
        pairs.sort(key=lambda pair: pair[1])
        if max_neighbors is not None:
            pairs = pairs[: int(max_neighbors)]
        for target, distance in pairs:
            insert_edge(edge_map, source, target, distance)

    supplement_min_degree(
        edge_map,
        indices,
        distances,
        int(radius_cfg.get("min_degree", 0)),
    )
    return edge_map, {"radius": float(radius)}


def build_knn_edges(indices: np.ndarray, distances: np.ndarray, k: int) -> dict[tuple[int, int], float]:
    edge_map: dict[tuple[int, int], float] = {}
    limit = min(indices.shape[1] - 1, k)
    for source in range(len(indices)):
        for neighbor_pos in range(1, limit + 1):
            target = int(indices[source, neighbor_pos])
            if source == target:
                continue
            insert_edge(edge_map, source, target, float(distances[source, neighbor_pos]))
    return edge_map


def build_delaunay_edges(
    coords: np.ndarray,
    scale_unit: float,
    delaunay_cfg: dict[str, Any],
    indices: np.ndarray,
    distances: np.ndarray,
) -> tuple[dict[tuple[int, int], float], dict[str, float]]:
    if Delaunay is None:
        raise SystemExit("scipy is required for Delaunay adjacency. Install requirements-spatial.txt")

    edge_map: dict[tuple[int, int], float] = {}
    if len(coords) < 3:
        edge_map = build_knn_edges(indices, distances, k=min(2, max(1, len(coords) - 1)))
        return edge_map, {"fallback_to_knn": 1.0}

    try:
        triangulation = Delaunay(coords)
    except (QhullError, ValueError):
        edge_map = build_knn_edges(indices, distances, k=min(2, max(1, len(coords) - 1)))
        return edge_map, {"fallback_to_knn": 1.0}

    max_scale = delaunay_cfg.get("max_edge_length_scale")
    length_cap = None if max_scale is None else float(max_scale) * scale_unit
    for simplex in triangulation.simplices:
        for source, target in combinations(simplex.tolist(), 2):
            distance = float(np.linalg.norm(coords[source] - coords[target]))
            if length_cap is not None and distance > length_cap:
                continue
            insert_edge(edge_map, int(source), int(target), distance)

    supplement_min_degree(
        edge_map,
        indices,
        distances,
        int(delaunay_cfg.get("min_degree", 0)),
    )
    return edge_map, {"length_cap": float(length_cap) if length_cap is not None else -1.0}


def build_permuted_primary_edges(
    coords: np.ndarray,
    graph_cfg: dict[str, Any],
    scale_unit: float,
    graph_seed: int,
) -> tuple[dict[tuple[int, int], float], dict[str, float]]:
    permutation_cfg = graph_cfg.get("permutation", {})
    if not permutation_cfg.get("enabled", True):
        return {}, {"permutation_seed": float(graph_seed)}
    if permutation_cfg.get("strategy", "shuffle_coordinates") != "shuffle_coordinates":
        raise SystemExit("Only permutation.strategy=shuffle_coordinates is supported")

    rng = np.random.default_rng(graph_seed)
    permuted_coords = coords[rng.permutation(len(coords))]
    primary = str(graph_cfg.get("primary_strategy", "scaled_radius")).lower()
    if primary != "scaled_radius":
        raise SystemExit("permuted_primary currently supports primary_strategy=scaled_radius only")

    anchor_k = max(
        int(graph_cfg["radius"].get("anchor_k", 1)),
        int(graph_cfg["knn"].get("k", 1)),
        int(graph_cfg["radius"].get("min_degree", 0)),
        int(graph_cfg["delaunay"].get("min_degree", 0)),
    )
    permuted_distances, permuted_indices = make_neighbors(
        permuted_coords,
        max_neighbors=anchor_k,
        metric=graph_cfg.get("distance_metric", "euclidean"),
    )

    primary_edges, radius_meta = build_scaled_radius_edges(
        permuted_coords,
        graph_cfg["radius"],
        graph_cfg.get("distance_metric", "euclidean"),
        permuted_indices,
        permuted_distances,
        scale_unit,
    )
    radius_meta["permutation_seed"] = float(graph_seed)
    return primary_edges, radius_meta


def edge_frame_from_map(
    nodes: pd.DataFrame,
    edge_map: dict[tuple[int, int], float],
    variant: str,
    scale_unit: float,
    derived: dict[str, float],
    weight_cfg: dict[str, Any],
) -> pd.DataFrame:
    if not edge_map:
        return pd.DataFrame(
            columns=[
                "graph_id",
                "patient_id",
                "sample_id",
                "cohort_id",
                "slide_id",
                "split_id",
                "variant",
                "source_local_index",
                "target_local_index",
                "source_node_id",
                "target_node_id",
                "source_compartment",
                "target_compartment",
                "same_compartment",
                "distance_euclidean",
                "distance_normalized",
                "edge_weight",
                "scale_unit",
                "radius_used",
                "length_cap",
                "permutation_seed",
            ]
        )

    records: list[dict[str, Any]] = []
    radius_used = derived.get("radius")
    length_cap = derived.get("length_cap")
    permutation_seed = derived.get("permutation_seed")
    for source, target in sorted(edge_map):
        source_row = nodes.iloc[source]
        target_row = nodes.iloc[target]
        distance = float(edge_map[(source, target)])
        normalized = distance / scale_unit if scale_unit > 0 else distance
        records.append(
            {
                "graph_id": source_row["graph_id"],
                "patient_id": source_row["patient_id"],
                "sample_id": source_row["sample_id"],
                "cohort_id": source_row["cohort_id"],
                "slide_id": source_row["slide_id"],
                "split_id": source_row["split_id"],
                "variant": variant,
                "source_local_index": int(source_row["local_node_index"]),
                "target_local_index": int(target_row["local_node_index"]),
                "source_node_id": source_row["node_id"],
                "target_node_id": target_row["node_id"],
                "source_compartment": source_row["compartment"],
                "target_compartment": target_row["compartment"],
                "same_compartment": bool(source_row["compartment"] == target_row["compartment"]),
                "distance_euclidean": distance,
                "distance_normalized": normalized,
                "edge_weight": gaussian_weight(distance, scale_unit, weight_cfg),
                "scale_unit": scale_unit,
                "radius_used": radius_used,
                "length_cap": length_cap,
                "permutation_seed": permutation_seed,
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_variant(
    nodes: pd.DataFrame,
    variant: str,
    edge_map: dict[tuple[int, int], float],
    scale_unit: float,
    derived: dict[str, float],
) -> dict[str, Any]:
    n_nodes = len(nodes)
    n_edges = len(edge_map)
    degree = degree_from_edges(edge_map, n_nodes)
    density = 0.0 if n_nodes <= 1 else float((2.0 * n_edges) / (n_nodes * (n_nodes - 1)))
    return {
        "graph_id": nodes.iloc[0]["graph_id"],
        "patient_id": nodes.iloc[0]["patient_id"],
        "sample_id": nodes.iloc[0]["sample_id"],
        "cohort_id": nodes.iloc[0]["cohort_id"],
        "slide_id": nodes.iloc[0]["slide_id"],
        "split_id": nodes.iloc[0]["split_id"],
        "variant": variant,
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "density": density,
        "mean_degree": float(degree.mean()) if degree.size else 0.0,
        "max_degree": int(degree.max()) if degree.size else 0,
        "isolated_nodes": int((degree == 0).sum()),
        "scale_unit": float(scale_unit),
        "radius_used": derived.get("radius"),
        "length_cap": derived.get("length_cap"),
        "permutation_seed": derived.get("permutation_seed"),
    }


def build_edges(nodes: pd.DataFrame, graph_cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    metric = graph_cfg.get("distance_metric", "euclidean")
    if str(graph_cfg.get("symmetrize_mode", "union")).lower() != "union":
        raise SystemExit("Only graph.symmetrize_mode=union is supported")
    radius_cfg = graph_cfg["radius"]
    knn_cfg = graph_cfg["knn"]
    delaunay_cfg = graph_cfg["delaunay"]
    permutation_cfg = graph_cfg.get("permutation", {})
    variants = list(dict.fromkeys(graph_cfg.get("variants_to_export", [])))
    primary_variant = str(graph_cfg.get("primary_strategy", "scaled_radius")).lower()
    if primary_variant not in variants:
        variants.insert(0, primary_variant)

    edge_frames_by_variant: dict[str, list[pd.DataFrame]] = {variant: [] for variant in variants}
    primary_alias_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    nodes_with_scale: list[pd.DataFrame] = []

    anchor_k = max(
        int(radius_cfg.get("anchor_k", 1)),
        int(knn_cfg.get("k", 1)),
        int(radius_cfg.get("min_degree", 0)),
        int(delaunay_cfg.get("min_degree", 0)),
    )
    global_seed = int(permutation_cfg.get("random_seed", 17))

    for graph_index, (_, graph_nodes) in enumerate(nodes.groupby("graph_id", sort=True)):
        graph_nodes = graph_nodes.copy().reset_index(drop=True)
        coords = graph_nodes[["coord_x", "coord_y"]].to_numpy(dtype=float)
        distances, indices = make_neighbors(coords, max_neighbors=anchor_k, metric=metric)
        scale_unit = estimate_scale_unit(distances)
        graph_nodes["coord_x_centered"] = graph_nodes["coord_x"] - float(graph_nodes["coord_x"].mean())
        graph_nodes["coord_y_centered"] = graph_nodes["coord_y"] - float(graph_nodes["coord_y"].mean())
        graph_nodes["coord_x_scaled"] = graph_nodes["coord_x_centered"] / scale_unit
        graph_nodes["coord_y_scaled"] = graph_nodes["coord_y_centered"] / scale_unit
        graph_nodes["scale_unit"] = scale_unit
        nodes_with_scale.append(graph_nodes)

        built_variants: dict[str, tuple[dict[tuple[int, int], float], dict[str, float]]] = {}
        built_variants["scaled_radius"] = build_scaled_radius_edges(
            coords,
            radius_cfg,
            metric,
            indices,
            distances,
            scale_unit,
        )
        built_variants["knn"] = (
            build_knn_edges(indices, distances, int(knn_cfg.get("k", 1))),
            {},
        )
        built_variants["delaunay"] = build_delaunay_edges(
            coords,
            scale_unit,
            delaunay_cfg,
            indices,
            distances,
        )
        if permutation_cfg.get("enabled", True):
            built_variants["permuted_primary"] = build_permuted_primary_edges(
                coords,
                graph_cfg,
                scale_unit,
                graph_seed=global_seed + graph_index,
            )

        if primary_variant not in built_variants:
            raise SystemExit(f"primary_strategy '{primary_variant}' is not a supported exported variant")

        for variant in variants:
            edge_map, derived = built_variants[variant]
            edge_frame = edge_frame_from_map(
                graph_nodes,
                edge_map,
                variant,
                scale_unit,
                derived,
                graph_cfg["weights"],
            )
            edge_frames_by_variant.setdefault(variant, []).append(edge_frame)
            summary_records.append(summarize_variant(graph_nodes, variant, edge_map, scale_unit, derived))
            if variant == primary_variant:
                primary_alias = edge_frame.copy()
                primary_alias["variant"] = "primary"
                primary_alias_frames.append(primary_alias)

    node_output = pd.concat(nodes_with_scale, ignore_index=True) if nodes_with_scale else nodes.copy()
    summary_output = pd.DataFrame.from_records(summary_records)
    edge_outputs = {
        variant: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for variant, frames in edge_frames_by_variant.items()
    }
    edge_outputs["primary"] = pd.concat(primary_alias_frames, ignore_index=True) if primary_alias_frames else pd.DataFrame()
    return edge_outputs, summary_output, node_output


def output_paths(root_dir: Path, prefix: str, variants: list[str]) -> dict[str, Path]:
    paths = {
        "nodes": root_dir / f"{prefix}__step-01_spatial_nodes.tsv.gz",
        "graph_summary": root_dir / f"{prefix}__step-02_graph_summary.tsv.gz",
        "manifest": root_dir / f"{prefix}__step-03_manifest.json",
    }
    for variant in variants:
        paths[f"edges_{variant}"] = root_dir / f"{prefix}__step-03_edges__{variant}.tsv.gz"
    return paths


def validate_graph_count(node_output: pd.DataFrame, validation_cfg: dict[str, Any], dataset_id: str) -> None:
    min_graphs = validation_cfg.get("min_graphs")
    if min_graphs is None:
        return
    n_graphs = int(node_output["graph_id"].nunique()) if not node_output.empty else 0
    if n_graphs >= int(min_graphs):
        return
    message = (
        f"{dataset_id}: spatial export produced only {n_graphs} graph(s), below the configured "
        f"minimum of {int(min_graphs)}. This dataset cannot support the frozen grouped CV protocol "
        f"without more slides/samples or a revised evaluation contract."
    )
    if bool(validation_cfg.get("fail_below_min_graphs", False)):
        raise SystemExit(message)
    print(f"[WARN] {message}")


def main() -> None:
    args = parse_args()
    if args.write_default_config:
        write_yaml(args.write_default_config, build_default_config())
        return
    if not args.config:
        raise SystemExit("Either --config or --write-default-config must be provided")

    config_path = args.config.resolve()
    config = load_config(config_path)
    project_root = discover_project_root(config_path)
    output_root = resolve_path(project_root, config["output"]["root_dir"])
    if output_root is None:
        raise SystemExit("output.root_dir must be set")
    variants = list(dict.fromkeys(["primary", *config["graph"].get("variants_to_export", [])]))
    paths = output_paths(output_root, config["output"]["prefix"], variants)
    output_root.mkdir(parents=True, exist_ok=True)

    input_frame, input_resolved_path = load_input(config["dataset"], project_root)
    nodes, filter_counts = standardize_nodes(input_frame, config["dataset"], config["filters"])
    edge_outputs, graph_summary, node_output = build_edges(nodes, config["graph"])
    validate_graph_count(node_output, config.get("validation", {}), config["dataset"]["dataset_id"])

    write_table(node_output, paths["nodes"])
    write_table(graph_summary, paths["graph_summary"])
    for variant in variants:
        write_table(edge_outputs.get(variant, pd.DataFrame()), paths[f"edges_{variant}"])

    manifest = {
        "config_path": config["_config_path"],
        "input_path": input_resolved_path,
        "dataset_id": config["dataset"]["dataset_id"],
        "primary_strategy": config["graph"]["primary_strategy"],
        "variants_exported": variants,
        "filter_counts": filter_counts,
        "n_graphs": int(node_output["graph_id"].nunique()) if not node_output.empty else 0,
        "n_nodes": int(len(node_output)),
        "edge_counts": {
            variant: int(len(edge_outputs.get(variant, pd.DataFrame())))
            for variant in variants
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "parameter_snapshot": {
            "dataset": config["dataset"],
            "graph": config["graph"],
            "filters": config["filters"],
        },
    }
    write_json(paths["manifest"], manifest)


if __name__ == "__main__":
    main()
