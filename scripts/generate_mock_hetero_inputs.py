#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import anndata as ad  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    ad = None


GENES = [
    "CD3D",
    "NKG7",
    "IFNG",
    "CXCL10",
    "PDCD1",
    "MS4A1",
    "CD79A",
    "LYZ",
    "FCER1G",
    "COL1A1",
    "TAGLN",
    "PECAM1",
    "VWF",
    "EPCAM",
    "KRT8",
    "KRT19",
    "ERBB2",
    "STAT1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a minimal mock dataset for the hetero graph builder")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def require_h5ad() -> Any:
    if ad is None:
        raise SystemExit(
            "anndata is required for the mock generator. "
            "Install dependencies from requirements-hetero.txt"
        )
    return ad


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def build_spatial_obs(graph_specs: list[dict[str, str]]) -> pd.DataFrame:
    rows = []
    for graph_index, spec in enumerate(graph_specs):
        for local_index in range(8):
            compartment = "tumor" if local_index < 3 else ("boundary" if local_index < 5 else "stroma")
            rows.append(
                {
                    "node_id": f"{spec['graph_id']}_spot_{local_index:02d}",
                    "patient_id": spec["patient_id"],
                    "sample_id": spec["sample_id"],
                    "slide_id": spec["slide_id"],
                    "cohort_id": spec["cohort_id"],
                    "split_id": spec["split_id"],
                    "label_source": "mock_rule",
                    "is_uncertain": False,
                    "in_tissue": True,
                    "compartment": compartment,
                    "time_label": spec["time_label"],
                    "infiltration_score": spec["infiltration_score"],
                    "penetration_score": spec["penetration_score"],
                    "retention_score": spec["retention_score"],
                    "activation_score": spec["activation_score"],
                    "cell_type": "Epithelial_Malignant" if compartment == "tumor" else ("T_NK" if local_index % 2 == 0 else "CAF"),
                    "tumor_probability": 0.9 if compartment == "tumor" else (0.45 if compartment == "boundary" else 0.15),
                    "immune_probability": 0.8 if local_index % 2 == 0 else 0.2,
                    "array_col": float(local_index % 4) + float(graph_index) * 10.0,
                    "array_row": float(local_index // 4),
                }
            )
    return pd.DataFrame(rows)


def build_spatial_expression(obs: pd.DataFrame) -> np.ndarray:
    matrix = np.full((len(obs), len(GENES)), 0.05, dtype=np.float32)
    gene_index = {gene: idx for idx, gene in enumerate(GENES)}
    for row_index, row in enumerate(obs.itertuples(index=False)):
        if row.time_label == "Hot":
            matrix[row_index, gene_index["CD3D"]] += 2.5
            matrix[row_index, gene_index["NKG7"]] += 2.0
            matrix[row_index, gene_index["IFNG"]] += 2.2
            matrix[row_index, gene_index["CXCL10"]] += 1.8
            matrix[row_index, gene_index["STAT1"]] += 1.7
        elif row.time_label == "Excluded":
            matrix[row_index, gene_index["COL1A1"]] += 2.4
            matrix[row_index, gene_index["TAGLN"]] += 1.9
            matrix[row_index, gene_index["EPCAM"]] += 1.4
            matrix[row_index, gene_index["KRT8"]] += 1.3
            matrix[row_index, gene_index["KRT19"]] += 1.3
        else:
            matrix[row_index, gene_index["EPCAM"]] += 1.0
            matrix[row_index, gene_index["ERBB2"]] += 0.9
        if row.compartment == "tumor":
            matrix[row_index, gene_index["EPCAM"]] += 2.3
            matrix[row_index, gene_index["KRT8"]] += 1.9
            matrix[row_index, gene_index["KRT19"]] += 1.7
        elif row.compartment == "stroma":
            matrix[row_index, gene_index["COL1A1"]] += 1.8
            matrix[row_index, gene_index["TAGLN"]] += 1.6
            matrix[row_index, gene_index["PECAM1"]] += 0.8
            matrix[row_index, gene_index["VWF"]] += 0.7
        matrix[row_index] = np.log1p(matrix[row_index])
    return matrix


def build_scrna_reference(project_root: Path) -> Path:
    ad_module = require_h5ad()
    output_dir = project_root / "outputs/mock_scrna/gse161529_mock__reference_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    obs = pd.DataFrame(
        {
            "cell_id": [f"ref_cell_{index:03d}" for index in range(30)],
            "sample_id": ["ref_sample"] * 30,
            "patient_id": ["ref_patient"] * 30,
            "cohort_id": ["mock_ref"] * 30,
            "cell_type": ["T_NK"] * 10 + ["CAF"] * 10 + ["Epithelial_Malignant"] * 10,
        }
    )
    base = np.random.default_rng(7).normal(loc=0.1, scale=0.02, size=(30, len(GENES))).clip(min=0.01)
    gene_index = {gene: idx for idx, gene in enumerate(GENES)}
    base[:10, [gene_index["CD3D"], gene_index["NKG7"], gene_index["IFNG"], gene_index["CXCL10"]]] += 2.0
    base[10:20, [gene_index["COL1A1"], gene_index["TAGLN"], gene_index["PECAM1"]]] += 1.5
    base[20:, [gene_index["EPCAM"], gene_index["KRT8"], gene_index["KRT19"], gene_index["ERBB2"]]] += 1.8
    matrix = np.log1p(base).astype(np.float32)
    adata = ad_module.AnnData(X=matrix, obs=obs.copy(), var=pd.DataFrame({"gene_symbol": GENES}, index=GENES))
    graph_ready_path = output_dir / "gse161529_mock__reference_v1__step-08_graph_ready.h5ad"
    adata.write_h5ad(graph_ready_path)

    gene_panel = pd.DataFrame(
        {
            "gene_id": GENES,
            "gene_symbol": GENES,
            "highly_variable": True,
        }
    )
    write_table(gene_panel, output_dir / "gse161529_mock__reference_v1__step-08_gene_panel.tsv")
    manifest = {
        "dataset_id": "gse161529_mock",
        "outputs": {
            "gene_panel": str(output_dir / "gse161529_mock__reference_v1__step-08_gene_panel.tsv"),
            "graph_ready_h5ad": str(graph_ready_path),
        },
    }
    manifest_path = output_dir / "gse161529_mock__reference_v1__step-08_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def build_priors(project_root: Path) -> Path:
    output_dir = project_root / "outputs/mock_priors/string_kegg_mock_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_master = pd.DataFrame(
        {
            "canonical_gene_symbol": GENES,
            "canonical_gene_symbol_norm": GENES,
            "in_gene_universe": True,
        }
    )
    gene_gene_edges = pd.DataFrame(
        [
            ("CD3D", "NKG7", 0.90, 900, 1),
            ("IFNG", "CXCL10", 0.87, 870, 1),
            ("COL1A1", "TAGLN", 0.82, 820, 1),
            ("EPCAM", "KRT8", 0.88, 880, 1),
            ("KRT8", "KRT19", 0.84, 840, 1),
            ("PECAM1", "VWF", 0.80, 800, 1),
        ],
        columns=["source_gene_symbol_norm", "target_gene_symbol_norm", "edge_weight", "combined_score_mean", "support_edge_count"],
    )
    gene_gene_edges["source_gene_symbol"] = gene_gene_edges["source_gene_symbol_norm"]
    gene_gene_edges["target_gene_symbol"] = gene_gene_edges["target_gene_symbol_norm"]
    gene_gene_edges = gene_gene_edges[
        [
            "source_gene_symbol",
            "source_gene_symbol_norm",
            "target_gene_symbol",
            "target_gene_symbol_norm",
            "support_edge_count",
            "combined_score_mean",
            "edge_weight",
        ]
    ]
    pathway_catalog = pd.DataFrame(
        {
            "pathway_id": ["path:hsa04660", "path:hsa04510", "path:hsa05224"],
            "pathway_name": ["T cell receptor signaling pathway", "Focal adhesion", "Breast cancer"],
            "pathway_top_class": ["Organismal Systems", "Cellular Processes", "Human Diseases"],
            "total_mapped_genes": [4, 4, 4],
            "selected_for_graph": [True, True, True],
        }
    )
    gene_pathway_edges = pd.DataFrame(
        [
            ("CD3D", "path:hsa04660", "T cell receptor signaling pathway", "Organismal Systems", 4, 0.5),
            ("NKG7", "path:hsa04660", "T cell receptor signaling pathway", "Organismal Systems", 4, 0.5),
            ("IFNG", "path:hsa04660", "T cell receptor signaling pathway", "Organismal Systems", 4, 0.5),
            ("CXCL10", "path:hsa04660", "T cell receptor signaling pathway", "Organismal Systems", 4, 0.5),
            ("COL1A1", "path:hsa04510", "Focal adhesion", "Cellular Processes", 4, 0.5),
            ("TAGLN", "path:hsa04510", "Focal adhesion", "Cellular Processes", 4, 0.5),
            ("PECAM1", "path:hsa04510", "Focal adhesion", "Cellular Processes", 4, 0.5),
            ("VWF", "path:hsa04510", "Focal adhesion", "Cellular Processes", 4, 0.5),
            ("EPCAM", "path:hsa05224", "Breast cancer", "Human Diseases", 4, 0.5),
            ("KRT8", "path:hsa05224", "Breast cancer", "Human Diseases", 4, 0.5),
            ("KRT19", "path:hsa05224", "Breast cancer", "Human Diseases", 4, 0.5),
            ("ERBB2", "path:hsa05224", "Breast cancer", "Human Diseases", 4, 0.5),
        ],
        columns=[
            "gene_symbol",
            "pathway_id",
            "pathway_name",
            "pathway_top_class",
            "mapped_gene_count",
            "edge_weight",
        ],
    )
    gene_pathway_edges["gene_symbol_norm"] = gene_pathway_edges["gene_symbol"]
    gene_pathway_edges["edge_weight_rule"] = "1 / sqrt(mapped_gene_count)"

    write_table(gene_master, output_dir / "string_kegg_mock_v1__step-04_gene_master.tsv.gz")
    write_table(gene_gene_edges, output_dir / "string_kegg_mock_v1__step-05_gene_gene_edges.tsv.gz")
    write_table(gene_pathway_edges, output_dir / "string_kegg_mock_v1__step-06_gene_pathway_edges.tsv.gz")
    write_table(pathway_catalog, output_dir / "string_kegg_mock_v1__step-03_pathway_catalog.tsv.gz")
    manifest = {"output_dir": str(output_dir)}
    manifest_path = output_dir / "string_kegg_mock_v1__step-07_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def build_spatial(project_root: Path) -> Path:
    ad_module = require_h5ad()

    graph_specs = []
    labels = ["Hot", "Excluded", "Cold"]
    for i in range(15):
        lab = labels[i % 3]
        graph_specs.append({
            "graph_id": f"patient_{i}__sample_{i}__slide_{i}",
            "patient_id": f"patient_{i}",
            "sample_id": f"sample_{i}",
            "slide_id": f"slide_{i}",
            "cohort_id": "mock_spatial",
            "split_id": "train" if i < 10 else "val",
            "time_label": lab,
            "infiltration_score": 0.82 if lab == "Hot" else (0.5 if lab == "Excluded" else 0.2),
            "penetration_score": 0.79 if lab == "Hot" else 0.3,
            "retention_score": 0.21 if lab == "Hot" else 0.8,
            "activation_score": 0.75 if lab == "Hot" else 0.4,
        })

    obs = build_spatial_obs(graph_specs)
    matrix = build_spatial_expression(obs)
    adata = ad_module.AnnData(X=matrix, obs=obs.set_index("node_id"), var=pd.DataFrame({"gene_symbol": GENES}, index=GENES))
    spatial_input_dir = project_root / "datasets/mock_spatial/processed"
    spatial_input_dir.mkdir(parents=True, exist_ok=True)
    spatial_input_path = spatial_input_dir / "input.h5ad"
    adata.write_h5ad(spatial_input_path)

    output_dir = project_root / "outputs/mock_spatial/visium_mock__spatial_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = obs.copy()
    nodes["graph_id"] = nodes["patient_id"] + "__" + nodes["sample_id"] + "__" + nodes["slide_id"]
    nodes["local_node_index"] = nodes.groupby("graph_id").cumcount()
    nodes["coord_x_scaled"] = nodes.groupby("graph_id")["array_col"].transform(lambda s: s - s.mean())
    nodes["coord_y_scaled"] = nodes.groupby("graph_id")["array_row"].transform(lambda s: s - s.mean())
    nodes = nodes.rename(columns={"array_col": "coord_x", "array_row": "coord_y"})
    write_table(nodes, output_dir / "visium_mock__spatial_v1__step-01_spatial_nodes.tsv.gz")

    edge_rows = []
    for graph_id, frame in nodes.groupby("graph_id"):
        frame = frame.sort_values("local_node_index")
        for source in frame.itertuples(index=False):
            for target in frame.itertuples(index=False):
                if target.local_node_index <= source.local_node_index:
                    continue
                distance = float(np.hypot(source.coord_x - target.coord_x, source.coord_y - target.coord_y))
                if distance <= 1.6:
                    edge_rows.append(
                        {
                            "graph_id": graph_id,
                            "patient_id": source.patient_id,
                            "sample_id": source.sample_id,
                            "cohort_id": source.cohort_id,
                            "slide_id": source.slide_id,
                            "split_id": source.split_id,
                            "variant": "primary",
                            "source_local_index": source.local_node_index,
                            "target_local_index": target.local_node_index,
                            "source_node_id": source.node_id,
                            "target_node_id": target.node_id,
                            "source_compartment": source.compartment,
                            "target_compartment": target.compartment,
                            "same_compartment": source.compartment == target.compartment,
                            "distance_euclidean": distance,
                            "distance_normalized": distance,
                            "edge_weight": float(np.exp(-0.5 * (distance**2))),
                        }
                    )
    edges = pd.DataFrame(edge_rows)
    write_table(edges, output_dir / "visium_mock__spatial_v1__step-03_edges__primary.tsv.gz")
    manifest = {
        "input_path": str(spatial_input_path),
        "outputs": {
            "nodes": str(output_dir / "visium_mock__spatial_v1__step-01_spatial_nodes.tsv.gz"),
            "edges_primary": str(output_dir / "visium_mock__spatial_v1__step-03_edges__primary.tsv.gz"),
        },
    }
    manifest_path = output_dir / "visium_mock__spatial_v1__step-03_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    scrna_manifest = build_scrna_reference(project_root)
    prior_manifest = build_priors(project_root)
    spatial_manifest = build_spatial(project_root)
    config_path = project_root / "configs/hetero_graph.mock.yaml"
    payload = {
        "input": {
            "dataset_id": "visium_mock",
            "scrna_manifest_path": str(scrna_manifest),
            "prior_manifest_path": str(prior_manifest),
            "spatial_manifest_path": str(spatial_manifest),
            "spatial_expression_path": None,
            "spatial_expression_gene_symbol_column": "gene_symbol",
            "assume_log1p_input": True,
        },
        "labels": {
            "time_label_column": "time_label",
            "is_uncertain_column": "is_uncertain",
            "phenotype_columns": {
                "infiltration_score": "infiltration_score",
                "penetration_score": "penetration_score",
                "retention_score": "retention_score",
                "activation_score": "activation_score",
            },
        },
        "output": {
            "root_dir": "outputs/hetero_graph/visium_mock__hetero_v1",
            "prefix": "visium_mock__hetero_v1",
        },
    }
    write_json(config_path, payload)
    print(str(config_path))


if __name__ == "__main__":
    main()
