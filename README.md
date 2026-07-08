# HGT-TIME Classification

Code and configurations for HGT-TIME, a heterogeneous-graph pipeline that types the tumor
immune microenvironment (TIME) into inflamed (Hot), immune-excluded (Excluded), and
immune-desert (Cold) phenotypes from spatial transcriptomics. Each tissue region is built into a
heterogeneous graph of cell, gene, and pathway nodes that integrates spatial adjacency, gene
expression, protein-protein interactions, pathway membership, and transcription-factor
regulation, and the graph is encoded by a heterogeneous graph transformer.

This repository contains source code and configuration files only. Datasets, cached downloads,
model checkpoints, logs, and generated outputs are excluded from version control.

## Requirements

- Python 3.10 or 3.11
- Linux or Windows
- A PyTorch and PyTorch Geometric build matching your CUDA or CPU setup

## Installation

Install a PyTorch build for your platform first, then the remaining dependencies. The example
below uses CPU wheels; for CUDA, substitute the matching command from the official PyTorch
selector.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. If script execution is
blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first.

## Repository layout

```text
configs/            experiment, preprocessing, graph, ablation, and fusion configs
  pretraining/      multimodal pretraining configs
models/             HGT-TIME model, baselines, losses, graph transforms, fusion, DG
scripts/            preprocessing, training, evaluation, ablation, interpretability
  pretraining/      GraphCL / multimodal pretraining entry points
requirements*.txt   aggregate (requirements.txt) and per-stage requirement files
```

All configs use repository-relative paths, so run commands from the repository root. Place inputs
under `datasets/` (public spatial-transcriptomics data) and `data/` (auxiliary assets); the
pipeline writes to `outputs/` and run logs to `logs/`.

## Minimal smoke test

This runs end to end on generated mock inputs and needs no external data.

```bash
python scripts/generate_mock_hetero_inputs.py
python scripts/build_hetero_graphs.py --config configs/hetero_graph.mock.yaml
python scripts/smoke_test_hgt_model.py --config configs/hgt_time.mock.yaml
```

## Reproducible pipeline

### 1. Build preprocessing artifacts

```bash
python scripts/scrna_preprocess.py    --config configs/scrna_preprocess.default.yaml
python scripts/build_prior_resources.py --config configs/prior_builder.default.yaml
python scripts/spatial_adjacency.py    --config configs/spatial_adjacency.default.yaml
python scripts/build_hetero_graphs.py  --config configs/hetero_graph.default.yaml
```

The prior builder assembles gene-gene edges from STRING, pathway membership from KEGG, and
transcription-factor to target edges from TRRUST. The graph builder emits one `graphs/<id>.pt`
per region together with a feature schema and manifest.

### 2. Train a single experiment

```bash
python scripts/train_eval_pipeline.py --config configs/hgt_time.default.yaml --seeds 42 123 2026
```

### 3. Run the registered suite

```bash
python scripts/train_eval_pipeline.py --run-all-registry --registry configs/experiment_registry.yaml
```

### 4. Aggregate results

```bash
python scripts/evaluate.py --results-dir outputs/results --format markdown
```

## Training protocol

Baselines and the main model share `scripts/train_eval_pipeline.py`, so data splits, metrics, and
logging are identical across models. Key settings, also stored in each config:

- Splitting: patient-grouped stratified k-fold cross-validation, so no patient appears in both
  train and validation.
- Seeds: three by default (42, 123, 2026); metrics are reported as mean and standard deviation.
- Early stopping: on validation macro-AUROC, patience 15.
- Schedule and clipping: cosine annealing, gradient-norm clipping at 1.0.
- Artifacts: per-fold best model to `checkpoints/`, per-experiment `results.json`, and a global
  `results_summary.tsv`.

## Experiment registry

Experiments are declared in `configs/experiment_registry.yaml`.

| ID | Model | Role |
| --- | --- | --- |
| EXP-B01-LINEAR | Linear deconvolution | baseline |
| EXP-B02-MLP | Non-graph multilayer perceptron | baseline |
| EXP-B03-GCN | Homogeneous GCN (spatial only) | baseline |
| EXP-B04-GAT | Homogeneous GAT (spatial only) | baseline |
| EXP-M01-HGT | HGT-TIME | main |

Reported metrics are macro-AUROC, macro-F1, and balanced accuracy, with macro-AUPRC, Brier score,
and expected calibration error as secondary diagnostics.

## Ablations

Ablation configs (`configs/ablation_*.yaml`) apply graph-level transforms from
`models/graph_transforms.py` after data loading.

| Config | Hypothesis | Transform |
| --- | --- | --- |
| ablation_h1_no_spatial | spatial edges | drop spatial edges |
| ablation_h1_permute_spatial | spatial edges | permute spatial edges |
| ablation_h2_no_pathway | typed schema | drop pathway nodes |
| ablation_h2_no_ppi | typed schema | drop protein-interaction edges |
| ablation_h2_homo_collapse | typed schema | collapse to a homogeneous graph |
| ablation_h3_no_ranking | auxiliary loss | drop ranking targets |

```bash
python scripts/train_eval_pipeline.py --config configs/ablation_h1_no_spatial.yaml --seeds 42 123 2026
```

## Generalization and fusion

- Cross-patient leave-one-patient-out: `scripts/train_eval_pipeline.py` with a LOPO config, and a
  stricter leave-spatial-block-out control in `scripts/leave_block_out.py`.
- Cross-platform (Visium to Xenium) and cross-cancer training use the matching configs under
  `configs/` and the graph builder on the target dataset.
- Reliability-weighted fusion of the graph, expression, and morphology views:
  `scripts/reliability_fusion_lopo.py`.
- Held-out immune-gene ranking recovery: `scripts/heldout_gene_recovery.py`.
- Class-balanced and base-rate-corrected calibration: `scripts/reviewer_calibration.py`.

## Interpretability

`scripts/interpretability_analysis.py` produces ranking stability (top-k Jaccard and Spearman
correlation across folds and seeds), perturbation sensitivity (per-node feature masking and the
resulting change in target-class probability), and biological concordance (overlap with known
immune genes and pathway enrichment).

```bash
python scripts/interpretability_analysis.py --config configs/hgt_time.default.yaml \
  --experiment-id EXP-M01-HGT --topk 50
```

`scripts/subtype_consensus_ranking.py` computes per-phenotype ranking consensus.

## Model interface

The main model is in `models/hgt_time_model.py`. `forward(batch)` returns `graph_logits`,
`graph_probs`, `pheno_pred`, `gene_score`, `pathway_score`, `embedding`, and `readout`, where
`readout` retains the typed-pooling weights used for interpretability and ablation.

## Optional

```bash
# Hyperparameter search (Optuna TPE with median pruning)
python scripts/hparam_search.py --config configs/hgt_time.default.yaml --n-trials 30

# End-to-end orchestration: baselines, search, main model, and comparison report
python scripts/run_full_pipeline.py --phases 1 2 3 --config configs/full_pipeline.yaml
```

## Data sources

All datasets used are public: breast, colorectal, and lung spatial transcriptomics from HEST-1k;
the breast single-cell reference from GEO (GSE161529); protein-interaction, pathway, and
regulatory priors from STRING, KEGG, and TRRUST. Place them under `datasets/` and `data/` as
described above.

## License

The code is released for academic research use. Add a license file of your choice before public distribution.
