# HGT-TIME Classification

Standalone codebase for multimodal pretraining, spatial graph construction, heterogeneous graph learning, domain generalization, and interpretability analysis for tumor immune microenvironment classification.

This repository contains code and configs only. Datasets, cached downloads, checkpoints, logs, and generated outputs are excluded.

## Requirements

- Python 3.10 or 3.11
- Linux or Windows
- PyTorch / PyG environment compatible with your CUDA or CPU setup

## Repository Layout

```text
configs/
  pretraining/
models/
scripts/
  pretraining/
data/
datasets/
logs/
outputs/
requirements-*.txt
```

## Installation

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-scrna.txt
pip install -r requirements-priors.txt
pip install -r requirements-spatial.txt
pip install -r requirements-hetero.txt
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-scrna.txt
pip install -r requirements-priors.txt
pip install -r requirements-spatial.txt
pip install -r requirements-hetero.txt
```

## Expected Directories

The configs use repository-relative paths. Before running experiments, place data in:

- `datasets/` for public raw or processed datasets
- `data/` for auxiliary assets used by specific preprocessing or pretraining steps
- `outputs/` for generated manifests, graphs, checkpoints, and reports
- `logs/` for run logs

## Minimal Smoke Test

```bash
python scripts/generate_mock_hetero_inputs.py
python scripts/build_hetero_graphs.py --config configs/hetero_graph.mock.yaml
python scripts/smoke_test_hgt_model.py --config configs/hgt_time.mock.yaml
```

## Reproducible Pipeline

### 1. Build preprocessing artifacts

```bash
python scripts/scrna_preprocess.py --config configs/scrna_preprocess.default.yaml
python scripts/build_prior_resources.py --config configs/prior_builder.default.yaml
python scripts/spatial_adjacency.py --config configs/spatial_adjacency.default.yaml
python scripts/build_hetero_graphs.py --config configs/hetero_graph.default.yaml
```

### 2. Train a single experiment

```bash
python scripts/train_eval_pipeline.py \
  --config configs/hgt_time.default.yaml \
  --seeds 42 123 2026
```

### 3. Run the registered suite

```bash
python scripts/train_eval_pipeline.py \
  --run-all-registry \
  --registry configs/experiment_registry.yaml
```

### 4. Aggregate results

```bash
python scripts/evaluate.py \
  --results-dir outputs/results \
  --format markdown
```

## Optional

Hyperparameter search:

```bash
python scripts/hparam_search.py --config configs/hgt_time.default.yaml --n-trials 30
```

End-to-end orchestration:

```bash
python scripts/run_full_pipeline.py --phases 1 2 3 --config configs/full_pipeline.yaml
```

## Notes

- Main model: `models/hgt_time_model.py`
- Large files should stay out of version control
- If you use Windows and PowerShell script execution is blocked, run `Set-ExecutionPolicy -Scope Process Bypass`
