# Manuscript results

## Classification and training-set size

| Manuscript item | Expression configuration | Cell expression features | Result file and key | HGT / GCN AUROC |
| --- | --- | ---: | --- | --- |
| Table 1, colorectal | Section-aligned | 8 | [hvg_budget_sweep.json](results/hgt_time/hvg_budget_sweep.json), `8` | 0.903 / 0.775 |
| Table 1, basal cell carcinoma | Section-aligned | 8 | [hvg_budget_sweep_BCC.json](results/hgt_time/hvg_budget_sweep_BCC.json), `8` | 0.873 / 0.561 |
| Table 2, absolute | Absolute expression | 48 | [colorectal_benchmark.json](results/colorectal_benchmark.json) | 0.793 / 0.805 |
| Table 2, aligned | Section-aligned | 48 | [hvg_budget_sweep.json](results/hgt_time/hvg_budget_sweep.json), `48` | 0.874 / 0.852 |
| Table 2, ensemble | Section-aligned | 48 | [ensemble_FINAL.json](results/hgt_time/ensemble_FINAL.json) | 0.896 / — |
| Table S7, 10% training | Section-aligned | 8 | [decisive_hvg8.json](results/hgt_time/decisive_hvg8.json), `0.1` | 0.733 / 0.670 |
| Table S7, full training | Section-aligned | 8 | [decisive_hvg8.json](results/hgt_time/decisive_hvg8.json), `1.0` | 0.903 / 0.775 |

The feature counts refer to cell-node expression features. Cell nodes also include two spatial coordinates; gene and pathway nodes carry molecular features and prior relations.

The K=8 feature sweep and full-training experiment have separate aggregate means, 0.9025923071 and 0.9025553165. Both round to 0.903. The earlier [colorectal_learning_curve.json](results/colorectal_learning_curve.json) evaluates absolute-expression graphs. Table S7 uses the section-aligned K=8 experiment in `decisive_hvg8.json`.

The absolute and section-aligned configurations use their corresponding HGT settings: four heads, dropout 0.2 and learning rate 0.001 for absolute expression; eight heads, dropout 0.1 and learning rate 0.0003 for section-aligned expression. Full parameters, grouping fields and seeds are in [configs/hgt_time](configs/hgt_time).

## Tables and figures

[experiment_manifest.json](configs/hgt_time/experiment_manifest.json) maps the main and supplementary tables and result panels to repository files, configuration IDs and numerical fields. It includes file checksums and full-precision queries for the headline AUROCs.

- [Main numerical tables](results/hgt_time/main_tables.tex)
- [Supplementary numerical tables](results/hgt_time/supplementary_tables.tex)
- [Figure source summaries](results/hgt_time/figure_data)
- [Experiment result summaries](results/hgt_time)

Configuration IDs label experiment settings. Run-level fields in the manifest use `null` where original graph-version manifests, saved fold assignments, execution IDs or training logs are unavailable.

## Experiment scripts

Run from the repository root after preparing the processed graphs under `outputs/hetero_graph/<tag>/graphs/`:

```bash
python scripts/hgt_time/hvg_sweep.py
python scripts/hgt_time/hvg_bcc.py
python scripts/hgt_time/hvg_bcc_mid.py
python scripts/hgt_time/decisive.py
python scripts/hgt_time/lc_norm.py
```

Graph tags and seeds are recorded in the configuration snapshots. These JSON files describe the experiments; the scripts define their execution settings. Basal cell carcinoma graphs use patient identifiers in `sample_id` for grouping. The colorectal scripts group by section/sample.

The scripts use the accompanying `scripts/hgt_time/train_eval_pipeline.py`. Its training functions match the repository's general wrapper; the two versions have different default CLI output and registry paths. Original script checksums and the repository-relative import changes are recorded in the manifest. The historical dependency commit is unspecified.

To inspect the published numerical summaries without loading data or training a model:

```bash
python scripts/check_manuscript_results.py
```
