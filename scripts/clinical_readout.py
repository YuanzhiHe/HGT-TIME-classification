"""Lightweight clinical readout for HGT-TIME (graph-free, GPU-free, laptop-runnable).

This is the deployment form of the pipeline for a pathology laboratory: it needs only the
expression of a 35-gene panel (no heterogeneous-graph construction, no deep model, no GPU). It
scores the immune-infiltration, immune-exclusion, and fibrotic-barrier signatures on each region
or spot, assigns the Hot / Excluded / Cold immune phenotype by pre-specified quantile rules, and
writes a per-spot table, a spatial phenotype map, and a one-page stratification summary.

It trades accuracy for simplicity and cost: on the internal breast benchmark this signature
readout reaches macro-AUROC 0.672 versus 0.781 for the full heterogeneous-graph model, so the
graph remains the research and discovery engine while this panel readout is the fast clinical
screen. Usage:

    python clinical_readout.py --input sample.h5ad --out-dir report/
    python clinical_readout.py --input expr.csv --coords coords.csv --out-dir report/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

# 35-gene deployable panel (immune infiltration | stromal exclusion | fibrotic barrier)
PANEL = {
    "infiltration": ["CD3D", "CD3E", "TRBC1", "TRBC2", "NKG7", "GNLY", "PRF1", "IFNG",
                     "CXCL9", "CXCL10", "CXCL13", "MS4A1", "CD79A", "LST1", "FCER1G", "TYROBP"],
    "exclusion": ["COL1A1", "COL1A2", "COL3A1", "TAGLN", "ACTA2", "THY1", "DCN", "LUM",
                  "TGFB1", "TGFBI", "FN1", "CXCL12"],
    "fibrotic_barrier": ["FAP", "POSTN", "PDGFRB", "PDGFRA", "SPARC", "CTHRC1", "COL1A1", "MMP2"],
}
ALL_GENES = sorted({g for v in PANEL.values() for g in v})


def load_expression(input_path: Path):
    """Return (expr DataFrame [spots x genes, log1p], coords DataFrame or None)."""
    if input_path.suffix == ".h5ad":
        import anndata as ad
        a = ad.read_h5ad(input_path)
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        expr = pd.DataFrame(X, index=a.obs_names.astype(str), columns=a.var_names.astype(str))
        coords = None
        for k in ("spatial", "X_spatial"):
            if k in a.obsm:
                coords = pd.DataFrame(np.asarray(a.obsm[k])[:, :2], index=expr.index, columns=["x", "y"])
        return expr, coords
    expr = pd.read_csv(input_path, index_col=0)
    if expr.shape[0] < expr.shape[1]:  # genes as rows -> transpose to spots x genes
        expr = expr.T
    return expr, None


def to_log1p(expr: pd.DataFrame) -> pd.DataFrame:
    if float(expr.to_numpy().max()) > 30:  # looks like raw counts
        return np.log1p(expr)
    return expr


def score(expr: pd.DataFrame, genes) -> np.ndarray:
    present = [g for g in genes if g in expr.columns]
    if not present:
        return np.zeros(expr.shape[0])
    z = (expr[present] - expr[present].mean()) / (expr[present].std() + 1e-9)
    return z.mean(axis=1).to_numpy()


def readout(expr: pd.DataFrame):
    inf = score(expr, PANEL["infiltration"])
    exc = score(expr, PANEL["exclusion"])
    fib = score(expr, PANEL["fibrotic_barrier"])
    qi, qr = np.quantile(inf, 0.66), np.quantile(exc, 0.60)
    phen = np.where(inf >= qi, "Hot", np.where(exc >= qr, "Excluded", "Cold"))
    return pd.DataFrame({"infiltration": inf, "exclusion": exc, "fibrotic_barrier": fib,
                         "phenotype": phen}, index=expr.index)


def render_map(res: pd.DataFrame, coords: pd.DataFrame, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    col = {"Hot": "#D1495B", "Excluded": "#E9A13B", "Cold": "#4A79C7"}
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=200)
    ax.scatter(coords["x"], -coords["y"], c=[col[p] for p in res["phenotype"]], s=8, edgecolors="none")
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title("Immune-phenotype map", fontsize=11)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7, label=l)
                       for l, c in col.items()], loc="lower left", frameon=False, fontsize=8)
    fig.savefig(out_png, bbox_inches="tight", dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser(description="Lightweight HGT-TIME clinical readout (panel-based).")
    ap.add_argument("--input", required=True, type=Path, help="h5ad or CSV expression matrix")
    ap.add_argument("--coords", type=Path, help="optional CSV of spot x,y coordinates")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    expr, coords = load_expression(args.input)
    expr = to_log1p(expr)
    if args.coords is not None:
        coords = pd.read_csv(args.coords, index_col=0).iloc[:, :2]
        coords.columns = ["x", "y"]
    res = readout(expr)
    res.to_csv(args.out_dir / "phenotype_per_spot.csv")

    frac = res["phenotype"].value_counts(normalize=True).reindex(["Hot", "Excluded", "Cold"]).fillna(0)
    report = {
        "n_spots": int(len(res)),
        "phenotype_fraction": {k: round(float(v), 3) for k, v in frac.items()},
        "mean_fibrotic_barrier_score": round(float(res["fibrotic_barrier"].mean()), 3),
        "dominant_phenotype": str(frac.idxmax()),
        "panel_genes_detected": int(sum(g in expr.columns for g in ALL_GENES)),
        "panel_genes_total": len(ALL_GENES),
        "note": "Signature-level screen (macro-AUROC 0.672 on internal benchmark); "
                "confirm ambiguous regions with the full HGT-TIME model or pathologist review.",
    }
    (args.out_dir / "stratification_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if coords is not None:
        render_map(res, coords, args.out_dir / "phenotype_map.png")
        print("wrote phenotype_map.png")
    print(f"wrote {args.out_dir}/phenotype_per_spot.csv and stratification_report.json")


if __name__ == "__main__":
    main()
