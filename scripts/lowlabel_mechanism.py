"""Which part of the graph prior buys label efficiency? (reviewer request)

Repeats the low-label point of the learning curve (and the full-data point for reference) with
ablated graphs, so the advantage can be attributed to a component rather than to "the graph" as a
whole. Same colorectal cohort, same sample-level subsampling and StratifiedGroupKFold protocol as
learning_curve_colorectal.py.

Variants: full model; no spatial cell-cell edges; no pathway nodes; no gene-gene prior edges
(STRING/TRRUST); no gene nodes at all.
"""
import sys, glob, json, argparse, copy
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np, torch
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold
from models.graph_transforms import TRANSFORM_REGISTRY as TRANSFORMS

RUNTIME = dict(batch_size=8, epochs=80, learning_rate=1e-3, weight_decay=1e-5,
               patience=12, scheduler="cosine", scheduler_kwargs={"T_max": 80})
LOSS = dict(classification_weight=1.0, phenotype_weight=0.0, region_weight=0.0,
            ranking_weight=0.0, label_smoothing=0.05, class_weights=None)
HGT = dict(hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, num_classes=3, pheno_dim=4,
           use_pheno_head=False, use_cell_state_head=False, cell_state_dim=4, use_ranking_heads=False)

VARIANTS = {
    "full":            [],
    "no spatial edges": ["drop_spatial_edges"],
    "no pathway nodes": ["drop_pathway_nodes"],
    "no prior edges":   ["drop_prior_edges"],
    "no gene nodes":    ["drop_gene_nodes"],
}


def transform_all(graphs, names):
    if not names:
        return graphs
    out = []
    for g in graphs:
        h = copy.deepcopy(g)
        for n in names:
            h = TRANSFORMS[n](h)
        out.append(h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="colorectal")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.1, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="outputs/results/lowlabel_mechanism.json")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [42]; RUNTIME["epochs"] = 5

    fs = sorted(glob.glob(f'outputs/hetero_graph/newcancer_{args.tag}/graphs/*.pt'))
    base = [torch.load(f, weights_only=False) for f in fs]
    y = np.array([int(g.y_graph[0]) for g in base]); grp = np.array([g.sample_id for g in base])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{args.tag}: n={len(base)} labels={dict(Counter(y.tolist()))} samples={len(set(grp))}', flush=True)

    results = {}
    for vname in args.variants:
        names = VARIANTS[vname]
        gs = transform_all(base, names)
        results[vname] = {}
        for frac in args.fracs:
            per = []
            for seed in args.seeds:
                P.set_seed(seed)
                sgkf = StratifiedGroupKFold(n_splits=args.folds)
                splits = list(sgkf.split(gs, y, grp))
                if args.smoke: splits = splits[:1]
                for fold, (tr, va) in enumerate(splits):
                    tr_samples = sorted(set(grp[tr]))
                    r = np.random.default_rng(1000 * seed + fold)
                    k = max(2, int(round(len(tr_samples) * frac)))
                    keep = set(r.choice(tr_samples, size=k, replace=False).tolist())
                    tr_sub = [i for i in tr if grp[i] in keep]
                    if len(set(y[tr_sub])) < 2:
                        continue
                    ck = Path(f'outputs/results/MECH-{args.tag}/{vname.replace(" ","_")}/ck'); ck.mkdir(parents=True, exist_ok=True)
                    m = build_model('hgt_time', {**HGT}, gs[0])
                    cfg = {'model_family': 'hgt_time', 'runtime': RUNTIME, 'loss': LOSS, 'model': HGT}
                    res = train_one_fold(model=m,
                                         train_loader=DataLoader([gs[i] for i in tr_sub], batch_size=RUNTIME["batch_size"], shuffle=True),
                                         val_loader=DataLoader([gs[i] for i in va], batch_size=RUNTIME["batch_size"], shuffle=False),
                                         config=cfg, device=dev, checkpoint_dir=ck, fold=fold, seed=seed)
                    per.append(res['final_metrics'])
            v = [m['macro_auroc'] for m in per if 'macro_auroc' in m and m['macro_auroc'] == m['macro_auroc']]
            mu, sd = (float(np.mean(v)), float(np.std(v))) if v else (float('nan'), 0.0)
            results[vname][str(frac)] = dict(auroc=mu, sd=sd, n_runs=len(v),
                                             approx_train_regions=int(round(len(base) * 0.8 * frac)))
            print(f'  {vname:18s} frac={frac:<5} AUROC {mu:.3f}+-{sd:.3f} (n={len(v)})', flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(results, open(args.out, 'w'), indent=2)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
