"""Learning curve on the colorectal cohort: does the heterogeneous knowledge prior help most
when training data are scarce? Subsamples the training split to a fraction of samples (grouped by
sample_id to keep the leakage control), retrains each model family, and evaluates on the held-out
fold. Tests the hypothesis that HGT-TIME's advantage on the small 36-region breast benchmark
reflects a genuine low-data advantage rather than noise."""
import sys, glob, json, argparse
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np, torch
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold

RUNTIME = dict(batch_size=8, epochs=80, learning_rate=1e-3, weight_decay=1e-5,
               patience=12, scheduler="cosine", scheduler_kwargs={"T_max": 80})
LOSS = dict(classification_weight=1.0, phenotype_weight=0.0, region_weight=0.0,
            ranking_weight=0.0, label_smoothing=0.05, class_weights=None)
HGT = dict(hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, num_classes=3, pheno_dim=4,
           use_pheno_head=False, use_cell_state_head=False, cell_state_dim=4, use_ranking_heads=False)

MODELS = {
    "HGT-TIME":        ("hgt_time",          HGT),
    "Homogeneous GCN": ("homogeneous_graph", dict(hidden_dim=128, num_classes=3, num_layers=2, dropout=0.2, conv_type="gcn")),
    "Non-graph MLP":   ("non_graph",         dict(hidden_dim=128, num_classes=3, num_layers=3, dropout=0.2)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="colorectal")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="outputs/results/colorectal_learning_curve.json")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [42]; args.fracs = [0.1, 1.0]; RUNTIME["epochs"] = 5

    fs = sorted(glob.glob(f'outputs/hetero_graph/newcancer_{args.tag}/graphs/*.pt'))
    gs = [torch.load(f, weights_only=False) for f in fs]
    y = np.array([int(g.y_graph[0]) for g in gs]); grp = np.array([g.sample_id for g in gs])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{args.tag}: n={len(gs)} labels={dict(Counter(y.tolist()))} samples={len(set(grp))}', flush=True)

    results = {}
    for name, (family, mkw) in MODELS.items():
        results[name] = {}
        for frac in args.fracs:
            per = []
            for seed in args.seeds:
                P.set_seed(seed)
                sgkf = StratifiedGroupKFold(n_splits=args.folds)
                splits = list(sgkf.split(gs, y, grp))
                if args.smoke: splits = splits[:1]
                for fold, (tr, va) in enumerate(splits):
                    # subsample TRAIN samples (group level) to `frac`
                    tr_samples = sorted(set(grp[tr]))
                    r = np.random.default_rng(1000 * seed + fold)
                    k = max(2, int(round(len(tr_samples) * frac)))
                    keep = set(r.choice(tr_samples, size=k, replace=False).tolist())
                    tr_sub = [i for i in tr if grp[i] in keep]
                    if len(set(y[tr_sub])) < 2:      # need >=2 classes to train
                        continue
                    ck = Path(f'outputs/results/LC-{args.tag}/{family}/ck'); ck.mkdir(parents=True, exist_ok=True)
                    m = build_model(family, {**mkw}, gs[0])
                    cfg = {'model_family': family, 'runtime': RUNTIME, 'loss': LOSS, 'model': mkw}
                    res = train_one_fold(model=m,
                                         train_loader=DataLoader([gs[i] for i in tr_sub], batch_size=RUNTIME["batch_size"], shuffle=True),
                                         val_loader=DataLoader([gs[i] for i in va], batch_size=RUNTIME["batch_size"], shuffle=False),
                                         config=cfg, device=dev, checkpoint_dir=ck, fold=fold, seed=seed)
                    per.append(res['final_metrics'])
            v = [m['macro_auroc'] for m in per if 'macro_auroc' in m and m['macro_auroc'] == m['macro_auroc']]
            mu, sd = (float(np.mean(v)), float(np.std(v))) if v else (float('nan'), 0.0)
            n_train = int(round(len(gs) * 0.8 * frac))
            results[name][str(frac)] = dict(auroc=mu, sd=sd, n_runs=len(v), approx_train_regions=n_train)
            print(f'  {name:16s} frac={frac:<5} (~{n_train:4d} train regions) AUROC {mu:.3f}+-{sd:.3f} (n={len(v)})', flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(results, open(args.out, 'w'), indent=2)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
