"""Well-powered baseline sweep on the 719-region colorectal cohort (addresses reviewer
request to establish the central benchmark on a cohort larger than the 36-region breast block).
Same leakage-controlled protocol as train_newcancer.py: 5-fold StratifiedGroupKFold grouped by
sample, three seeds. Reports mean +/- SD of macro-AUROC, macro-F1, balanced accuracy per model."""
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
    "Non-graph MLP":   ("non_graph",         dict(hidden_dim=128, num_classes=3, num_layers=3, dropout=0.2)),
    "Homogeneous GCN": ("homogeneous_graph", dict(hidden_dim=128, num_classes=3, num_layers=2, dropout=0.2, conv_type="gcn")),
    "Homogeneous GAT": ("homogeneous_graph", dict(hidden_dim=128, num_classes=3, num_layers=2, dropout=0.2, conv_type="gat", heads=4)),
    "Linear deconv.":  ("linear_deconvolution", dict(num_classes=3)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="colorectal")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 fold, 5 epochs")
    ap.add_argument("--class-weighted", action="store_true", help="apply inverse-frequency class weights to every model")
    ap.add_argument("--out", default="outputs/results/colorectal_benchmark.json")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = [42]; RUNTIME["epochs"] = 5

    fs = sorted(glob.glob(f'outputs/hetero_graph/newcancer_{args.tag}/graphs/*.pt'))
    gs = [torch.load(f, weights_only=False) for f in fs]
    y = np.array([int(g.y_graph[0]) for g in gs]); grp = np.array([g.sample_id for g in gs])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnt = Counter(y.tolist())
    print(f'{args.tag}: n={len(gs)} labels={dict(cnt)} samples={len(set(grp))} device={dev}', flush=True)
    if args.class_weighted:
        tot = sum(cnt.values()); cw = [tot / (len(cnt) * cnt.get(c, 1)) for c in range(3)]
        LOSS['class_weights'] = cw
        print('class weights (all models):', [round(w, 3) for w in cw], flush=True)

    results = {}
    for name in args.models:
        family, mkw = MODELS[name]
        per = []
        sgkf = StratifiedGroupKFold(n_splits=args.folds)
        for seed in args.seeds:
            P.set_seed(seed)
            splits = list(sgkf.split(gs, y, grp))
            if args.smoke: splits = splits[:1]
            for fold, (tr, va) in enumerate(splits):
                ck = Path(f'outputs/results/BENCH-{args.tag}/{family}_{mkw.get("conv_type","")}/ck'); ck.mkdir(parents=True, exist_ok=True)
                m = build_model(family, {**mkw}, gs[0])
                cfg = {'model_family': family, 'runtime': RUNTIME, 'loss': LOSS, 'model': mkw}
                r = train_one_fold(model=m,
                                   train_loader=DataLoader([gs[i] for i in tr], batch_size=RUNTIME["batch_size"], shuffle=True),
                                   val_loader=DataLoader([gs[i] for i in va], batch_size=RUNTIME["batch_size"], shuffle=False),
                                   config=cfg, device=dev, checkpoint_dir=ck, fold=fold, seed=seed)
                per.append(r['final_metrics'])
        def agg(k):
            v = [mm[k] for mm in per if k in mm and mm[k] == mm[k]]
            return (float(np.mean(v)), float(np.std(v))) if v else (float('nan'), 0.0)
        results[name] = {k: agg(k) for k in ['macro_auroc', 'macro_f1', 'balanced_accuracy']}
        r = results[name]
        print(f'  {name:16s} AUROC {r["macro_auroc"][0]:.3f}+-{r["macro_auroc"][1]:.2f} '
              f'F1 {r["macro_f1"][0]:.3f} BalAcc {r["balanced_accuracy"][0]:.3f}', flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, 'w'), indent=2)
    print("saved", args.out, flush=True)


if __name__ == '__main__':
    main()
