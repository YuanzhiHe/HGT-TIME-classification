"""Hyperparameter and class-imbalance search for HGT-TIME on the powered colorectal cohort.
The cohort is heavily imbalanced (485 / 199 / 35), so the canonical configuration may understate
the model. Sweeps class weighting, depth/heads/hidden width, dropout, and learning rate under the
same leakage-controlled protocol (StratifiedGroupKFold by sample). Reports macro-AUROC per config
so the best configuration can be compared against the baselines on equal footing."""
import sys, glob, json, argparse, itertools
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np, torch
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold

BASE_RUNTIME = dict(batch_size=8, epochs=80, learning_rate=1e-3, weight_decay=1e-5,
                    patience=12, scheduler="cosine", scheduler_kwargs={"T_max": 80})
BASE_MODEL = dict(hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, num_classes=3, pheno_dim=4,
                  use_pheno_head=False, use_cell_state_head=False, cell_state_dim=4, use_ranking_heads=False)

# (tag, model overrides, runtime overrides, use_class_weights)
GRID = [
    ("canonical",            {},                                   {},                      False),
    ("class-weighted",       {},                                   {},                      True),
    ("wide+cw",              dict(hidden_dim=256),                 {},                      True),
    ("deep+cw",              dict(num_layers=4, num_heads=8),      {},                      True),
    ("shallow+cw",           dict(num_layers=2, num_heads=4),      {},                      True),
    ("dropout0.4+cw",        dict(dropout=0.4),                    {},                      True),
    ("lr3e-4+cw",            {},                                   dict(learning_rate=3e-4), True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="colorectal")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--out", default="outputs/results/colorectal_tuning.json")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [42]; BASE_RUNTIME["epochs"] = 5

    fs = sorted(glob.glob(f'outputs/hetero_graph/newcancer_{args.tag}/graphs/*.pt'))
    gs = [torch.load(f, weights_only=False) for f in fs]
    y = np.array([int(g.y_graph[0]) for g in gs]); grp = np.array([g.sample_id for g in gs])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnt = Counter(y.tolist())
    print(f'{args.tag}: n={len(gs)} labels={dict(cnt)} samples={len(set(grp))}', flush=True)
    # inverse-frequency class weights
    tot = sum(cnt.values()); cw = [tot / (len(cnt) * cnt.get(c, 1)) for c in range(3)]
    print('class weights:', [round(w, 3) for w in cw], flush=True)

    grid = [g for g in GRID if (args.configs is None or g[0] in args.configs)]
    results = {}
    for name, mov, rov, use_cw in grid:
        mkw = {**BASE_MODEL, **mov}
        rt = {**BASE_RUNTIME, **rov}
        loss = dict(classification_weight=1.0, phenotype_weight=0.0, region_weight=0.0,
                    ranking_weight=0.0, label_smoothing=0.05,
                    class_weights=(cw if use_cw else None))
        per = []
        for seed in args.seeds:
            P.set_seed(seed)
            sgkf = StratifiedGroupKFold(n_splits=args.folds)
            splits = list(sgkf.split(gs, y, grp))
            if args.smoke: splits = splits[:1]
            for fold, (tr, va) in enumerate(splits):
                ck = Path(f'outputs/results/TUNE-{args.tag}/{name}/ck'); ck.mkdir(parents=True, exist_ok=True)
                m = build_model('hgt_time', {**mkw}, gs[0])
                cfg = {'model_family': 'hgt_time', 'runtime': rt, 'loss': loss, 'model': mkw}
                r = train_one_fold(model=m,
                                   train_loader=DataLoader([gs[i] for i in tr], batch_size=rt["batch_size"], shuffle=True),
                                   val_loader=DataLoader([gs[i] for i in va], batch_size=rt["batch_size"], shuffle=False),
                                   config=cfg, device=dev, checkpoint_dir=ck, fold=fold, seed=seed)
                per.append(r['final_metrics'])
        def agg(k):
            v = [mm[k] for mm in per if k in mm and mm[k] == mm[k]]
            return (float(np.mean(v)), float(np.std(v))) if v else (float('nan'), 0.0)
        results[name] = {k: agg(k) for k in ['macro_auroc', 'macro_f1', 'balanced_accuracy']}
        r = results[name]
        print(f'  {name:16s} AUROC {r["macro_auroc"][0]:.3f}+-{r["macro_auroc"][1]:.3f} '
              f'F1 {r["macro_f1"][0]:.3f} BalAcc {r["balanced_accuracy"][0]:.3f}', flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, 'w'), indent=2)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
