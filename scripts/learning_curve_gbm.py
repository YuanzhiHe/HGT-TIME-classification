"""Second non-graph baseline for the label-efficiency analysis (reviewer request): gradient boosting
on pooled region features. Uses the same colorectal region graphs, the same sample-level subsampling
and the same StratifiedGroupKFold protocol as learning_curve_colorectal.py, so the curves are
directly comparable. Region features are the mean and standard deviation of the cell-node feature
matrix, i.e. the pooled representation a non-graph learner sees."""
import sys, glob, json, argparse
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np, torch
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def pooled_features(g):
    """mean + std over cell nodes -> fixed-length region vector (graph-free view)."""
    x = g['cell'].x if hasattr(g, 'node_types') and 'cell' in g.node_types else g.x
    x = np.asarray(x, dtype=np.float64)
    return np.concatenate([x.mean(0), x.std(0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="colorectal")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="outputs/results/colorectal_lc_gbm.json")
    args = ap.parse_args()

    fs = sorted(glob.glob(f'outputs/hetero_graph/newcancer_{args.tag}/graphs/*.pt'))
    gs = [torch.load(f, weights_only=False) for f in fs]
    X = np.vstack([pooled_features(g) for g in gs])
    y = np.array([int(g.y_graph[0]) for g in gs])
    grp = np.array([g.sample_id for g in gs])
    print(f'{args.tag}: n={len(gs)} feat_dim={X.shape[1]} labels={dict(Counter(y.tolist()))}', flush=True)

    res = {}
    for frac in args.fracs:
        aurocs = []
        for seed in args.seeds:
            sgkf = StratifiedGroupKFold(n_splits=args.folds)
            for fold, (tr, va) in enumerate(sgkf.split(X, y, grp)):
                tr_samples = sorted(set(grp[tr]))
                r = np.random.default_rng(1000 * seed + fold)
                k = max(2, int(round(len(tr_samples) * frac)))
                keep = set(r.choice(tr_samples, size=k, replace=False).tolist())
                tr_sub = np.array([i for i in tr if grp[i] in keep])
                if len(set(y[tr_sub])) < 2:
                    continue
                clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                     random_state=seed)
                clf.fit(X[tr_sub], y[tr_sub])
                prob = clf.predict_proba(X[va])
                present = sorted(set(y[va].tolist()))
                if len(present) < 2:
                    continue
                try:
                    cols = [list(clf.classes_).index(c) for c in present]
                    au = roc_auc_score(y[va], prob[:, cols] / prob[:, cols].sum(1, keepdims=True),
                                       multi_class='ovr', average='macro', labels=present)
                except Exception:
                    continue
                aurocs.append(float(au))
        mu, sd = (float(np.mean(aurocs)), float(np.std(aurocs))) if aurocs else (float('nan'), 0.0)
        res[str(frac)] = dict(auroc=mu, sd=sd, n_runs=len(aurocs),
                              approx_train_regions=int(round(len(gs) * 0.8 * frac)))
        print(f'  GBM frac={frac:<5} (~{res[str(frac)]["approx_train_regions"]:4d} regions) '
              f'AUROC {mu:.3f}+-{sd:.3f} (n={len(aurocs)})', flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, 'w'), indent=2)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
