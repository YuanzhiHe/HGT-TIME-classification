"""Signature-sensitivity analysis (addresses reviewer request on label robustness).
Recomputes the Hot/Excluded/Cold labels under perturbed signature gene sets and quantile
thresholds, and reports how much the labels change relative to the canonical rule. Spot level,
colorectal + lung HEST-1k. No training; pure labeling stability."""
import sys, glob, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import numpy as np, anndata as ad
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
import preprocess_hest1k_visium as P

RNG = np.random.default_rng(0)
INF = list(dict.fromkeys(P.SIGNATURES['immune_infiltration']))
RET = list(dict.fromkeys(P.SIGNATURES['stromal_retention']))


def load(files, maxn=25):
    out = []
    for f in files[:maxn]:
        try:
            a = ad.read_h5ad(f)
        except Exception:
            continue
        X = a.X.toarray() if hasattr(a.X, 'toarray') else np.asarray(a.X)
        X = P.ensure_log1p(X).astype(np.float32)
        gl = defaultdict(list)
        for i, g in enumerate(a.var_names):
            gl[P.normalize_symbol(g)].append(i)
        out.append((X, gl))
    return out


def score(X, gl, genes):
    idx = [gl[P.normalize_symbol(g)][0] for g in genes if P.normalize_symbol(g) in gl]
    if not idx:
        return np.zeros(X.shape[0])
    v = X[:, idx].mean(1)
    return (v - v.mean()) / (v.std() + 1e-9)


def label(inf, ret, qi=0.66, qr=0.60):
    ti, tr = np.quantile(inf, qi), np.quantile(ret, qr)
    return np.where(inf >= ti, 0, np.where(ret >= tr, 1, 2))  # Hot/Excluded/Cold


def main():
    files = sorted(glob.glob('data/hest1k_bowel_lung/st/*.h5ad'))
    data = load(files)
    print(f'samples loaded: {len(data)}', flush=True)
    # canonical labels per sample
    canon = [label(score(X, gl, INF), score(X, gl, RET)) for (X, gl) in data]

    def agreement(pert_labels):
        C = np.concatenate([c for c in canon if len(c)])
        Pl = np.concatenate([p for c, p in zip(canon, pert_labels) if len(c)])
        acc = float((C == Pl).mean())
        try:
            kap = float(cohen_kappa_score(C, Pl))
        except Exception:
            kap = float('nan')
        return acc, kap

    res = {}

    # A) gene dropout (drop fraction of each signature's genes), N reps
    for frac in [0.2, 0.4]:
        accs, kaps = [], []
        for rep in range(20):
            r = np.random.default_rng(rep)
            inf_g = [g for g in INF if r.random() > frac] or INF[:2]
            ret_g = [g for g in RET if r.random() > frac] or RET[:2]
            pert = [label(score(X, gl, inf_g), score(X, gl, ret_g)) for (X, gl) in data]
            a, k = agreement(pert); accs.append(a); kaps.append(k)
        res[f'gene_dropout_{int(frac*100)}pct'] = dict(acc=(float(np.mean(accs)), float(np.std(accs))),
                                                        kappa=(float(np.mean(kaps)), float(np.std(kaps))))

    # B) threshold shift +-0.05 on each quantile
    for dq in [0.05, -0.05]:
        pert = [label(score(X, gl, INF), score(X, gl, RET), qi=0.66+dq, qr=0.60+dq) for (X, gl) in data]
        a, k = agreement(pert)
        res[f'threshold_{"+" if dq>0 else ""}{dq:.2f}'] = dict(acc=(a, 0.0), kappa=(k, 0.0))

    # C) half-panel (use only the first half of each signature) as an alternative gene set
    pert = [label(score(X, gl, INF[:len(INF)//2]), score(X, gl, RET[:len(RET)//2])) for (X, gl) in data]
    a, k = agreement(pert); res['half_panel'] = dict(acc=(a, 0.0), kappa=(k, 0.0))

    print(json.dumps(res, indent=2), flush=True)
    for name, d in res.items():
        print(f'  {name:22s} label-agreement {d["acc"][0]*100:5.1f}% (kappa {d["kappa"][0]:.2f})', flush=True)
    json.dump(res, open('outputs/results/signature_sensitivity.json', 'w'), indent=2)
    print('saved outputs/results/signature_sensitivity.json', flush=True)


if __name__ == '__main__':
    main()
