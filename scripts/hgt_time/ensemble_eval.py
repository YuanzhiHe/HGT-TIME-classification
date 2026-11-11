"""Seed-ensembled evaluation under the study's grouped cross-validation protocol.

The default protocol trains one model per (fold, seed) and averages the resulting metrics. This
script instead averages the predicted class probabilities across seeds within each fold, and scores
the averaged prediction once per fold. Averaging predictions rather than metrics is the standard
way to report an ensemble, and it changes nothing about the split: a sample never appears in both
the training and the evaluation side of a fold, and seeds differ only in initialisation and batch
order, so no additional data is seen.

Both numbers are printed, so the gain attributable to ensembling is explicit.
"""
from __future__ import annotations
import sys, glob, json, argparse

import numpy as np, torch
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(_REPO_ROOT)]
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold, compute_metrics

RUNTIME = dict(batch_size=8, epochs=80, learning_rate=1e-3, weight_decay=1e-5, patience=12,
               scheduler="cosine", scheduler_kwargs={"T_max": 80})
LOSS = dict(classification_weight=1.0, phenotype_weight=0.0, region_weight=0.0,
            ranking_weight=0.0, label_smoothing=0.05, class_weights=None)
HGT = dict(hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, num_classes=3, pheno_dim=4,
           use_pheno_head=False, use_cell_state_head=False, cell_state_dim=4, use_ranking_heads=False)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        logits = out['graph_logits'] if isinstance(out, dict) else out
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        y = batch.y_graph.view(-1).long().cpu().numpy() if hasattr(batch, 'y_graph') else None
        ys.append(y)
    return np.concatenate(probs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='newcancer_colorectal_secnorm')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7, 2024, 31])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--ls', type=float, default=0.05)
    ap.add_argument('--out', default='outputs/results/ensemble_eval.json')
    args = ap.parse_args()

    HGT.update(hidden_dim=args.hidden, num_layers=args.layers, num_heads=args.heads,
               dropout=args.dropout)
    RUNTIME.update(learning_rate=args.lr, weight_decay=args.wd, epochs=args.epochs,
                   patience=max(12, args.epochs // 6),
                   scheduler_kwargs={'T_max': args.epochs})
    LOSS.update(label_smoothing=args.ls)

    fs = sorted(glob.glob(f'outputs/hetero_graph/{args.tag}/graphs/*.pt'))
    gs = [torch.load(f, weights_only=False) for f in fs]
    y = np.array([int(g.y_graph[0]) for g in gs])
    grp = np.array([g.sample_id for g in gs])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{args.tag}: n={len(gs)} samples={len(set(grp))} '
          f'| hidden {args.hidden} layers {args.layers} heads {args.heads} '
          f'dropout {args.dropout} lr {args.lr}', flush=True)

    P.set_seed(2024)
    splits = list(StratifiedGroupKFold(n_splits=args.folds).split(gs, y, grp))

    single, ens = [], []
    for fold, (tr, va) in enumerate(splits):
        va_loader = DataLoader([gs[i] for i in va], batch_size=8, shuffle=False)
        acc = None
        for seed in args.seeds:
            P.set_seed(seed)
            ck = Path(f'outputs/results/ENS/{args.tag}/ck'); ck.mkdir(parents=True, exist_ok=True)
            m = build_model('hgt_time', {**HGT}, gs[0])
            train_one_fold(model=m,
                           train_loader=DataLoader([gs[i] for i in tr], batch_size=8, shuffle=True),
                           val_loader=va_loader,
                           config={'model_family': 'hgt_time', 'runtime': RUNTIME,
                                   'loss': LOSS, 'model': HGT},
                           device=dev, checkpoint_dir=ck, fold=fold, seed=seed)
            pr, yt = predict(m, va_loader, dev)
            single.append(compute_metrics(yt, pr).get('macro_auroc'))
            acc = pr if acc is None else acc + pr
        em = compute_metrics(yt, acc / len(args.seeds))
        ens.append(em)
        print(f'  fold {fold}: single mean {np.nanmean(single[-len(args.seeds):]):.3f} '
              f'| ensembled {em.get("macro_auroc"):.3f}', flush=True)

    s = [x for x in single if x == x]
    e = [m['macro_auroc'] for m in ens if m.get('macro_auroc') == m.get('macro_auroc')]
    f1 = [m.get('macro_f1') for m in ens if m.get('macro_f1') == m.get('macro_f1')]
    ba = [m.get('balanced_accuracy') for m in ens if m.get('balanced_accuracy') == m.get('balanced_accuracy')]
    print(f'\n  single-model mean  macro-AUROC {np.mean(s):.3f} +- {np.std(s):.3f} (n={len(s)})')
    print(f'  seed-ensembled     macro-AUROC {np.mean(e):.3f} +- {np.std(e):.3f} (n={len(e)})'
          f'  macro-F1 {np.mean(f1):.3f}  bal.acc {np.mean(ba):.3f}')
    print(f'  ensembling gain    {np.mean(e) - np.mean(s):+.3f}')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(tag=args.tag, config=HGT, lr=args.lr, seeds=args.seeds,
                   single_mean=float(np.mean(s)), single_sd=float(np.std(s)),
                   ensemble_mean=float(np.mean(e)), ensemble_sd=float(np.std(e)),
                   ensemble_macro_f1=float(np.mean(f1)), ensemble_bal_acc=float(np.mean(ba))),
              open(args.out, 'w'), indent=2)
    print('saved', args.out)


if __name__ == '__main__':
    main()
