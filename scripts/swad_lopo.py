#!/usr/bin/env python3
"""SWAD (Stochastic Weight Averaging Densely) for cross-patient generalization.

Cross-patient LOPO is a domain-generalization problem: the MLP/graph overfit the training patients
and their errors concentrate under the shift to a new patient. SWAD seeks flat minima by averaging
the weights collected densely within an overfit-aware window around the best validation point, which
is the strongest DomainBed method and requires no architecture change. We compare, from one training
run per fold, the standard best-validation model (ERM) against the SWAD weight average.
"""
from __future__ import annotations
import copy, glob, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE)); sys.path.insert(0, str(CORE / "scripts"))
import train_eval_pipeline as P
from train_eval_pipeline import build_model
from models.losses import HGTTimeLoss

GRAPHS = CORE / "outputs/hetero_graph/hest1k_visium__fusion3/graphs"
KEEP = {0, 2}; SEEDS = [42, 123, 2026]
GMODEL = dict(hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, num_classes=2,
              pheno_dim=4, use_pheno_head=True, use_cell_state_head=False, cell_state_dim=4,
              use_ranking_heads=True)
EPOCHS = 100; LR = 1e-3; WD = 1e-5; TOL = 1.25  # SWAD overfit-aware window ratio


def load():
    gs = []
    for p in sorted(glob.glob(str(GRAPHS / "*.pt"))):
        g = torch.load(p, weights_only=False)
        y = int(g.y_graph[0].item())
        if y not in KEEP: continue
        g.y_graph = torch.tensor([0 if y == 0 else 1])
        g.sample = str(g["cell"].node_id[0]).split("__")[0]
        gs.append(g)
    return gs


def val_loss_of(model, loader, lossfn, device):
    model.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for b in loader:
            b = b.to(device); out = model(b)
            tot += lossfn(out, b)["loss"].item() * out["graph_logits"].size(0); n += out["graph_logits"].size(0)
    return tot / max(n, 1)


@torch.no_grad()
def probs(model, graphs, device):
    model.eval(); out = []
    for b in DataLoader(graphs, batch_size=8):
        out.append(torch.softmax(model(b.to(device))["graph_logits"], -1).cpu().numpy())
    return np.concatenate(out, 0)[:, 1]


def avg_states(states):
    avg = copy.deepcopy(states[0])
    for k in avg:
        if avg[k].is_floating_point():
            avg[k] = torch.stack([s[k].float() for s in states], 0).mean(0)
    return avg


def train_fold(graphs, tr, te, seed, device):
    P.set_seed(seed)
    model = build_model("hgt_time", GMODEL, graphs[0]).to(device)
    lossfn = HGTTimeLoss(classification_weight=1.0, phenotype_weight=0.3, region_weight=0.0,
                         ranking_weight=0.1, label_smoothing=0.05, class_weights=None).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    tl = DataLoader([graphs[i] for i in tr], batch_size=4, shuffle=True)
    vl = DataLoader([graphs[i] for i in te], batch_size=4, shuffle=False)
    snaps, vlosses, best_state, best_v = [], [], None, 1e9
    for ep in range(EPOCHS):
        model.train()
        for b in tl:
            b = b.to(device); opt.zero_grad()
            loss = lossfn(model(b), b)["loss"]
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        v = val_loss_of(model, vl, lossfn, device)
        snaps.append({k: x.detach().cpu().clone() for k, x in model.state_dict().items()}); vlosses.append(v)
        if v < best_v: best_v, best_state = v, snaps[-1]
    # SWAD: average snapshots inside the overfit-aware window (val loss <= best * TOL)
    thr = best_v * TOL
    win = [s for s, vv in zip(snaps, vlosses) if vv <= thr]
    swad_state = avg_states(win) if len(win) >= 2 else best_state
    return best_state, swad_state


def evalrun(which, graphs, groups, y, device):
    N = len(graphs); Pp = np.zeros(N)
    for held in sorted(set(groups.tolist())):
        tr = np.where(groups != held)[0]; te = np.where(groups == held)[0]
        acc = np.zeros(len(te))
        for seed in SEEDS:
            best_state, swad_state = train_fold(graphs, tr, te, seed, device)
            m = build_model("hgt_time", GMODEL, graphs[0]).to(device)
            m.load_state_dict(best_state if which == "erm" else swad_state)
            acc += probs(m, [graphs[i] for i in te], device)
        Pp[te] = acc / len(SEEDS)
    yp = (Pp >= 0.5).astype(int)
    return roc_auc_score(y, Pp), f1_score(y, yp, average="macro"), balanced_accuracy_score(y, yp)


def main():
    graphs = load(); groups = np.array([g.sample for g in graphs]); y = np.array([int(g.y_graph[0]) for g in graphs])
    print("samples:", dict(Counter(groups.tolist())), "n=", len(graphs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # single pass: train once per fold/seed, evaluate BOTH ERM and SWAD from the same run
    N = len(graphs); Perm = np.zeros(N); Pswad = np.zeros(N)
    for held in sorted(set(groups.tolist())):
        tr = np.where(groups != held)[0]; te = np.where(groups == held)[0]
        ae = np.zeros(len(te)); asw = np.zeros(len(te))
        for seed in SEEDS:
            best_state, swad_state = train_fold(graphs, tr, te, seed, device)
            me = build_model("hgt_time", GMODEL, graphs[0]).to(device); me.load_state_dict(best_state)
            ms = build_model("hgt_time", GMODEL, graphs[0]).to(device); ms.load_state_dict(swad_state)
            ae += probs(me, [graphs[i] for i in te], device); asw += probs(ms, [graphs[i] for i in te], device)
        Perm[te] = ae / len(SEEDS); Pswad[te] = asw / len(SEEDS)
    for name, Pp in [("ERM (best-val)", Perm), ("SWAD", Pswad)]:
        yp = (Pp >= 0.5).astype(int)
        print(f"{name:16s} AUROC {roc_auc_score(y,Pp):.3f}  F1 {f1_score(y,yp,average='macro'):.3f}  BAcc {balanced_accuracy_score(y,yp):.3f}")


if __name__ == "__main__":
    main()
