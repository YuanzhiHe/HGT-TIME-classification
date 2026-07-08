#!/usr/bin/env python3
"""Deconvolution-free region-level heterogeneous graph builder for new cancers (colorectal/lung).

Cell features: top-K highly variable log-expression + normalized spatial coords (no CIBERSORTx).
Gene nodes: expression stats + STRING degree + KEGG membership + immune flag.
Pathway nodes: KEGG pathway stats. Edges: spatial kNN, cell->gene (top-k expressed), gene-gene
(STRING), gene->pathway (KEGG). Labels: pan-cancer signature rule (same as HEST breast).
Model auto-infers input dims, so exact widths need not match the breast graphs.
"""
from __future__ import annotations
import glob, sys
from collections import defaultdict
from pathlib import Path

import numpy as np, pandas as pd, torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "scripts"))
import preprocess_hest1k_visium as P

PRIOR = CORE / "outputs/priors/string_kegg_v1"
IMMUNE = set(P.SIGNATURES['immune_infiltration'] + P.SIGNATURES['immune_activation'] +
             P.SIGNATURES['immune_penetration'])
LBL = {'Hot': 0, 'Excluded': 1, 'Cold': 2}


def load_priors():
    gg = pd.read_csv(PRIOR / "string_kegg_v1__step-05_gene_gene_edges.tsv.gz", sep="\t")
    gg = gg[gg["retained_for_graph"] == True][["source_gene_symbol", "target_gene_symbol", "edge_weight"]]
    gp = pd.read_csv(PRIOR / "string_kegg_v1__step-06_gene_pathway_edges.tsv.gz", sep="\t")
    gp = gp[["gene_symbol", "pathway_id", "pathway_top_class"]]
    deg = defaultdict(int)
    for a, b in zip(gg.source_gene_symbol, gg.target_gene_symbol):
        deg[a] += 1; deg[b] += 1
    kegg_cnt = gp.groupby("gene_symbol").size().to_dict()
    return gg, gp, deg, kegg_cnt


def build_region(spots_idx, expr, coords, var, gene_deg, kegg_cnt, gp, hvg_idx, y):
    n = len(spots_idx)
    sub = expr[spots_idx]                                  # cells x genes
    xy = coords[spots_idx].astype(np.float32)
    xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-6)
    cell_x = np.concatenate([sub[:, hvg_idx], xy], axis=1).astype(np.float32)   # K HVG + 2 coords
    # gene nodes: expressed in region AND in prior universe
    reg_mean = sub.mean(0); reg_var = sub.var(0)
    present = [j for j in np.where(reg_mean > 0.1)[0] if var[j] in gene_deg or var[j] in kegg_cnt]
    present = present[:200]
    gsym = [var[j] for j in present]; gidx = {s: i for i, s in enumerate(gsym)}
    gene_x = np.array([[reg_mean[j], reg_var[j], np.log1p(gene_deg.get(var[j], 0)),
                        np.log1p(kegg_cnt.get(var[j], 0)), 1.0 if var[j] in IMMUNE else 0.0]
                       for j in present], dtype=np.float32)
    # pathway nodes
    gp_sub = gp[gp.gene_symbol.isin(gsym)]
    paths = gp_sub.pathway_id.value_counts()
    paths = paths[paths >= 2].index.tolist()[:60]
    pidx = {p: i for i, p in enumerate(paths)}
    path_x = []
    for p in paths:
        members = gp_sub[gp_sub.pathway_id == p].gene_symbol.tolist()
        mi = [gidx[m] for m in members if m in gidx]
        path_x.append([len(mi), float(np.mean([reg_mean[present[k]] for k in mi])) if mi else 0.0])
    path_x = np.array(path_x, dtype=np.float32) if paths else np.zeros((0, 2), np.float32)

    d = HeteroData()
    d['cell'].x = torch.tensor(cell_x)
    d['gene'].x = torch.tensor(gene_x) if len(gsym) else torch.zeros((1, 5))
    d['pathway'].x = torch.tensor(path_x) if len(paths) else torch.zeros((1, 2))
    # spatial kNN edges
    k = min(6, n - 1)
    if k >= 1:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
        _, idx = nn.kneighbors(xy)
        src = np.repeat(np.arange(n), k); dst = idx[:, 1:].reshape(-1)
        d['cell', 'spatial', 'cell'].edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    else:
        d['cell', 'spatial', 'cell'].edge_index = torch.zeros((2, 0), dtype=torch.long)
    # cell -> gene (each cell to its top-8 expressed present genes)
    ce, ge = [], []
    if len(gsym):
        gcols = np.array(present)
        for ci in range(n):
            vals = sub[ci, gcols]
            top = np.argsort(vals)[-8:]
            for t in top:
                if vals[t] > 0: ce.append(ci); ge.append(int(t))
    ei = torch.tensor(np.vstack([ce, ge]), dtype=torch.long) if ce else torch.zeros((2, 0), dtype=torch.long)
    d['cell', 'expresses', 'gene'].edge_index = ei
    d['gene', 'rev_expresses', 'cell'].edge_index = ei[[1, 0]] if ei.numel() else torch.zeros((2, 0), dtype=torch.long)
    # gene - gene (STRING within present)
    gs = set(gsym)
    gg_sub = [(gidx[a], gidx[b]) for a, b in zip(GG.source_gene_symbol, GG.target_gene_symbol) if a in gs and b in gs]
    if gg_sub:
        arr = np.array(gg_sub).T; arr = np.hstack([arr, arr[[1, 0]]])
        d['gene', 'interacts', 'gene'].edge_index = torch.tensor(arr, dtype=torch.long)
    else:
        d['gene', 'interacts', 'gene'].edge_index = torch.zeros((2, 0), dtype=torch.long)
    # gene -> pathway
    pe_g, pe_p = [], []
    for _, r in gp_sub.iterrows():
        if r.gene_symbol in gidx and r.pathway_id in pidx:
            pe_g.append(gidx[r.gene_symbol]); pe_p.append(pidx[r.pathway_id])
    ep = torch.tensor(np.vstack([pe_g, pe_p]), dtype=torch.long) if pe_g else torch.zeros((2, 0), dtype=torch.long)
    d['gene', 'in_pathway', 'pathway'].edge_index = ep
    d['pathway', 'rev_in_pathway', 'gene'].edge_index = ep[[1, 0]] if ep.numel() else torch.zeros((2, 0), dtype=torch.long)
    d.y_graph = torch.tensor([y]); d.label_mask = torch.tensor([True])
    return d


def process_sample(sid, hvg_k=48):
    import anndata as ad
    a = ad.read_h5ad(f'data/hest1k_bowel_lung/st/{sid}.h5ad')
    expr = a.X.toarray() if hasattr(a.X, 'toarray') else np.asarray(a.X)
    expr = P.ensure_log1p(expr).astype(np.float32)
    var = list(a.var_names)
    gl = defaultdict(list)
    for i, g in enumerate(var): gl[P.normalize_symbol(g)].append(i)
    S = lambda k: P.mean_signature(expr, gl, P.SIGNATURES[k])
    lab, unc = P.assign_time_labels(S('immune_infiltration'), S('immune_penetration'),
                                    S('stromal_retention'), S('immune_activation'))
    coords = a.obsm['spatial'] if 'spatial' in a.obsm else a.obs[['array_row', 'array_col']].values
    coords = np.asarray(coords, dtype=np.float32)
    hvg_idx = np.argsort(expr.var(0))[-hvg_k:]
    # adaptive grid so each region has ~30 spots
    gr = max(4, int(np.sqrt(len(expr) / 30)))
    r = pd.qcut(coords[:, 0], gr, labels=False, duplicates='drop')
    c = pd.qcut(coords[:, 1], gr, labels=False, duplicates='drop')
    graphs = []
    for reg in set(zip(r, c)):
        idx = np.where((r == reg[0]) & (c == reg[1]) & (~unc))[0]
        if len(idx) < 25: continue
        vc = pd.Series(lab[idx]).value_counts()
        y = LBL[vc.index[0]]
        g = build_region(idx, expr, coords, var, GENE_DEG, KEGG_CNT, GP, hvg_idx, y)
        g.sample_id = sid; graphs.append(g)
    return graphs


GG, GP, GENE_DEG, KEGG_CNT = load_priors()

if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else 'MISC57'
    gs = process_sample(sid)
    print(f"{sid}: {len(gs)} region graphs")
    if gs:
        g = gs[0]
        print("  cell.x", tuple(g['cell'].x.shape), "gene.x", tuple(g['gene'].x.shape),
              "pathway.x", tuple(g['pathway'].x.shape))
        print("  edges:", {str(e): g[e].edge_index.shape[1] for e in g.edge_types})
        print("  labels:", [int(x.y_graph[0]) for x in gs])
