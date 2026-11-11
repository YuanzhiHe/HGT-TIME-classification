"""Graph models at the combination that matters: normalised features, 8-gene cell budget,
and a low label budget. Baselines for this cell are already measured (cell-only 0.563,
information-matched 0.707 at 58 regions; 0.740 / 0.871 at 575)."""
import sys,glob,json,numpy as np,torch
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(_REPO_ROOT)]
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold
LOSS=dict(classification_weight=1.0,phenotype_weight=0.0,region_weight=0.0,
          ranking_weight=0.0,label_smoothing=0.05,class_weights=None)
M={"HGT-TIME":("hgt_time",dict(hidden_dim=128,num_layers=3,num_heads=8,dropout=0.1,num_classes=3,
      pheno_dim=4,use_pheno_head=False,use_cell_state_head=False,cell_state_dim=4,use_ranking_heads=False),3e-4),
   "Homogeneous GCN":("homogeneous_graph",dict(hidden_dim=128,num_classes=3,num_layers=2,dropout=0.2,conv_type="gcn"),1e-3)}
fs=sorted(glob.glob('outputs/hetero_graph/colo_sn_hvg8/graphs/*.pt'))
gs=[torch.load(f,weights_only=False) for f in fs]
y=np.array([int(g.y_graph[0]) for g in gs]); grp=np.array([g.sample_id for g in gs])
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'colo_sn_hvg8: n={len(gs)} samples={len(set(grp))}',flush=True)
res={}
for frac in (0.1,1.0):
    res[frac]={}
    for name,(fam,mk,lr) in M.items():
        RT=dict(batch_size=8,epochs=80,learning_rate=lr,weight_decay=1e-4,patience=12,
                scheduler="cosine",scheduler_kwargs={"T_max":80})
        au=[]
        for seed in (42,123):
            P.set_seed(seed); rng=np.random.default_rng(seed)
            for fold,(tr,va) in enumerate(StratifiedGroupKFold(n_splits=5).split(gs,y,grp)):
                ts=sorted(set(grp[tr])); k=max(2,int(round(len(ts)*frac)))
                keep=set(rng.choice(ts,size=k,replace=False).tolist())
                sub=[i for i in tr if grp[i] in keep]
                if len(set(y[sub]))<2: continue
                ck=Path(f'outputs/results/DEC/{fam}'); ck.mkdir(parents=True,exist_ok=True)
                m=build_model(fam,{**mk},gs[0])
                r=train_one_fold(model=m,train_loader=DataLoader([gs[i] for i in sub],batch_size=8,shuffle=True),
                    val_loader=DataLoader([gs[i] for i in va],batch_size=8,shuffle=False),
                    config={'model_family':fam,'runtime':RT,'loss':LOSS,'model':mk},
                    device=dev,checkpoint_dir=ck,fold=fold,seed=seed)
                v=r['final_metrics'].get('macro_auroc')
                if v==v: au.append(v)
        res[frac][name]=(float(np.mean(au)),float(np.std(au)),len(au))
        n_reg = 58 if frac==0.1 else 575
        print(f'  {n_reg:3d} 区域  {name:17s} {np.mean(au):.3f} +- {np.std(au):.3f} (n={len(au)})',flush=True)
        json.dump({str(k):v for k,v in res.items()},open('outputs/results/decisive_hvg8.json','w'),indent=2)
BASE={0.1:{'cell':0.563,'info':0.707},1.0:{'cell':0.740,'info':0.871}}
print('\n对比信息对齐基线:')
for frac in (0.1,1.0):
    n_reg=58 if frac==0.1 else 575
    h=res[frac]['HGT-TIME'][0]
    print(f'  {n_reg:3d} 区域: HGT {h:.3f} | 信息对齐 {BASE[frac]["info"]:.3f} -> {h-BASE[frac]["info"]:+.3f}'
          f' | 仅细胞 {BASE[frac]["cell"]:.3f} -> {h-BASE[frac]["cell"]:+.3f}')
open('outputs/results/DEC_DONE','w').write('ok')
