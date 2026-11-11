"""Second-stage search on the fully section-normalised graphs.
Scored on an inner 3-fold split of the OUTER fold-0 training samples only; the five outer folds
used for the reported number are never used to pick a configuration."""
import sys,glob,json,numpy as np,torch
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(_REPO_ROOT)]
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold

fs=sorted(glob.glob('outputs/hetero_graph/newcancer_colorectal_secnorm2/graphs/*.pt'))
gs=[torch.load(f,weights_only=False) for f in fs]
y=np.array([int(g.y_graph[0]) for g in gs]); grp=np.array([g.sample_id for g in gs])
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
P.set_seed(2024)
tr_idx,_=list(StratifiedGroupKFold(n_splits=5).split(gs,y,grp))[0]
inner=list(StratifiedGroupKFold(n_splits=3).split([gs[i] for i in tr_idx],y[tr_idx],grp[tr_idx]))

# 围绕上一轮最优 (128/3/8/0.1/3e-4) 展开:batch、weight decay、epochs、label smoothing
GRID=[]
for bs in (8,16):
    for wd in (1e-5,1e-4):
        for ep in (80,140):
            for ls in (0.0,0.05):
                GRID.append(dict(batch_size=bs,weight_decay=wd,epochs=ep,label_smoothing=ls))
print(f'{len(GRID)} 个配置',flush=True)
HGT=dict(hidden_dim=128,num_layers=3,num_heads=8,dropout=0.1,num_classes=3,pheno_dim=4,
         use_pheno_head=False,use_cell_state_head=False,cell_state_dim=4,use_ranking_heads=False)
res=[]
for ci,c in enumerate(GRID):
    RT=dict(batch_size=c['batch_size'],epochs=c['epochs'],learning_rate=3e-4,
            weight_decay=c['weight_decay'],patience=max(12,c['epochs']//6),
            scheduler="cosine",scheduler_kwargs={"T_max":c['epochs']})
    LS=dict(classification_weight=1.0,phenotype_weight=0.0,region_weight=0.0,
            ranking_weight=0.0,label_smoothing=c['label_smoothing'],class_weights=None)
    au=[]
    for fold,(a,b) in enumerate(inner):
        P.set_seed(42)
        A=[tr_idx[i] for i in a]; B=[tr_idx[i] for i in b]
        if len(set(y[A]))<3 or len(set(y[B]))<3: continue
        ck=Path('outputs/results/TUNE2/ck'); ck.mkdir(parents=True,exist_ok=True)
        m=build_model('hgt_time',{**HGT},gs[0])
        r=train_one_fold(model=m,train_loader=DataLoader([gs[i] for i in A],batch_size=c['batch_size'],shuffle=True),
            val_loader=DataLoader([gs[i] for i in B],batch_size=c['batch_size'],shuffle=False),
            config={'model_family':'hgt_time','runtime':RT,'loss':LS,'model':HGT},
            device=dev,checkpoint_dir=ck,fold=fold,seed=42)
        v=r['final_metrics'].get('macro_auroc')
        if v==v: au.append(v)
    s=float(np.mean(au)) if au else 0.0
    res.append((s,c)); print(f'  [{ci+1:2d}/{len(GRID)}] {c} -> {s:.3f}',flush=True)
    json.dump(sorted(res,key=lambda t:-t[0]),open('outputs/results/tune2_secnorm2.json','w'),indent=1)
res.sort(key=lambda t:-t[0])
open('outputs/results/tune2_DONE','w').write('ok')
print('\n最佳 3:'); [print(f'  {s:.3f}  {c}') for s,c in res[:3]]
