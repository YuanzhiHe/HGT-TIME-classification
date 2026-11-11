"""Hyperparameter search on the section-normalised colorectal graphs.
Search is run on a fixed inner split of the TRAINING samples only; the outer 5-fold grouped
evaluation is untouched, so no configuration is chosen using the folds it is scored on."""
import sys,glob,json,itertools,numpy as np,torch
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(_REPO_ROOT)]
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold

fs=sorted(glob.glob('outputs/hetero_graph/newcancer_colorectal_secnorm/graphs/*.pt'))
gs=[torch.load(f,weights_only=False) for f in fs]
y=np.array([int(g.y_graph[0]) for g in gs]); grp=np.array([g.sample_id for g in gs])
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定一个"调参用"的外层划分:只在其 train 部分内部再切,绝不碰其余折
P.set_seed(2024)
outer=list(StratifiedGroupKFold(n_splits=5).split(gs,y,grp))
tr_idx,_=outer[0]
sub_grp=grp[tr_idx]
inner=list(StratifiedGroupKFold(n_splits=3).split([gs[i] for i in tr_idx],y[tr_idx],sub_grp))

GRID=[dict(hidden_dim=h,num_layers=l,num_heads=nh,dropout=d,learning_rate=lr)
      for h in (128,256) for l in (2,3) for nh in (4,8) for d in (0.1,0.2) for lr in (1e-3,3e-4)]
print(f'{len(GRID)} 个配置,内层 3 折(仅用外层 fold0 的训练样本)',flush=True)

res=[]
for ci,cfg in enumerate(GRID):
    RUNTIME=dict(batch_size=8,epochs=60,learning_rate=cfg['learning_rate'],weight_decay=1e-5,
                 patience=10,scheduler="cosine",scheduler_kwargs={"T_max":60})
    LOSS=dict(classification_weight=1.0,phenotype_weight=0.0,region_weight=0.0,
              ranking_weight=0.0,label_smoothing=0.05,class_weights=None)
    HGT=dict(hidden_dim=cfg['hidden_dim'],num_layers=cfg['num_layers'],num_heads=cfg['num_heads'],
             dropout=cfg['dropout'],num_classes=3,pheno_dim=4,use_pheno_head=False,
             use_cell_state_head=False,cell_state_dim=4,use_ranking_heads=False)
    au=[]
    for fold,(a,b) in enumerate(inner):
        P.set_seed(42)
        A=[tr_idx[i] for i in a]; B=[tr_idx[i] for i in b]
        if len(set(y[A]))<3 or len(set(y[B]))<3: continue
        ck=Path('outputs/results/TUNE-secnorm/ck'); ck.mkdir(parents=True,exist_ok=True)
        m=build_model('hgt_time',{**HGT},gs[0])
        r=train_one_fold(model=m,
            train_loader=DataLoader([gs[i] for i in A],batch_size=8,shuffle=True),
            val_loader=DataLoader([gs[i] for i in B],batch_size=8,shuffle=False),
            config={'model_family':'hgt_time','runtime':RUNTIME,'loss':LOSS,'model':HGT},
            device=dev,checkpoint_dir=ck,fold=fold,seed=42)
        v=r['final_metrics'].get('macro_auroc')
        if v==v: au.append(v)
    s=float(np.mean(au)) if au else 0.0
    res.append((s,cfg))
    print(f'  [{ci+1:2d}/{len(GRID)}] {cfg} -> {s:.3f}',flush=True)
    json.dump([(a,b) for a,b in sorted(res,reverse=True)],open('outputs/results/tune_secnorm.json','w'),indent=1)
res.sort(reverse=True)
print('\n最佳 3 个:')
for s,c in res[:3]: print(f'  {s:.3f}  {c}')
