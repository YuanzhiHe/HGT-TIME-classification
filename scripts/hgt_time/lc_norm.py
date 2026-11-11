"""Learning curve on the section-normalised colorectal graphs.
Same protocol as learning_curve_colorectal.py: sample-level subsampling of the training split,
every model retrained from scratch at each size, StratifiedGroupKFold grouped by sample."""
import sys,glob,json,numpy as np,torch
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(_REPO_ROOT)]
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold

TAG='newcancer_colorectal_secnorm2'
FRACS=[0.1,0.25,0.5,1.0]; SEEDS=[42,123]
# 采用调参后的配置(两类图模型用同一 hidden,公平)
MODELS={
 "HGT-TIME": ("hgt_time", dict(hidden_dim=128,num_layers=3,num_heads=8,dropout=0.1,num_classes=3,
              pheno_dim=4,use_pheno_head=False,use_cell_state_head=False,cell_state_dim=4,use_ranking_heads=False), 3e-4),
 "Homogeneous GCN": ("homogeneous_graph", dict(hidden_dim=128,num_classes=3,num_layers=2,dropout=0.2,conv_type="gcn"), 1e-3),
 "Non-graph MLP": ("non_graph", dict(hidden_dim=128,num_classes=3,num_layers=3,dropout=0.2), 1e-3),
}
LOSS=dict(classification_weight=1.0,phenotype_weight=0.0,region_weight=0.0,
          ranking_weight=0.0,label_smoothing=0.05,class_weights=None)

fs=sorted(glob.glob(f'outputs/hetero_graph/{TAG}/graphs/*.pt'))
gs=[torch.load(f,weights_only=False) for f in fs]
y=np.array([int(g.y_graph[0]) for g in gs]); grp=np.array([g.sample_id for g in gs])
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{TAG}: n={len(gs)} labels={dict(Counter(y.tolist()))} samples={len(set(grp))}',flush=True)

res={}
for name,(fam,mk,lr) in MODELS.items():
    res[name]={}
    RT=dict(batch_size=8,epochs=80,learning_rate=lr,weight_decay=1e-4,patience=12,
            scheduler="cosine",scheduler_kwargs={"T_max":80})
    for frac in FRACS:
        per,sizes=[],[]
        for seed in SEEDS:
            P.set_seed(seed)
            for fold,(tr,va) in enumerate(StratifiedGroupKFold(n_splits=5).split(gs,y,grp)):
                tr_s=sorted(set(grp[tr])); r=np.random.default_rng(1000*seed+fold)
                k=max(2,int(round(len(tr_s)*frac)))
                keep=set(r.choice(tr_s,size=k,replace=False).tolist())
                sub=[i for i in tr if grp[i] in keep]
                if len(set(y[sub]))<2: continue
                ck=Path(f'outputs/results/LCN/{fam}/ck'); ck.mkdir(parents=True,exist_ok=True)
                m=build_model(fam,{**mk},gs[0])
                rr=train_one_fold(model=m,train_loader=DataLoader([gs[i] for i in sub],batch_size=8,shuffle=True),
                    val_loader=DataLoader([gs[i] for i in va],batch_size=8,shuffle=False),
                    config={'model_family':fam,'runtime':RT,'loss':LOSS,'model':mk},
                    device=dev,checkpoint_dir=ck,fold=fold,seed=seed)
                per.append(rr['final_metrics']); sizes.append(len(sub))
        au=[m['macro_auroc'] for m in per if m.get('macro_auroc')==m.get('macro_auroc')]
        mu,sd=(float(np.mean(au)),float(np.std(au))) if au else (float('nan'),0.0)
        res[name][str(frac)]=dict(auroc=mu,sd=sd,n_runs=len(au),approx_train_regions=int(np.mean(sizes)) if sizes else 0)
        print(f'  {name:17s} frac={frac:<5} train~{int(np.mean(sizes)) if sizes else 0:4d}  AUROC {mu:.3f}+-{sd:.3f} (n={len(au)})',flush=True)
        json.dump(res,open('outputs/results/colorectal_learning_curve_secnorm.json','w'),indent=2)
print('\n低标注端(58 区域)对比:')
for n in res: print(f'  {n:17s} {res[n]["0.1"]["auroc"]:.3f}')
g=res['HGT-TIME']['0.1']['auroc']-res['Non-graph MLP']['0.1']['auroc']
print(f'  HGT - MLP = {g:+.3f}   (标准化前为 +0.137)')
open('outputs/results/LCN_DONE','w').write('ok')
