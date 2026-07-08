import sys, glob; sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, torch
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold
GMODEL=dict(hidden_dim=128,num_layers=3,num_heads=4,dropout=0.2,num_classes=3,pheno_dim=4,
            use_pheno_head=False,use_cell_state_head=False,cell_state_dim=4,use_ranking_heads=False)
RUNTIME=dict(batch_size=8,epochs=80,learning_rate=1e-3,weight_decay=1e-5,patience=12,scheduler="cosine",scheduler_kwargs={"T_max":80})
LOSS=dict(classification_weight=1.0,phenotype_weight=0.0,region_weight=0.0,ranking_weight=0.0,label_smoothing=0.05,class_weights=None)
def run(tag,seeds=(42,123,2026)):
    fs=sorted(glob.glob(f'outputs/hetero_graph/newcancer_{tag}/graphs/*.pt'))
    gs=[torch.load(f,weights_only=False) for f in fs]
    y=np.array([int(g.y_graph[0]) for g in gs]); grp=np.array([g.sample_id for g in gs])
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=Path(f'outputs/results/EXP-NEW-{tag}/ck'); ck.mkdir(parents=True,exist_ok=True)
    from collections import Counter; print(f'{tag}: n={len(gs)} labels={dict(Counter(y.tolist()))} samples={len(set(grp))}',flush=True)
    yp=np.zeros(len(gs)); pr=np.zeros((len(gs),3))
    sgkf=StratifiedGroupKFold(n_splits=5)
    per=[]
    for seed in seeds:
        P.set_seed(seed)
        for fold,(tr,va) in enumerate(sgkf.split(gs,y,grp)):
            m=build_model('hgt_time',GMODEL,gs[0])
            cfg={'model_family':'hgt_time','runtime':RUNTIME,'loss':LOSS,'model':GMODEL}
            r=train_one_fold(model=m,train_loader=DataLoader([gs[i] for i in tr],batch_size=8,shuffle=True),
                             val_loader=DataLoader([gs[i] for i in va],batch_size=8,shuffle=False),
                             config=cfg,device=dev,checkpoint_dir=ck,fold=fold,seed=seed)
            per.append(r['final_metrics'])
    def agg(k): v=[m[k] for m in per if k in m]; return (np.mean(v),np.std(v)) if v else (float('nan'),0)
    for k in ['macro_auroc','macro_f1','balanced_accuracy']:
        mu,sd=agg(k); print(f'  {tag} {k}: {mu:.3f} +- {sd:.2f}',flush=True)
if __name__=='__main__':
    run(sys.argv[1] if len(sys.argv)>1 else 'colorectal')
