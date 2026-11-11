"""Does the knowledge graph compensate when the cell-feature budget is narrow?
Three models on identical graphs/labels/splits at 8, 16, 32 and 48 cell features."""
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
   "Homogeneous GCN":("homogeneous_graph",dict(hidden_dim=128,num_classes=3,num_layers=2,dropout=0.2,conv_type="gcn"),1e-3),
   "Non-graph MLP":("non_graph",dict(hidden_dim=128,num_classes=3,num_layers=3,dropout=0.2),1e-3)}
res={}
for tag,K in [('bcc_sn_hvg8',8),('bcc_sn_hvg48',48)]:
    fs=sorted(glob.glob(f'outputs/hetero_graph/{tag}/graphs/*.pt'))
    gs=[torch.load(f,weights_only=False) for f in fs]
    y=np.array([int(g.y_graph[0]) for g in gs]); grp=np.array([g.sample_id for g in gs])
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res[K]={}
    for name,(fam,mk,lr) in M.items():
        RT=dict(batch_size=32,epochs=50,learning_rate=lr,weight_decay=1e-4,patience=8,
                scheduler="cosine",scheduler_kwargs={"T_max":50})
        au=[]
        for seed in (42,):
            P.set_seed(seed)
            for fold,(tr,va) in enumerate(StratifiedGroupKFold(n_splits=5).split(gs,y,grp)):
                ck=Path(f'outputs/results/HVGS_BCC/{tag}/{fam}'); ck.mkdir(parents=True,exist_ok=True)
                m=build_model(fam,{**mk},gs[0])
                r=train_one_fold(model=m,train_loader=DataLoader([gs[i] for i in tr],batch_size=32,shuffle=True),
                    val_loader=DataLoader([gs[i] for i in va],batch_size=32,shuffle=False),
                    config={'model_family':fam,'runtime':RT,'loss':LOSS,'model':mk},
                    device=dev,checkpoint_dir=ck,fold=fold,seed=seed)
                v=r['final_metrics'].get('macro_auroc')
                if v==v: au.append(v)
        res[K][name]=dict(auroc=float(np.mean(au)),sd=float(np.std(au)),n=len(au))
        print(f'  {K:3d} HVG  {name:17s} {np.mean(au):.3f} +- {np.std(au):.3f}',flush=True)
        json.dump(res,open('outputs/results/hvg_budget_sweep_BCC.json','w'),indent=2)
    d=res[K]["HGT-TIME"]["auroc"]-res[K]["Non-graph MLP"]["auroc"]
    print(f'       -> HGT - MLP = {d:+.3f}\n',flush=True)
open('outputs/results/HVGS_BCC_DONE','w').write('ok')
