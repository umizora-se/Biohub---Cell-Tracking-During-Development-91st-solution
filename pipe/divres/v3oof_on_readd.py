"""Honest comparison: v3 classifier trained OOF on base-graph candidates (folds != k), applied to readd-graph table rows of fold k."""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, lightgbm as lgb
from pathlib import Path
import divclf, divclf3
from divclf_oof import LGB_PARAMS
C3=pd.read_parquet('divres/oof3/cands_oof3.parquet'); FT=pd.read_parquet('divres/oof3/forks_oof3.parquet')
base=pd.concat([C3,FT],ignore_index=True)
T=pd.read_parquet('divres/oofR/table_oof.parquet'); sub=pd.read_csv('divres/readd98.csv')
folds={k:set(Path(f'../det/folds/fold{k}.txt').read_text().split()) for k in range(5)}
T['oof_v3']=0.0
for seed in range(3):
    for k in range(5):
        trm=(~C3.ds.isin(folds[k])&(C3.label>=0)).values
        m=lgb.LGBMClassifier(**{**LGB_PARAMS,'random_state':seed}).fit(divclf3.design3(C3[trm]),C3.label[trm])
        tem=T.ds.isin(folds[k]).values; T.loc[tem,'oof_v3']+=m.predict_proba(divclf3.design3(T[tem]))[:,1]/3
T['oof_mix']=(T.oof_v3+T.oof)/2
for col in ['oof_v3','oof_mix']:
  for thr,pthr in [(0.4,0.05),(0.3,0.05)]:
    parts=[]
    for name,grp in sub.groupby('dataset',sort=False):
        nodes=grp[grp.row_type=='node']; edges=grp[grp.row_type=='edge'][['source_id','target_id']].astype(int)
        f=T[(T.ds==name)&(T.kind=='fork')&(T[col]<pthr)]; rm=set(zip(f.p,f.o))
        edges=edges[[(s,t) not in rm for s,t in zip(edges.source_id,edges.target_id)]]
        a=T[(T.ds==name)&(T.kind=='add')]; new=divclf.select(a,a[col].values,thr) if len(a) else pd.DataFrame(columns=['source_id','target_id'])
        parts.append(nodes.drop(columns=['id'])); parts.append(pd.concat([edges,new]).assign(dataset=name,row_type='edge',node_id=-1,t=-1,z=-1,y=-1,x=-1))
    res=pd.concat(parts,ignore_index=True)[['dataset','row_type','node_id','t','z','y','x','source_id','target_id']]; res.insert(0,'id',np.arange(len(res)))
    res.to_csv(f'divres/oofR/sub_{col}_t{thr}_p{pthr}.csv',index=False); print('wrote',col,thr,pthr,flush=True)
