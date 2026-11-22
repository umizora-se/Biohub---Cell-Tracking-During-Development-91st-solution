import sys; sys.path.insert(0,'.'); sys.path.insert(0,'../eda')
import pandas as pd, numpy as np, zarr
from concurrent.futures import ProcessPoolExecutor
import divclf2
from io_utils import DATA
C=pd.read_parquet('divres/oof/cands_all.parquet')
sub=pd.read_csv('exps/r_base98/kaggle/working/submission.csv')
def run(name):
    c=C[C.ds==name].copy(); g=sub[sub.dataset==name]
    nodes=g[g.row_type=='node'][['node_id','t','z','y','x']]; edges=g[g.row_type=='edge'][['source_id','target_id']].astype(int)
    zp=DATA/'train'/f'{name}.zarr'; q=dict(zarr.open_group(str(zp),mode='r').attrs)['image_statistics']['quantiles']
    return divclf2.add_temporal_features(c,nodes,edges,zarr.open_array(str(zp/'0'),mode='r'),q)
with ProcessPoolExecutor(4) as ex: C2=pd.concat(list(ex.map(run,sorted(C.ds.unique()))),ignore_index=True)
C2.to_parquet('divres/oof/cands_all2.parquet'); print(C2.shape)
