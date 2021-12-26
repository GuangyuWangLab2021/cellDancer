import scvelo as scv
from utilities import find_neighbors,moments
from sampling import adata_to_detail
import numpy as np
import pandas as pd

adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
find_neighbors(adata, n_pcs=30, n_neighbors=30) # calculate !!!!!!!!
moments(adata)

detail_test = adata_to_detail(adata, para=['Mu', 'Ms'], gene='Tmem163')

genelist_all=adata.var_names
i=0
for g in genelist_all:
    i=i+1
    print("processing:"+str(i)+"/"+str(len(adata.var_names)))
    data_onegene = adata_to_detail(adata, para=['Mu', 'Ms'], gene=g)
    data_onegene = pd.concat([data_onegene,adata.obs['clusters'].reset_index()['clusters']],ignore_index=True,axis=1)
    data_onegene.to_csv('data/scv_data.csv',mode='a',header=False)

scvelo_data=pd.read_csv("data/scv_data.csv",names=['gene', 'u0','s0',"clusters"])