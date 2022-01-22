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




genelist_all=adata.var['highly_variable_genes'].index #2000

#adata.obsm['X_umap'][:,0]
#adata.obsm['X_umap'][:,1]

i=0
embed_xy=pd.DataFrame(adata.obsm['X_umap'], columns = ['embed_x','embed_y'])
data=pd.DataFrame()
for g in genelist_all:
    i=i+1
    print("processing:"+str(i)+"/"+"2000")
    ix=np.where(adata.var['highly_variable_genes'].index == g)[0][0]
    data_onegene = pd.DataFrame({'gene': g, 'u0': adata.layers['Mu'].T[ix,:],'s0':adata.layers['Ms'].T[ix,:],'clusters':adata.obs['clusters'],'CellID':adata.obs.index}, columns=['gene', 'u0','s0','clusters','CellID'])
    data_onegene=pd.concat([data_onegene.reset_index(drop=True),embed_xy],axis=1)
    data_onegene.to_csv('data/scv_data_full.csv',mode='a',header=False)
load_raw_data=pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/data/scv_data_full.csv',names=['gene_list', 'u0','s0',"clusters",'CellID','embed_x','embed_y'])
data.to_pickle("denGyr.pkl")

import matplotlib.pyplot as plt
i=200
for g in genelist_all[200:]:
    i=i+1
    print(i)
    plt.figure()
    plt.scatter(load_raw_data[load_raw_data.gene_list==g]['s0'], load_raw_data[load_raw_data.gene_list==g]['u0'],s=10,c='orange')
    plt.title(g)
    plt.savefig("output/scv_data_s0u0/"+g+".png")
