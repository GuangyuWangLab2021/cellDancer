from ipaddress import summarize_address_range
from scipy.io import mmread
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# unspliced
m_u = mmread('/Users/wanglab/Documents/ShengyuLi/Velocity/data/jmlab_velocyto/atlas_data/weiqing_sparse2normal/unspliced_trans.mtx')
M_u=pd.DataFrame.sparse.from_spmatrix(m_u)

# spliced
m_s = mmread('/Users/wanglab/Documents/ShengyuLi/Velocity/data/jmlab_velocyto/atlas_data/weiqing_sparse2normal/spliced_trans.mtx')
M_s=pd.DataFrame.sparse.from_spmatrix(m_s)

# metadata
meta=pd.read_table('/Users/wanglab/Documents/ShengyuLi/Velocity/data/jmlab_velocyto/atlas_data/meta.tab') # 139331 rows × 18 columns
genes=pd.read_table('/Users/wanglab/Documents/ShengyuLi/Velocity/data/jmlab_velocyto/atlas_data/genes.tsv',header=None) 

##########
# filter #
##########

# select by cell - stage
stage_list=['E7.25', 'E7.5', 'E7.75', 'E8.0', 'E8.25']
meta_staged=meta[meta.stage.isin(stage_list)] # 86591 rows × 18 columns
plt.scatter(meta_staged.umapX,meta_staged.umapY)

idx_staged_cell=meta_staged.index
M_s_staged=M_s[M_s.columns.intersection(idx_staged_cell)] # 29452 rows × 86591 columns
M_u_staged=M_u[M_u.columns.intersection(idx_staged_cell)] # 29452 rows × 86591 columns

# select by cell - blood cluster
blood_cluster=['Haematoendothelial progenitors','Blood progenitors 1', 'Blood progenitors 2', 'Endothelium']
meta_blood=meta_staged[meta_staged.celltype.isin(blood_cluster)]
idx_meta_blood=meta_blood.index
M_s_blood_cluster=M_s_staged[M_s_staged.columns.intersection(idx_meta_blood)] # 29452 rows × 5437 columns
M_u_blood_cluster=M_u_staged[M_u_staged.columns.intersection(idx_meta_blood)] # 29452 rows × 5437 columns

M_s_cell_col_sum=M_s_blood_cluster.sum()
M_u_cell_col_sum=M_u_blood_cluster.sum()

# filter by gene - u or s zero
M_s_gene_row_sum=M_s_blood_cluster.sum(axis=1)
M_u_gene_row_sum=M_u_blood_cluster.sum(axis=1)


filter_by_id_us_zero=list(set.union(set(M_s_gene_row_sum[M_s_gene_row_sum==0].index), set(M_u_gene_row_sum[M_u_gene_row_sum==0].index)))
M_s_filtered_zero=M_s_blood_cluster.drop(filter_by_id_us_zero) # 14348 rows × 5437 columns
M_u_filtered_zero=M_u_blood_cluster.drop(filter_by_id_us_zero) # 14348 rows × 5437 columns

# filter by gene - normalize
M_s_norm=(M_s_filtered_zero-M_s_filtered_zero.min())/M_s_filtered_zero.max()
M_u_norm=(M_u_filtered_zero-M_u_filtered_zero.min())/M_u_filtered_zero.max()

sum_u_s=M_s_norm+M_u_norm
M_us_gene_row_sum=sum_u_s.sum(axis=1)

filter_by_top_us_sum=M_us_gene_row_sum.nlargest(2000).sort_index().index
M_s_top_us=M_s_filtered_zero.loc[filter_by_top_us_sum] # 2000 rows × 5437 columns
M_u_top_us=M_u_filtered_zero.loc[filter_by_top_us_sum] # 2000 rows × 5437 columns

# filter meta info
genes_filtered=genes.loc[filter_by_top_us_sum]
meta_filtered=meta.loc[M_u_top_us.columns]
meta_filtered['ID_barcode']='ID_'+meta_filtered['barcode']

##################
# build raw_data #
##################

gene_num=len(genes_filtered)
cell_num=len(M_u_top_us.columns)

# gene_list
gene_list_df = pd.DataFrame(np.repeat(genes_filtered[1], cell_num))
gene_list_df.columns=['gene_list']

# u0 & s0
M_s_fin=pd.DataFrame()
M_u_fin=pd.DataFrame()
for i,row in enumerate(genes_filtered.index):
    print(i)
    # s0
    one_gene_s0=pd.DataFrame(M_s_top_us.loc[row])
    one_gene_s0.columns=['s0']
    M_s_fin=M_s_fin.append(one_gene_s0)
    # u0
    one_gene_u0=pd.DataFrame(M_u_top_us.loc[row])
    one_gene_u0.columns=['u0']
    M_u_fin=M_u_fin.append(one_gene_u0)
    
# clusters & cellID & embedding1 & embedding2
embed_info = meta_filtered[['barcode','celltype','umapX','umapY']]
embed_info_df = pd.concat([embed_info]*gene_num)
embed_info_df=embed_info_df.rename(columns={"barcode": "cellID", "celltype": "clusters","umapX": "embedding1", "umapY": "embedding2"})

# COMBINE

raw_data_blood=pd.concat([gene_list_df.reset_index(drop=True), M_s_fin.reset_index(drop=True),M_u_fin.reset_index(drop=True), embed_info_df.reset_index(drop=True) ], ignore_index=True,axis=1)
raw_data_blood.columns=['gene_list','s0','u0','cellID','clusters','embedding1','embedding2']
raw_data_blood.to_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/mouse_blood/raw/mouse_blood_full.csv',index=False)

#############
# plot test #
#############
raw_data_blood=pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/mouse_blood/raw/mouse_blood_full.csv')
onegene=raw_data_blood[raw_data_blood.gene_list==raw_data_blood.gene_list.loc[0]]
# ax = plt.gca()
# ax.set_ylim([5, 15])
onegene['color']=''
colorlist=['red','blue','green','orange']
for cluster,color in zip(blood_cluster,colorlist):
    onegene.loc[onegene.clusters==cluster,'color']=color
plt.scatter(onegene.embedding1,onegene.embedding2,c = onegene['color'],scale=1)

#########################################
# raw data to adata (for moments - scv) #
#########################################
import scvelo as scv
from utilities import adata_to_raw

M_s_top_us.columns=list(meta_filtered['barcode'])
M_s_top_us.index=list(genes_filtered[1])

import anndata

###
import anndata
adata_build_all_gene_in_blood=anndata.AnnData(
    X=M_s_blood_cluster.T,
    layers={
        'unspliced':M_u_blood_cluster.T.to_numpy(),
        'spliced':M_s_blood_cluster.T.to_numpy()
    }
)
adata_build_all_gene_in_blood.var.index=genes[1]
adata_build_all_gene_in_blood.obs.index=meta_filtered['barcode']
# adata_build_all_gene_in_blood.write(filename='/Users/wanglab/Documents/ShengyuLi/Velocity/data/mouse_blood/raw/mouse_blood_build_all_gene_in_blood_cell.h5ad')

adata_build_all_gene_in_blood_copy=adata_build_all_gene_in_blood.copy()
adata_build_all_gene_in_blood_copy.obs_names_make_unique()
# adata_build_all_gene_in_blood_copy.var_names_make_unique()
# scv.pp.filter_and_normalize(adata_build_all_gene_in_blood_copy)
scv.pp.remove_duplicate_cells(adata_build_all_gene_in_blood_copy)
scv.pp.neighbors(adata_build_all_gene_in_blood_copy)
# for n in[10,30,50,100,200]:
for n in[300,400]:
    scv.pp.moments(adata_build_all_gene_in_blood_copy, n_neighbors=n)
    # for gene in adata_build_all_gene_in_blood_copy.var.index[0:30]:
    for gene in ['Smim1', 'Hba-x']:
    
        data2 = adata_build_all_gene_in_blood_copy[:,adata_build_all_gene_in_blood_copy.var.index.isin([gene])].copy()
        s0 = data2.layers['Ms']
        u0=data2.layers['Mu']
        plt.figure()
        plt.title('n'+str(n)+'_moment_'+gene)
        plt.scatter(s0,u0,alpha=0.3)

        u0 = data2.layers['spliced']
        s0=data2.layers['unspliced']
        plt.figure()
        plt.title('n'+str(n)+'_raw_'+gene)
        plt.scatter(s0,u0,alpha=0.3)
# ###
# adata_build=anndata.AnnData(
#     X=M_u_top_us.T,
#     layers={
#         'unspliced':M_u_top_us.T.to_numpy(),
#         'spliced':M_s_top_us.T.to_numpy()
#     }
# )
# adata_build.write(filename='/Users/wanglab/Documents/ShengyuLi/Velocity/data/mouse_blood/raw/mouse_blood_200_full.h5ad')

# adata_build.var_names=meta_filtered['barcode']
# adata_build.obs_names=genes_filtered[1]

# adata_build_copy=adata_build.copy()
# adata_build_copy.var_names_make_unique()
# n_list=[30]
# n_pcs_list=[10]
# for n in n_list:
#     for n_pcs in n_pcs_list:
#         scv.pp.filter_and_normalize(adata_build_copy, min_shared_counts=30, n_top_genes=2000)
#         scv.pp.moments(adata_build_copy,n_neighbors=n,n_pcs=n_pcs)
#         gene_list=genes_filtered[1]
#         # for gene in gene_list[100:200]:
#         for gene in ['Smim1', 'Hba-x']:
#             data2 = adata_build_copy[ adata_build_copy.obs.index.isin([gene]),:].copy()
#             u0 = data2.layers['Ms']
#             s0=data2.layers['Mu']
#             plt.figure()
#             plt.scatter(s0,u0)


# adata_to_raw(adata_build_copy,'/Users/wanglab/Documents/ShengyuLi/Velocity/bin/cellDancer-development_20220128/src/output/test.csv',gene_list=genes_filtered[1])

# Mouse gastrulation
import scanpy
adata_paper =scanpy.read ('/Users/wanglab/Documents/ShengyuLi/Velocity/data/Gastrulation/erythroid_lineage.h5ad')

cellID_common=list(set.intersection(set(adata_paper.obs.index),set(adata_build_all_gene_in_blood_copy.obs.index)))
adata_build_all_gene_in_blood_copy.obs_names_make_unique()

for gene in ['Smim1', 'Hba-x']:
    # my preprocess
    data2 = adata_build_all_gene_in_blood_copy[adata_build_all_gene_in_blood_copy.obs.index.isin(cellID_common),adata_build_all_gene_in_blood_copy.var.index.isin([gene])].copy()
    s0_2 = data2.layers['spliced'].T[0]
    u0_2=data2.layers['unspliced'].T[0]

    su_2=pd.DataFrame({'cellID':data2.obs.index,
                       's0_2':s0_2,
                       'u0_2':u0_2
                       })

    # scv process
    data3=adata_paper[adata_paper.obs.index.isin(cellID_common),adata_paper.var.index.isin([gene])].copy()
    data3.obs.index
    s0_3 = data3.layers['spliced'].toarray().T[0]
    u0_3 = data3.layers['unspliced'].toarray().T[0]
    su_3 = pd.DataFrame({'cellID':data3.obs.index,
                    's0_3':s0_3,
                    'u0_3':u0_3
                    })

    # merged
    merged=pd.merge(su_2, su_3, left_on='cellID', right_on='cellID')

    plt.figure()
    plt.title('curr'+gene)
    plt.scatter(merged.s0_2,merged.u0_2)

    plt.figure()
    plt.title('scv'+gene)
    plt.scatter(merged.s0_3,merged.u0_3)

    plt.figure()
    plt.title('correlation-s_No_fix')
    plt.scatter(merged.s0_2,merged.s0_3,alpha=0.3)
    plt.figure()
    plt.title('correlation-s_fix')
    plt.axis([0,100,0,100])
    plt.scatter(merged.s0_2,merged.s0_3,alpha=0.3)

    plt.figure()
    plt.title('correlation-u_No_fix')
    plt.scatter(merged.u0_2,merged.u0_3,alpha=0.3)
    plt.figure()
    plt.title('correlation-u')
    plt.axis([0,10,0,10])
    plt.scatter(merged.u0_2,merged.u0_3,alpha=0.3)


########################## same cell type as adata_paper


adata_paper







