# modified form https://github.com/GuangyuWangLab2021/veloNN/blob/main/tests/scanpy_th9_velocyto.py
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import anndata
from sampling import sampling_neighbors
from colormap import *
import os
# Load the data

def sim_raw_add_ID(load_raw_data):
    gene_amt=len(set(load_raw_data.gene_list))
    raw_one_gene=load_raw_data[load_raw_data.gene_list==load_raw_data.gene_list.iloc[0]]
    cell_amt=len(raw_one_gene)
    cellID_onegene = np.arange(0, cell_amt, 1)
    cellID_np=np.tile(cellID_onegene,gene_amt)
    cellID_df=pd.DataFrame({'cellID':cellID_np})
    load_raw_data=pd.concat([load_raw_data,cellID_df], axis=1)
    return(load_raw_data)


def pipline_run_sim_data(load_raw_data,name,ratio):

    # u0&s0
    s0_mat=load_raw_data.pivot(index='gene_list', values='s0', columns='cellID')
    u0_mat=load_raw_data.pivot(index='gene_list', values='u0', columns='cellID')

    one_gene_raw=load_raw_data[load_raw_data.gene_list==load_raw_data.gene_list[0]]

    cols=one_gene_raw['cellID']
    s0_mat = s0_mat[cols].T
    u0_mat = u0_mat[cols].T
    s0_mat.to_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/build_scv_compatible_raw/s0_'+ratio+'.csv')
    u0_mat.to_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/build_scv_compatible_raw/u0_'+ratio+'.csv')

    s0_mat=pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/build_scv_compatible_raw/s0_'+ratio+'.csv')
    u0_mat=pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/build_scv_compatible_raw/u0_'+ratio+'.csv')

    adata_building=sc.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/build_scv_compatible_raw/s0_'+ratio+'.csv', delimiter=',', first_column_names=True, dtype='float32')
    adata_building.layers['Ms']=s0_mat.to_numpy()[:,1:]
    adata_building.layers['Mu']=u0_mat.to_numpy()[:,1:]

    adata_building.layers['Ms']=np.array(adata_building.layers['Ms'], dtype=float)
    adata_building.layers['Mu']=np.array(adata_building.layers['Mu'], dtype=float)

    adata_building_test=adata_building.copy()
    # adata_building_test2=adata_building.copy()

    # velocity
    # scv.pp.moments(adata_building_test, n_pcs=30, n_neighbors=30)
    # scv.pp.moments(adata_building_test, n_pcs=50, n_neighbors=50)

    # steady_state_velocity
    scv.tl.velocity(adata_building_test, vkey='steady_state_velocity', mode='steady_state')
    scv.tl.velocity_graph(adata_building_test, vkey='steady_state_velocity')

    # dynamical_velocity
    scv.tl.recover_dynamics(adata_building_test, n_jobs=30, n_top_genes = 1000)
    scv.tl.velocity(adata_building_test, vkey='dynamical_velocity',mode='dynamical',filter_genes=False)
    
    # plot - steady_state_velocity
    gene_choice=load_raw_data['gene_list'].drop_duplicates()

    for gene in list(gene_choice)[0:3]:
        scv.pl.velocity_embedding(adata_building_test, vkey='steady_state_velocity', basis=gene,
                            scale=.6, width=.0035, frameon=False, title=gene,show=False)
        # plot- dynamical_velocity
        scv.pl.velocity_embedding(adata_building_test, vkey='dynamical_velocity', basis=gene, scale=4, width=.0035,
                            frameon=False, title=gene,show=False)
        
    adata_building_test.write(filename='/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/build_scv_compatible_raw/wingpath_'+ratio+'.h5ad')

    # gene velocity plot
    gene_choice=load_raw_data['gene_list'].drop_duplicates()
    # gene_choice=['simulation0','simulation1']

    outpath='/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/velocity_result/scv'
    gene_s0_u0_s1_u1_dynamic_and_steady_df=pd.DataFrame()
    for nth,gene in enumerate(gene_choice):
        print(nth)
        # dynamical_velocity
        outpath='/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/velocity_result/scv'
        save_path=os.path.join(outpath,'scvelo_result',('wing_path_'),(gene+'_dynamic.pdf'))
        X, V=scv.pl.velocity_embedding(adata_building_test, vkey='dynamical_velocity', basis=gene,
                            fontsize=16, frameon=False,show=False)
        s0_u0_s1_u1=np.concatenate((X, V), axis=1)

        # steady_state_velocity
        save_path=os.path.join(outpath,'scvelo_result',('wing_path_'),(gene+'_static.pdf'))
        X_steady, V_steady=scv.pl.velocity_embedding(adata_building_test, vkey='steady_state_velocity', basis=gene,
                          scale=.6, width=.0035, frameon=False,show=False)
        s0_u0_s1_u1_steady=np.concatenate((X_steady, V_steady), axis=1)

        one_gene_s0_u0_s1_u1_dynamic_and_steady=np.hstack((s0_u0_s1_u1,s0_u0_s1_u1_steady))
        one_gene_s0_u0_s1_u1_dynamic_and_steady_df = pd.DataFrame(one_gene_s0_u0_s1_u1_dynamic_and_steady, columns = ['dynamic_s0','dynamic_u0','dynamic_s1','dynamic_u1','static_s0','static_u0','static_s1','static_u1'])
        one_gene_s0_u0_s1_u1_dynamic_and_steady_df.insert (0, "gene_list", gene)
        gene_s0_u0_s1_u1_dynamic_and_steady_df = pd.concat([gene_s0_u0_s1_u1_dynamic_and_steady_df,one_gene_s0_u0_s1_u1_dynamic_and_steady_df])
        
    gene_s0_u0_s1_u1_dynamic_and_steady_df.to_csv(os.path.join(outpath,'scvelo_result_wing_path__s0_u0_s1_u1_dynamic_and_steady_df_'+ratio+'.csv'),index=False)


    #
if __name__ == "__main__":
    # raw_data_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full_two_genes.csv'
    # load_raw_data=pd.read_csv(raw_data_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])

    # load_raw_data_twogenes=load_raw_data[load_raw_data.gene_list.isin(['Rgs20','Gpm6b'])]
    # load_raw_data_twogenes.to_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full_two_genes.csv',index=False, header=True)
    # load_raw_data=pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full_two_genes.csv')
    ratio_list=[0.2,0.4,0.6,0.8]

    for ratio in ratio_list:
        print(ratio)
        ratio=str(ratio)
        load_raw_data=pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/raw/wing_path_Path2Upper_1000__R'+ratio+'.csv')
        load_raw_data=sim_raw_add_ID(load_raw_data)
        name='sim_wing_path'
        pipline_run_sim_data(load_raw_data,name,ratio)


    # plot (take a look)
    import pandas as pd
    from velocity_plot import velocity_plot as vpl

    scv_result=pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/velocity_result/scv/scvelo_result_wing_path__s0_u0_s1_u1_dynamic_and_steady_df_0.2.csv')    
    
    s0_u0_s1_u1_dynamic=scv_result[['gene_list','dynamic_s0','dynamic_u0','dynamic_s1','dynamic_u1']]
    s0_u0_s1_u1_static=scv_result[['gene_list','static_s0','static_u0','static_s1','static_u1']]
    s0_u0_s1_u1_dynamic=s0_u0_s1_u1_dynamic.rename(columns={'gene_list': 'gene_name', 'dynamic_s0': 's0', 'dynamic_u0': 'u0', 'dynamic_s1': 's1', 'dynamic_u1': 'u1'})
    s0_u0_s1_u1_static=s0_u0_s1_u1_static.rename(columns={'gene_list': 'gene_name', 'static_s0': 's0', 'static_u0': 'u0', 'static_s1': 's1', 'static_u1': 'u1'})

    s0_u0_s1_u1_dynamic.s1=s0_u0_s1_u1_dynamic.s0+s0_u0_s1_u1_dynamic.s1
    s0_u0_s1_u1_dynamic.u1=s0_u0_s1_u1_dynamic.u0+s0_u0_s1_u1_dynamic.u1
    
    s0_u0_s1_u1_static.s1=s0_u0_s1_u1_static.s0+s0_u0_s1_u1_static.s1
    s0_u0_s1_u1_static.u1=s0_u0_s1_u1_static.u0+s0_u0_s1_u1_static.u1
    
    gene_list=list(set(scv_result.gene_list))
    for gene in gene_list[0:50]:
        vpl.velocity_gene(gene,s0_u0_s1_u1_dynamic,color_scatter='orange')
        vpl.velocity_gene(gene,s0_u0_s1_u1_static,color_scatter='green')



"""







###############################################
######    scvelo pipeline copyed(below content)            #########
###############################################



# load embedding df and reset build matched CellID
embedding_df=pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Melanoma/data/embedding/embedding.csv')
embedding_df['CellID'] = embedding_df['cell_id'].str.replace('-1','x')
for i in range(1,9):
    embedding_df['CellID'] = embedding_df['CellID'].str.replace((str(i)+'_'),('SC146!'+str(i)+':'))
for i in range(1,9):
    embedding_df['CellID'] = embedding_df['CellID'].str.replace('!','_')


###############################################
######    scvelo pipeline             #########
###############################################
def pipline(adata,name):
    '''
    preprocess and filter
    '''

    print('uniqueing')
    adata.var_names_make_unique()

    # filter Gene
    print('filtering')
    sc.pp.filter_cells(adata, min_genes=1000)
    sc.pp.filter_genes(adata, min_cells=8000)
    sc.pp.filter_genes(adata, min_counts=40000)
    sc.pp.filter_cells(adata, min_counts=5000)

    # preprocess
    print('preprocess')
    adata.var['mt'] = adata.var.index.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
    adata.var['rp'] = adata.var.index.str.startswith('Rp')  # annotate the group of mitochondrial genes as 'Rp'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['rp'], percent_top=None, log1p=False, inplace=True)
    # sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True)
    # sc.pl.violin(adata, ['pct_counts_mt', 'pct_counts_rp'], jitter=0.4, multi_panel=True)
    # sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    # sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

    # filter gene
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 0.6, :]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    #sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pl.highly_variable_genes(adata)

    #adata.var[(adata.var['rp']==False)&(adata.var['highly_variable'])]
    adata = adata[:, adata.var.highly_variable]
    adata = adata[:, adata.var.highly_variable&(adata.var['rp']==False)&(adata.var['mt'])==False]
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_rp'])
    sc.pp.scale(adata, max_value=10)

    # pca
    print('pca')
    sc.tl.pca(adata, svd_solver='arpack')

    # Computing the neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    # Embedding the neighborhood graph
    #sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    #sc.tl.umap(adata, init_pos='paga')
    sc.tl.umap(adata, random_state=2)

    # cluster the neighborhood graph
    #sc.tl.leiden(adata, key_added = "Sox17") # default resolution in 1.0
    sc.tl.leiden(adata, key_added = "leiden") # default resolution in 1.0

    sc.pl.umap(adata, color="leiden", use_raw=False)

    '''
    add and check cluster info
    '''
    # add cluster info
    def adata_add_clusters(adata, embedding_df):
        CellID_loom = pd.DataFrame({'CellID':adata.obs_names})
        embedding_loom=pd.merge(CellID_loom, embedding_df, on="CellID",how='left')
        adata.obs['clusters']=''
        for i in range(0,len(adata.obs['clusters'])):
            adata.obs['clusters'][i]=embedding_loom['clusters'][i]
        
    adata_add_clusters(adata, embedding_df)

    # check cluster info
    # cluster_list=['Dendritic cells', 'Fibroblasts', 'Macrophages', 'Monocytes', 'NK cells', 'T cells', np.NaN]
    # for i in cluster_list:
    #     s1 = adata.obs['clusters'][adata.obs['clusters']==i].shape
    #     # s2 = adatabkup.obs['clusters'][adatabkup.obs['clusters']==i].shape
    #     print(i)
    #     print(s1)
    #     # print(s2)

    '''
    velocity calculation
    '''
    # start to calculate velocity
    print('moments')
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # Dynamical Model (dynamic)
    print('recover_dynamics')
    scv.tl.recover_dynamics(adata, n_jobs=12)
    scv.tl.velocity(adata, mode='dynamical')
    scv.tl.velocity_graph(adata)

    scv.settings.set_figure_params('scvelo', dpi_save=200, dpi=80, transparent=True)
    scv.settings.plot_prefix = 'scvelo_'
    scv.settings.verbosity = 2
    scv.pl.velocity_embedding_stream(adata, basis='umap',legend_loc='right margin', save='_stream_'+name+'.png')

    scv.tl.velocity_pseudotime(adata)
    scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot',save= '_pseudotime_'+name+'.png')

    scv.tl.velocity_confidence(adata)
    keys = 'velocity_length', 'velocity_confidence'
    scv.pl.scatter(adata, c=keys, cmap='coolwarm', perc=[5, 95],save= '_confidence_'+name+'.png')

    df = adata.obs.groupby('clusters')[keys].mean().T
    df.style.background_gradient(cmap='coolwarm', axis=1)
    scv.pl.velocity_graph(adata, threshold=.1,legend_loc='right margin',save= '_velocity_grapy_'+name+'.png')

    x, y = scv.utils.get_cell_transitions(adata, basis='umap', starting_cell=70)
    ax = scv.pl.velocity_graph(adata, c='lightgrey', edge_width=.05, show=False)
    ax = scv.pl.scatter(adata, x=x, y=y, s=120, c='ascending', cmap='gnuplot', ax=ax, save= '_anscestors_'+name+'.png')

    # Latent time
    scv.tl.latent_time(adata)
    scv.pl.scatter(adata, color='latent_time', color_map='gnuplot', size=80,save= '_latent_time_'+name+'.png')

    # Scatter plot of velocities on the embedding.
    scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, color='leiden',save= '_scatter_embedding_velocity_pipeline_cluster_color_'+name+'.png')
    scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save= '_scatter_embedding_velocity_cluster_color_'+name+'.png')

if __name__ == "__main__":
    # embedding info obtained from hanan's program
    embedding_df=pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Melanoma/data/embedding/embedding.csv')
    embedding_df['CellID'] = embedding_df['cell_id'].str.replace('-1','x')
    for i in range(1,9):
        embedding_df['CellID'] = embedding_df['CellID'].str.replace((str(i)+'_'),('SC146!'+str(i)+':'))
    for i in range(1,9):
        embedding_df['CellID'] = embedding_df['CellID'].str.replace('!','_')

    filename='/Users/shengyuli/OneDrive - Houston Methodist/work/Melanoma/data/loom/combined/Sedentary.loom'
    #file = '/Users/shengyuli/OneDrive - Houston Methodist/work/Melanoma/data/loom/combined/Exercise.loom'
    name='Sedentary_2nd_another_filter_strategy'
    adata= sc.read_loom(filename)

    pipline(adata,name)



########################################################################
######     Interprete the velocities in gene level (TO DO)     #########
########################################################################
# Interprete the velocities
scv.pl.velocity(adata, ['Lilr4b', 'Med13l', 'Emb', 'Rrbp1', 'Il9'], ncols=1, color='leiden')
scv.pl.velocity(adata, ['Il9'], ncols=1, color='leiden')

# Cluster-specific top-likelihood genes
scv.tl.rank_dynamical_genes(adata, groupby='leiden')
df = scv.get_df(adata, 'rank_dynamical_genes/names')
df.head(5)
for cluster in ['2', '3', '1', '0']:
    scv.pl.scatter(adata, df[cluster][:5], ylabel=cluster, frameon=True, color='leiden')

# Interprete the velocities
scv.pl.velocity(adata, ['Il9'], ncols=1, color='leiden')
# cycling progenitors
scv.tl.score_genes_cell_cycle(adata)
scv.pl.scatter(adata, color_gradients=['S_score', 'G2M_score'], smooth=True, perc=[5, 95])

"""