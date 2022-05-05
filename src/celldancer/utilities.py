# functions borrowed from scv; 
# TO DO: Change code style for every functions!!!!!!

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# progress bar
import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
# end - progress bar

######### pseudotime rsquare
from statsmodels.nonparametric.kernel_regression import KernelReg

def _non_para_kernel_t4(X,Y,down_sample_idx):
    # (no first cls),pseudotime r square calculation
    # this version has downsampling section
    # TO DO WHEN ONLY USING ONE GENE, WILL CAUSL PROBLEM WHEN COMBINING
    # Usage: Gene pseudotime fitting and r square (moved to utilities)
    # input: X,Y
    # return: estimator, r_square
    # example: 
    # X = pd.DataFrame(np.arange(100)*np.pi/100)
    # Y = pd.DataFrame(np.sin(X)+np.random.normal(loc = 0, scale = 0.5, size = (100,1)))
    # estimator,r_square=non_para_kernel(X,Y)
    
    # X2=pd.DataFrame(np.random.randint(0,100,size=[200,1]))
    # Y2=pd.DataFrame(np.random.normal(9,5,size=[200]))
    # X = pd.DataFrame(np.arange(100)*np.pi/100)
    # Y = pd.DataFrame(np.sin(X)+np.random.normal(loc = 0, scale = 0.5, size = (100,1)))
    from statsmodels.nonparametric.kernel_regression import KernelReg
    import matplotlib.pyplot as plt
    print('_non_para_kernel_t4')
    Y_sampled=Y[X['index'].isin(down_sample_idx)]
    X_sampled=X[X['index'].isin(down_sample_idx)].time
    kde=KernelReg(endog=Y_sampled,
                           exog=X_sampled,
                           var_type='c',#变量的类型
                           #ckertype='gaussian',#用于连续变量的内核
                           #bw='cv_ls'#带宽，数值：指定带宽； ‘cv_ls’：最小二乘交叉验证； ‘aic’：AIC Hurvich带宽估计
                           )
    #X=merged.time
    #Y=merged.s0
    #print(kde.r_squared())
    n=X_sampled.shape[0]

    estimator = kde.fit(X_sampled)
    estimator = np.reshape(estimator[0],[n,1])

    return(estimator,kde.r_squared())

def getidx_downSampling_embedding(load_cellDancer,cell_choice=None):
    # find the origional id

    if cell_choice is not None:
        load_cellDancer=load_cellDancer[load_cellDancer.cellIndex.isin(cell_choice)]
        
    embedding=load_cellDancer.loc[load_cellDancer.gene_name==list(load_cellDancer.gene_name)[0]][['embedding1','embedding2']]

    # get transfer id
    from sampling import sampling_embedding
    idx_downSampling_embedding = sampling_embedding(embedding,
                para='neighbors',
                target_amount=0,
                step=(30,30) # TODO: default is 30 
                )
    if cell_choice is None:
        return(idx_downSampling_embedding)
    else:
        # transfer to the id of origional all detail list
        onegene=load_cellDancer[load_cellDancer.gene_name==list(load_cellDancer.gene_name)[0]].copy()
        onegene.loc[:,'transfer_id']=range(len(onegene))
        sampled_left=onegene[onegene.transfer_id.isin(idx_downSampling_embedding)]
        transfered_index=sampled_left.cellIndex
        return(transfered_index)


def combine_parallel_result(result,gene_list,sampled_idx,merged_part_time):
    # combine result of rsquare and non-para fitting obtained from parallel computing
    for i,result_i in enumerate(result):

        r_square=result_i[1]
        non_para_fit=result_i[0]
        #print(r_square)
        if i == 0:
            r_square_list = r_square
            non_para_fit_list = np.transpose(non_para_fit)
        else:
            r_square_list = np.vstack((r_square_list, r_square))
            non_para_fit_list = np.vstack((non_para_fit_list, np.transpose(non_para_fit)[0]))
    r_square=pd.DataFrame({'gene_name':gene_list,'r_square':np.transpose(r_square_list)[0]})

    non_para_fit_heat=pd.DataFrame(non_para_fit_list,index=gene_list)
    non_para_fit_heat.columns=merged_part_time[merged_part_time['index'].isin(sampled_idx)]['index']

    non_para_list=pd.DataFrame(non_para_fit_list)
    non_para_list['combined']=non_para_list.values.tolist()
    r_square
    r_square_non_para_list=pd.concat([r_square,non_para_list['combined']],axis=1)
    r_square_non_para_list_sort=r_square_non_para_list.sort_values(by=['r_square'], axis=0, ascending=False)
    
    return(r_square_non_para_list_sort,non_para_fit_heat,non_para_fit_list)    
    
def get_rsquare(load_cellDancer,gene_list,s0_merged_part_time,s0_merged_part_gene,cell_choice=None,):
    # downsample
    sampled_idx=getidx_downSampling_embedding(load_cellDancer,cell_choice=cell_choice)
    
    # PARALLEL thread
    from joblib import Parallel, delayed
    # run parallel
    with tqdm_joblib(tqdm(desc="Calculate rsquare", total=len(gene_list))) as progress_bar:
        result = Parallel(n_jobs= -1, backend="loky")( # TODO: FIND suitable njobs
            delayed(_non_para_kernel_t4)(s0_merged_part_time,s0_merged_part_gene[gene_list[gene_index]],sampled_idx)
            for gene_index in range(0,len(gene_list)))

    # combine
    r_square_non_para_list_sort,non_para_fit_heat,non_para_fit_list=combine_parallel_result(result,gene_list,sampled_idx,s0_merged_part_time)
    
    return (r_square_non_para_list_sort,non_para_fit_heat,non_para_fit_list,sampled_idx)


######### pseudotime rsquare


def get_gene_s0_by_time(cell_time,load_cellDancer): # pseudotime
    cell_time_time_sort=cell_time.sort_values('pseudotime')
    cell_time_time_sort.columns=['index','time']

    s0_heatmap_raw=load_cellDancer.pivot(index='cellIndex', columns='gene_name', values='s0')

    s0_heatmap_raw
    s0_merged=pd.merge(cell_time_time_sort,s0_heatmap_raw,left_on='index', right_on='cellIndex') # TODO: NOT cellIndex in the future

    s0_merged_part_gene=s0_merged.loc[:, s0_merged.columns[2:]]
    s0_merged_part_time=s0_merged.loc[:, s0_merged.columns[0:2]]
    
    return(s0_merged_part_gene,s0_merged_part_time)

def rank_rsquare(load_cellDancer,gene_list=None,cluster_choice=None):
    cell_time=load_cellDancer[load_cellDancer.gene_name==load_cellDancer.gene_name[0]][['cellIndex','pseudotime']]
    s0_merged_part_gene,s0_merged_part_time=get_gene_s0_by_time(cell_time,load_cellDancer)
    
    onegene=load_cellDancer[load_cellDancer.gene_name==load_cellDancer.gene_name[0]]
    
    if cluster_choice is None:
        cluster_choice=list(onegene.clusters.drop_duplicates())
    cell_idx=list(onegene[onegene.clusters.isin(cluster_choice)].cellIndex)
    
    if gene_list is None:
        gene_list=s0_merged_part_gene.columns
    r_square_non_para_list_sort,non_para_fit_heat,non_para_fit_list,sampled_idx=get_rsquare(load_cellDancer,gene_list,s0_merged_part_time,s0_merged_part_gene,cell_choice=cell_idx)
    return(r_square_non_para_list_sort[['gene_name','r_square']].reset_index(drop=True))


def adata_to_raw_with_embed(adata,save_path,gene_list=None):
    '''convert adata to raw data format with embedding info
    data:
    save_path:
    gene_list (optional):

    return: panda dataframe with gene_list,u0,s0,cellID,clusters,embedding1,embedding2
    
    run: test=adata_to_raw(adata,'/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/bin/cellDancer-development_20220128/src/output/test.csv',gene_list=genelist_all)
    ref: mel - loom_to_celldancer_raw.py
    '''
    def adata_to_raw_one_gene(data, para, gene):
        '''
        convert adata to raw data format (one gene)
        data: an anndata
        para: the varable name of u0, s0, and gene name
        para = ['Mu', 'Ms']
        '''
        data2 = data[:, data.var.index.isin([gene])].copy()
        u0 = data2.layers[para[0]][:,0].copy().astype(np.float32)
        s0 = data2.layers[para[1]][:,0].copy().astype(np.float32)
        raw_data = pd.DataFrame({'gene_list':gene, 'u0':u0, 's0':s0})
        return(raw_data)

    if gene_list is None: gene_list=adata.var.index

    for i,gene in enumerate(gene_list):
        print("processing:"+str(i)+"/"+str(len(adata.var_names)))
        data_onegene = adata_to_raw_one_gene(adata, para=['Mu', 'Ms'], gene=gene)
        if i==0:
            data_onegene.to_csv(save_path,header=True,index=False)
        else:
            data_onegene.to_csv(save_path,mode='a',header=False,index=False)
    
    # cell info
    gene_num=len(gene_list)
    cellID=pd.DataFrame({'cellID':adata.obs.index})
    celltype_meta=adata.obs['celltype'].reset_index(drop=True)
    celltype=pd.DataFrame({'clusters':celltype_meta})#
    embed_map=pd.DataFrame({'embedding1':adata.obsm['X_umap'][:,0],'embedding2':adata.obsm['X_umap'][:,1]})
    # embed_info_df = pd.concat([embed_info]*gene_num)
    embed_info=pd.concat([cellID,celltype,embed_map],axis=1)
    embed_raw=pd.concat([embed_info]*gene_num)
    embed_raw=embed_raw.reset_index(drop=True)
    
    raw_data=pd.read_csv(save_path)
    raw_data=pd.concat([raw_data,embed_raw],axis=1)
    
    raw_data.to_csv(save_path,header=True,index=False)

    return(raw_data)



def adata_to_raw(adata,save_path,gene_list=None):
    '''convert adata to raw data format
    data:
    save_path:
    gene_list (optional):

    return: panda dataframe with gene_list,u0,s0,cellID
    
    run: test=adata_to_raw(adata,'/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/bin/cellDancer-development_20220128/src/output/test.csv',gene_list=genelist_all)
    ref: mel - loom_to_celldancer_raw.py
    '''
    def adata_to_raw_one_gene(data, para, gene):
        '''
        convert adata to raw data format (one gene)
        data: an anndata
        para: the varable name of u0, s0, and gene name
        para = ['Mu', 'Ms']
        '''
        data2 = data[:, data.var.index.isin([gene])].copy()
        u0 = data2.layers[para[0]][:,0].copy().astype(np.float32)
        s0 = data2.layers[para[1]][:,0].copy().astype(np.float32)
        raw_data = pd.DataFrame({'gene_list':gene, 'u0':u0, 's0':s0})
        raw_data['cellID']=adata.obs.index
        return(raw_data)

    if gene_list is None: gene_list=adata.var.index

    for i,gene in enumerate(gene_list):
        print("processing:"+str(i)+"/"+str(len(adata.var_names)))
        data_onegene = adata_to_raw_one_gene(adata, para=['Mu', 'Ms'], gene=gene)
        if i==0:
            data_onegene.to_csv(save_path,header=True,index=False)
        else:
            data_onegene.to_csv(save_path,mode='a',header=False,index=False)
    raw_data=pd.read_csv(save_path)

    return(raw_data)

def filter_by_neighbor_sample_parallel(load_raw_data,step_i=15,step_j=15,cutoff_s0_zero_ratio=0.2,cutoff_u0_zero_ratio=0.2,gene_amt_each_job=100):
    from joblib import Parallel, delayed
    import pandas as pd
    import numpy as np

    '''filter genes with'''
    # parallel filter gene_by_neighbor_sample_one_gene
    def filter_gene_by_neighbor_sample_one_gene(gene,load_raw_data,step_i=None,step_j=None,cutoff_s0_zero_ratio=None,cutoff_u0_zero_ratio=None,gene_amt_each_job=None):
        # print(gene)
        u_s= np.array(load_raw_data[load_raw_data['gene_list']==gene][["u0","s0"]]) # u_s
        sampling_idx=sampling_neighbors(u_s[:,0:2], step_i=step_i,step_j=step_j,percentile=15) # Sampling
        u_s_downsample = u_s[sampling_idx,0:4]
        u_s_df=pd.DataFrame({"s0":u_s_downsample[:, 1],'u0':u_s_downsample[:, 0]})
        u_s_df=u_s_df[~((u_s_df.s0==0) & (u_s_df.u0==0))]
        # print(u_s_df)
        u_s_df_zero_amt=u_s_df.agg(lambda x: x.eq(0).sum())
        sampled_gene_amt=len(u_s_df)
        u_s_df_zero_ratio=u_s_df_zero_amt/sampled_gene_amt
        # plt.figure(None,(6,6))
        # plt.scatter(u_s_df.s0,u_s_df.u0,alpha=0.1)
        # plt.show()
        # return [u_s_df_zero_ratio.s0,u_s_df_zero_ratio.u0]
        # return(u_s_df)
        if ~(u_s_df_zero_ratio.s0>cutoff_s0_zero_ratio or u_s_df_zero_ratio.u0>cutoff_u0_zero_ratio):
            return(gene)

    def filter_gene_by_neighbor_sample(start_point,load_raw_data,gene_list=None,step_i=None,step_j=None,cutoff_s0_zero_ratio=None,cutoff_u0_zero_ratio=None,gene_amt_each_job=None):
        if start_point+gene_amt_each_job<len(load_raw_data.gene_list.drop_duplicates()):
            gene_list=load_raw_data.gene_list.drop_duplicates()[start_point:(start_point+gene_amt_each_job)]
        else:
            gene_list=load_raw_data.gene_list.drop_duplicates()[start_point:,]
        print(gene_list)
        gene_list_keep=[]
        for i,gene in enumerate(gene_list):
            print(i)
            filter_result=filter_gene_by_neighbor_sample_one_gene(gene,load_raw_data,step_i=step_i,step_j=step_j,cutoff_s0_zero_ratio=cutoff_s0_zero_ratio,cutoff_u0_zero_ratio=cutoff_u0_zero_ratio,gene_amt_each_job=gene_amt_each_job)
            if filter_result is not None:gene_list_keep.append(filter_result)
        return(gene_list_keep)

    def parallel_get_gene(load_raw_data,gene_list=None,step_i=None,step_j=None,cutoff_s0_zero_ratio=None,cutoff_u0_zero_ratio=None,gene_amt_each_job=None):
        if gene_list is None:
            gene_list=load_raw_data.gene_list.drop_duplicates().reset_index(drop=True)
        else:
            load_raw_data=load_raw_data[load_raw_data.gene_list.isin(gene_list)]
        print(gene_list)
        result = Parallel(n_jobs=-1, backend="loky",verbose=10)(
            delayed(filter_gene_by_neighbor_sample)(start_point,load_raw_data,gene_list=gene_list,step_i=step_i,step_j=step_j,cutoff_s0_zero_ratio=cutoff_s0_zero_ratio,cutoff_u0_zero_ratio=cutoff_u0_zero_ratio,gene_amt_each_job=gene_amt_each_job)
            for start_point in range(0,len(gene_list),gene_amt_each_job))
        return(result)

    gene_list_keep=parallel_get_gene(load_raw_data,step_i=step_i,step_j=step_j,cutoff_s0_zero_ratio=cutoff_s0_zero_ratio,cutoff_u0_zero_ratio=cutoff_u0_zero_ratio,gene_amt_each_job=gene_amt_each_job)

    # combine parallel results
    gene_list_keep_fin=[]
    for segment_list in gene_list_keep:
        gene_list_keep_fin=gene_list_keep_fin+segment_list
    len(gene_list_keep_fin)
    gene_list_keep_fin_pd=pd.DataFrame({'gene_list':gene_list_keep_fin})

    return(gene_list_keep_fin_pd)


def panda_to_adata():
    '''panda_to_adata'''
    print('test')


def filter_by_cost():
    '''filter_by_cost'''
    print('test')


def calculate_occupy_ratio_and_cor(gene_choice,data, u_fragment=30, s_fragment=30):
    '''calculate occupy ratio and the correlation between u0 and s0
    ref: analysis_calculate_occupy_ratio.py
    parameters
    data -> rawdata[['gene_list', 'u0','s0']]

    return(ratio2, cor2)
    ratio2 [['gene_choice','ratio']]
    ratio2 [['gene_choice','correlation']]
    '''
    def identify_in_grid(u, s, onegene_u0_s0):
        select_cell =onegene_u0_s0[(onegene_u0_s0[:,0]>u[0]) & (onegene_u0_s0[:,0]<u[1]) & (onegene_u0_s0[:,1]>s[0]) & (onegene_u0_s0[:,1]<s[1]), :]
        if select_cell.shape[0]==0:
            return False
        else:
            return True

    def build_grid_list(u_fragment,s_fragment,onegene_u0_s0):
        min_u0 = min(onegene_u0_s0[:,0])
        max_u0 = max(onegene_u0_s0[:,0])
        min_s0 = min(onegene_u0_s0[:,1])
        max_s0 = max(onegene_u0_s0[:,1])
        u0_coordinate=np.linspace(start=min_u0, stop=max_u0, num=u_fragment+1).tolist()
        s0_coordinate=np.linspace(start=min_s0, stop=max_s0, num=s_fragment+1).tolist()
        u0_array = np.array([u0_coordinate[0:(len(u0_coordinate)-1)], u0_coordinate[1:(len(u0_coordinate))]]).T
        s0_array = np.array([s0_coordinate[0:(len(s0_coordinate)-1)], s0_coordinate[1:(len(s0_coordinate))]]).T
        return u0_array, s0_array

    # data = raw_data2
    ratio = np.empty([len(gene_choice), 1])
    cor = np.empty([len(gene_choice), 1])
    for idx, gene in enumerate(gene_choice):
        print(idx)
        onegene_u0_s0=data[data.gene_list==gene][['u0','s0']].to_numpy()
        u_grid, s_grid=build_grid_list(u_fragment,s_fragment,onegene_u0_s0)
        # occupy = np.empty([1, u_grid.shape[0]*s_grid.shape[0]])
        occupy = 0
        for i, s in enumerate(s_grid):
            for j,u in enumerate(u_grid):
                #print(one_grid)
                if identify_in_grid(u, s,onegene_u0_s0):
                    # print(1)
                    occupy = occupy + 1
        occupy_ratio=occupy/(u_grid.shape[0]*s_grid.shape[0])
        # print('occupy_ratio for '+gene+"="+str(occupy_ratio))
        ratio[idx,0] = occupy_ratio
        cor[idx, 0] = np.corrcoef(onegene_u0_s0[:,0], onegene_u0_s0[:,1])[0,1]
    ratio2 = pd.DataFrame({'gene_choice': gene_choice, 'ratio': ratio[:,0]})
    cor2 = pd.DataFrame({'gene_choice': gene_choice, 'correlation': cor[:,0]})
    return(ratio2, cor2)

def para_cluster_heatmap():
    '''para_cluster_heatmap'''
    print('para_cluster_heatmap')

def find_neighbors(adata, n_pcs=30, n_neighbors=30):
    '''Find neighbors by using pca on UMAP'''
    from scanpy import Neighbors
    import warnings

    neighbors = Neighbors(adata)
    with warnings.catch_warnings():  # ignore numba warning (umap/issues/252)
        warnings.simplefilter("ignore")
        neighbors.compute_neighbors(
            n_neighbors=n_neighbors,
            knn=True,
            n_pcs=n_pcs,
            method="umap",
            use_rep="X_pca",
            random_state=0,
            metric="euclidean",
            metric_kwds={},
            write_knn_indices=True,
        )

    adata.obsp["distances"] = neighbors.distances
    adata.obsp["connectivities"] = neighbors.connectivities
    adata.uns["neighbors"]["connectivities_key"] = "connectivities"
    adata.uns["neighbors"]["distances_key"] = "distances"

    if hasattr(neighbors, "knn_indices"):
        adata.uns["neighbors"]["indices"] = neighbors.knn_indices
        adata.uns["neighbors"]["params"] = {
            "n_neighbors": n_neighbors,
            "method": "umap",
            "metric": "euclidean",
            "n_pcs": n_pcs,
            "use_rep": "X_pca",
        }

def moments(adata):
    '''Calculate moments'''
    connect = adata.obsp['connectivities'] > 0
    connect.setdiag(1)
    connect = connect.multiply(1.0 / connect.sum(1))
    #pd.DataFrame(connect.todense(), adata.obs.index.tolist(), adata.obs.index.tolist())
    #pd.DataFrame(connect.multiply(1.0 / connect.sum(1)).dot(adata.layers['unspliced'].todense()), 
    #adata.obs.index.tolist(), adata.var.index.tolist())
    adata.layers["Mu"] = csr_matrix.dot(connect, csr_matrix(adata.layers["unspliced"])).astype(np.float32).A
    adata.layers["Ms"] = csr_matrix.dot(connect, csr_matrix(adata.layers["spliced"])).astype(np.float32).A


def find_nn_neighbors(
        data=None, 
        gridpoints_coordinates=None, 
        n_neighbors=None,
        radius=None):
    '''
    data: numpy ndarray
    gridpoints_coordinates: numpy ndarray
    n_neighbors: int
    raidus: float
    '''

    if gridpoints_coordinates is None:
        gridpoints_coordinates = data

    if n_neighbors is None and radius is not None:
        nn = NearestNeighbors(radius = radius, n_jobs = -1)
        nn.fit(data)
        dists, neighs = nn.radius_neighbors(gridpoints_coordinates)
    elif n_neighbors is not None and radius is None:
        nn = NearestNeighbors(n_neighbors = n_neighbors, n_jobs = -1)
        nn.fit(data)
        dists, neighs = nn.kneighbors(gridpoints_coordinates)

    return(dists, neighs)


def extract_from_df(load_cellDancer, attr_list, gene_name=None):
    '''
    Extract a single copy of a list of columns from the load_cellDancer data frame
    Returns a numpy array.
    '''
    if gene_name is None:
        gene_name = load_cellDancer.gene_name[0]
    one_gene_idx = load_cellDancer.gene_name == gene_name
    data = load_cellDancer[one_gene_idx][attr_list].dropna()
    return data.to_numpy()


def set_rcParams(fontsize=12): 
    try:
        import IPython
        ipython_format = ["png2x"]
        IPython.display.set_matplotlib_formats(*ipython_format)
    except:
        pass

    from matplotlib import rcParams

    # dpi options (mpl default: 100, 100)
    rcParams["figure.dpi"] = 100
    rcParams["savefig.dpi"] = 150

    # figure (mpl default: 0.125, 0.96, 0.15, 0.91)
    rcParams["figure.figsize"] = (6, 4)
    rcParams["figure.subplot.left"] = 0.18
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.91

    # lines (defaults:  1.5, 6, 1)
    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]

    fontsize = fontsize
    labelsize = 0.92 * fontsize

    # fonsizes (mpl default: 10, medium, large, medium)
    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = labelsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = labelsize

    # legend (mpl default: 1, 1, 2, 0.8)
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4

    # axes
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks (mpl default: k, k, medium, medium)
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = labelsize
    rcParams["ytick.labelsize"] = labelsize

    # axes grid (mpl default: False, #b0b0b0)
    rcParams["axes.grid"] = False
    rcParams["grid.color"] = ".8"

    # color map
    rcParams["image.cmap"] = "RdBu_r"



if __name__ == "__main__":
    import sys
    sys.path.append('.')
