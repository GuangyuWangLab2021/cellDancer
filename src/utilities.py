# functions borrowed from scv; 
# TO DO: Change code style for every functions!!!!!!

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

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

    for i,gene in enumerate(gene_list):
        print("processing:"+str(i)+"/"+str(len(adata.var_names)))
        data_onegene = adata_to_raw_one_gene(adata, para=['Mu', 'Ms'], gene=gene)
        if i==0:
            data_onegene.to_csv(save_path,header=True,index=False)
        else:
            data_onegene.to_csv(save_path,mode='a',header=False,index=False)
    raw_data=pd.read_csv(save_path)

    return(raw_data)

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