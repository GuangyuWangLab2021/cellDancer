import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import scvelo as scv




def sampling_neighbors(gene_u0_s0,step_i=20,step_j=20):
    from scipy.stats import norm
    def gaussian_kernel(X, mu = 0, sigma=1):
        return np.exp(-(X - mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    steps = step_i, step_j
    grs = []
    for dim_i in range(gene_u0_s0.shape[1]):
        m, M = np.min(gene_u0_s0[:, dim_i]), np.max(gene_u0_s0[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, steps[dim_i])
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
    gridpoints_coordinates = gridpoints_coordinates + norm.rvs(loc=0, scale=0.15, size=gridpoints_coordinates.shape)
    
    np.random.seed(10) # set random seed
    nn = NearestNeighbors()
    nn.fit(gene_u0_s0[:,0:2])
    dist, ixs = nn.kneighbors(gridpoints_coordinates, 20)
    ix_choice = ixs[:,0].flat[:]
    ix_choice = np.unique(ix_choice)

    nn = NearestNeighbors()
    nn.fit(gene_u0_s0[:,0:2])
    dist, ixs = nn.kneighbors(gene_u0_s0[ix_choice, 0:2], 20)
    density_extimate = gaussian_kernel(dist, mu=0, sigma=0.5).sum(1)
    bool_density = density_extimate > np.percentile(density_extimate, 25)
    ix_choice = ix_choice[bool_density]
    return(ix_choice)

def sampling_inverse(gene_u0_s0):
    u0 = gene_u0_s0[:,0]
    s0 = gene_u0_s0[:,1]
    values = np.vstack([u0,s0])
    kernel = scipy.stats.gaussian_kde(values)
    p = kernel(values)
    # p2 = (1/p)/sum(1/p)
    p2 = (1/p)/sum(1/p)
    idx = np.arange(values.shape[1])
    r = scipy.stats.rv_discrete(values=(idx, p2))
    idx_choice = r.rvs(size=500)
    return(idx_choice)

def sampling_circle(gene_u0_s0):
    u0 = gene_u0_s0[:,0]
    s0 = gene_u0_s0[:,1]
    values = np.vstack([u0,s0])
    kernel = scipy.stats.gaussian_kde(values)
    p = kernel(values)
    idx = np.arange(values.shape[1])
    tmp_p = np.square((1-(p/(max(p)))**2))+0.0001
    # tmp_p = np.square((1-(((p+0.4*max(p))*4-2*max(p+0.4*max(p)))/(2*max(p+0.4*max(p))))**2))+0.0001
    p2 = tmp_p/sum(tmp_p)
    r = scipy.stats.rv_discrete(values=(idx, p2))
    idx_choice = r.rvs(size=500)
    return(idx_choice)


def sampling_adata(detail, para):
    if para == 'neighbors':
        data_U_S= np.array(detail[["u0","s0"]])
        idx = sampling_neighbors(data_U_S)
    elif para == 'inverse':
        data_U_S= np.array(detail[["u0","s0"]])
        idx = sampling_inverse(data_U_S)
    elif para == 'circle':
        data_U_S= np.array(detail[["u0","s0"]])
        idx = sampling_circle(data_U_S)
    else:
        print('para is neighbors or inverse or circle')
    return(idx)

def adata_to_detail(data, para, gene):
    '''
    convert adata to detail format
    data: an anndata
    para: the varable name of u0, s0, and gene name
    para = ['Mu', 'Ms']
    '''
    data2 = data[:, data.var.index.isin([gene])].copy()
    u0 = data2.layers[para[0]][:,0].copy().astype(np.float32)
    s0 = data2.layers[para[1]][:,0].copy().astype(np.float32)
    detail = pd.DataFrame({'gene_list':gene, 'u0':u0, 's0':s0})
    return(detail)




adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
find_neighbors(adata, n_pcs=30, n_neighbors=30)
moments(adata)

detail2 = adata_to_detail(adata, para=['Mu', 'Ms'], gene='Tmem163')
idx = sampling_adata(detail2, para='neighbors')
detail_down_sampling  = detail2[detail2.index.isin(idx)]
