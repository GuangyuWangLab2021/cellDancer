import numpy as np
from numpy.core.fromnumeric import size
from sklearn.neighbors import NearestNeighbors
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import scvelo as scv




def sampling_neighbors(gene_u0_s0,step_i=30,step_j=30): # current version will obtain ~100 cells. e.g. Ntrk2:109; Tmem163:104
    #step 250 will got 4000 from den data 
    from scipy.stats import norm
    def gaussian_kernel(X, mu = 0, sigma=1):
        return np.exp(-(X - mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    steps = step_i, step_j
    # print(steps)
    grs = []
    # print(gene_u0_s0.shape[1])
    for dim_i in range(gene_u0_s0.shape[1]):
        m, M = np.min(gene_u0_s0[:, dim_i]), np.max(gene_u0_s0[:, dim_i])
        # print(m, M)
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, steps[dim_i])
        grs.append(gr)
    # print(grs)
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


def sampling_inverse(gene_u0_s0,target_amount=500):
    u0 = gene_u0_s0[:,0]
    s0 = gene_u0_s0[:,1]
    values = np.vstack([u0,s0])
    kernel = scipy.stats.gaussian_kde(values)
    p = kernel(values)
    # p2 = (1/p)/sum(1/p)
    p2 = (1/p)/sum(1/p)
    idx = np.arange(values.shape[1])
    r = scipy.stats.rv_discrete(values=(idx, p2))
    idx_choice = r.rvs(size=target_amount)
    return(idx_choice)

def sampling_circle(gene_u0_s0,target_amount=500):
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
    idx_choice = r.rvs(size=target_amount)
    return(idx_choice)

def sampling_random(gene_u0_s0, target_amount=500):
    idx = np.random.choice(gene_u0_s0.shape[0], size = target_amount, replace=False)
    return(idx)
    

def sampling_adata(detail, 
                    para,
                    target_amount=500,
                    step_i=30,
                    step_j=30):
    if para == 'neighbors':
        data_U_S= np.array(detail[["u0","s0"]])
        idx = sampling_neighbors(data_U_S,step_i,step_j)
    elif para == 'inverse':
        data_U_S= np.array(detail[["u0","s0"]])
        idx = sampling_inverse(data_U_S,target_amount)
    elif para == 'circle':
        data_U_S= np.array(detail[["u0","s0"]])
        idx = sampling_circle(data_U_S,target_amount)
    elif para == 'random':
        data_U_S= np.array(detail[["u0","s0"]])
        idx = sampling_random(data_U_S,target_amount)
    else:
        print('para is neighbors or inverse or circle')
    return(idx)

def sampling_embedding(detail, 
                    para,
                    target_amount=500,
                    step_i=30,
                    step_j=30):

    '''
    Guangyu
    '''
    if para == 'neighbors':
        data_U_S= np.array(detail[["embedding1","embedding2"]])
        idx = sampling_neighbors(data_U_S,step_i,step_j)
    elif para == 'inverse':
        print('inverse')
        data_U_S= np.array(detail[["embedding1","embedding2"]])
        idx = sampling_inverse(data_U_S,target_amount)
    elif para == 'circle':
        data_U_S= np.array(detail[["embedding1","embedding2"]])
        idx = sampling_circle(data_U_S,target_amount)
    elif para == 'random':
        # print('random')
        data_U_S= np.array(detail[["embedding1","embedding2"]])
        idx = sampling_random(data_U_S,target_amount)
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


def downsampling_embedding(data_df,para,target_amount,step_i,step_j, n_neighbors):
    '''
    Guangyu
    sampling cells by embedding
    return: sampled embedding, the indexs of sampled cells, and the neighbors of sampled cells
    '''

    gene = data_df['gene_list'].drop_duplicates().iloc[0]
    embedding = data_df.loc[data_df['gene_list']==gene][['embedding1','embedding2']]
    print(para)
    idx_downSampling_embedding = sampling_embedding(embedding,
                para=para,
                target_amount=target_amount,
                step_i=step_i,
                step_j=step_j
                )
    embedding_downsampling = embedding.iloc[idx_downSampling_embedding][['embedding1','embedding2']]
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(embedding_downsampling)  # NOTE should support knn in high dimensions
    embedding_knn = nn.kneighbors_graph(mode="connectivity")
    # neighbor_ixs = embedding_knn.indices.reshape((-1, n_neighbors + 1))
    return(embedding_downsampling, idx_downSampling_embedding, embedding_knn)

def downsampling(data_df, gene_choice, downsampling_ixs):
    '''
    Guangyu
    '''
    data_df_downsampled=pd.DataFrame()
    for gene in gene_choice:
        data_df_one_gene=data_df[data_df['gene_list']==gene]
        data_df_one_gene_downsampled = data_df_one_gene.iloc[downsampling_ixs]
        data_df_downsampled=data_df_downsampled.append(data_df_one_gene_downsampled)

        # plt.scatter(data_df_one_gene['embedding1'], data_df_one_gene['embedding2'])
        # plt.scatter(data_df_one_gene.iloc[downsampling_ixs]['embedding1'], data_df_one_gene.iloc[downsampling_ixs]['embedding2'])
        # plt.scatter(embedding_downsampling.iloc[neighbor_ixs[0,:]]['embedding1'], embedding_downsampling.iloc[neighbor_ixs[0,:]]['embedding2'])
        # plt.scatter(embedding_downsampling.iloc[0]['embedding1'], embedding_downsampling.iloc[0]['embedding2'])
        # plt.show()
    return(data_df_downsampled)


# old version
# def downsampling(data_df,gene_choice,para,target_amount,step_i,step_j):
#     data_df_downsampled=pd.DataFrame()
#     for gene in gene_choice:
#         data_df_one_gene=data_df[data_df['gene_list']==gene]
#         idx = sampling_adata(data_df_one_gene, 
#                                 para=para,
#                                 target_amount=target_amount,
#                                 step_i=step_i,
#                                 step_j=step_j)
#         data_df_one_gene_downsampled = data_df_one_gene[data_df_one_gene.index.isin(idx)]
#         data_df_downsampled=data_df_downsampled.append(data_df_one_gene_downsampled)
#     return(data_df_downsampled)
