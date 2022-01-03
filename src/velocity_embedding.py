from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sampling import *
import pandas as pd


def corr_coeff(ematrix, vmatrix, i):
    '''
    Calculate the correlation between the predict velocity (velocity_matrix[:,i])
    and the difference between a cell and every other (cell_matrix - cell_matrix[:, i])
    '''
    ematrix = ematrix.T
    vmatrix = vmatrix.T
    ematrix = ematrix - ematrix[i, :]
    vmatrix = vmatrix[i, :][None,:]
    ematrix_m = ematrix - ematrix.mean(1)[:, None]
    vmatrix_m = vmatrix - vmatrix.mean(1)[:, None]

    # Sum of squares across rows
    ematrix_ss = (ematrix_m**2).sum(1)
    vmatrix_ss = (vmatrix_m**2).sum(1)
    cor = np.dot(ematrix_m, vmatrix_m.T) / np.sqrt(np.dot(ematrix_ss[:, None],vmatrix_ss[None]))
    return cor.T

def velocity_correlation(cell_matrix, velocity_matrix):
    """Calculate the correlation between the predict velocity (velocity_matrix[:,i])
    and the difference between a cell and every other (cell_matrix - cell_matrix[:, i])
    
    Arguments
    ---------
    cell_matrix: np.ndarray (ngenes, ncells)
        gene expression matrix
    velocity_matrix: np.ndarray (ngenes, ncells)
    Return
    ---------
    c_matrix: np.ndarray (ncells, ncells)
    """
    c_matrix = np.zeros((cell_matrix.shape[1], velocity_matrix.shape[1]))
    for i in range(cell_matrix.shape[1]):
        c_matrix[i, :] = corr_coeff(cell_matrix, velocity_matrix, i)[0,:]
    np.fill_diagonal(c_matrix, 0)
    return c_matrix


def velocity_projection(cell_matrix, velocity_matrix, embedding, knn_embedding):
    '''
    cell_matrix: np.ndarray (ngenes, ncells)
        gene expression matrix
    velocity_matrix: np.ndarray (ngenes, ncells)
    '''
    sigma_corr = 0.05
    corrcoef = velocity_correlation(cell_matrix, velocity_matrix)
    probability_matrix = np.exp(corrcoef / sigma_corr)*knn_embedding.A
    probability_matrix /= probability_matrix.sum(1)[:, None]
    unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)
        np.fill_diagonal(unitary_vectors[0, ...], 0)
        np.fill_diagonal(unitary_vectors[1, ...], 0)
    velocity_embedding = (probability_matrix * unitary_vectors).sum(2)
    velocity_embedding -= (knn_embedding.A * unitary_vectors).sum(2) / knn_embedding.sum(1).A.T  # embedding_knn.A * 
    velocity_embedding = velocity_embedding.T
    return velocity_embedding



config = pd.read_csv('/Users/guangyuwang/OneDrive - Houston Methodist/Work/cellDancer/data/neighbor/config/inverse_SGD_costV2.csv', sep=';',header=None)
data_source = config.iloc[0][0]
platform = config.iloc[0][1]
epoches=[int(config.iloc[0][2])]
num_jobs=int(config.iloc[0][3])
learning_rate=float(config.iloc[0][4])
cost_version=int(config.iloc[0][5])
cost1_ratio=float(config.iloc[0][6])
cost2_cutoff=float(config.iloc[0][7])
downsample_method=config.iloc[0][8]
downsample_target_amount=int(config.iloc[0][9])
step_i=int(config.iloc[0][10])
step_j=int(config.iloc[0][11])
sampling_ratio=float(config.iloc[0][12])
n_neighbors=int(config.iloc[0][13])
optimizer=config.iloc[0][14] #["SGD","Adam"]

gene_choice=["Ntrk2","Tmem163"]

raw_data_path="/Users/guangyuwang/OneDrive - Houston Methodist/Work/cellDancer/data/neighbor/denGyr_test_2.csv" 
load_raw_data=pd.read_csv(raw_data_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])
data_df=load_raw_data[['gene_list', 'u0','s0','cellID','embedding1','embedding2']][load_raw_data.gene_list.isin(gene_choice)]
embedding_downsampling, sampling_ixs, knn_embedding = downsampling_embedding(data_df,
                    para=downsample_method,
                    target_amount=downsample_target_amount,
                    step_i=25,
                    step_j=25,
                    n_neighbors=15)




embedding = np.loadtxt(open("/Users/guangyuwang/OneDrive - Houston Methodist/Work/cellDancer/data/loom/OneDrive_1_12-27-2021/vlm_embedding.csv", "rb"), delimiter=",")
hi_dim = np.loadtxt(open("/Users/guangyuwang/OneDrive - Houston Methodist/Work/cellDancer/data/loom/OneDrive_1_12-27-2021/vlm_Sx_sz.csv", "rb"), delimiter=",")
delta_S = np.loadtxt(open("/Users/guangyuwang/OneDrive - Houston Methodist/Work/cellDancer/data/loom/OneDrive_1_12-27-2021/vlm_delta_S.csv", "rb"), delimiter=",")
delta_embedding = np.loadtxt(open("/Users/guangyuwang/OneDrive - Houston Methodist/Work/cellDancer/data/loom/OneDrive_1_12-27-2021/vlm_delta_embedding.csv", "rb"), delimiter=",")
delta_embedding_random = np.loadtxt(open("/Users/guangyuwang/OneDrive - Houston Methodist/Work/cellDancer/data/loom/OneDrive_1_12-27-2021/vlm_delta_embedding_random.csv", "rb"), delimiter=",")
used_delta_t = float(0.5)
psc = 1

hi_dim_t = hi_dim + used_delta_t * delta_S  # [:, :ndims] [:, :ndims]
delta_hi_dim = hi_dim_t - hi_dim # (2159, 18140)
dmatrix = np.sqrt(np.abs(delta_hi_dim) + psc) * np.sign(delta_hi_dim) # (2159, 18140)

embedding_downsampling, sampling_ixs, knn_embedding = downsampling_embedding(data_df,
                    para=downsample_method,
                    target_amount=downsample_target_amount,
                    step_i=20,
                    step_j=20,
                    n_neighbors=10)
A = hi_dim[:, sampling_ixs]
B = dmatrix[:, sampling_ixs]
velocity_embedding = velocity_projection(A, B, embedding[sampling_ixs,:], knn_embedding)

plt.scatter(embedding[sampling_ixs, 0], embedding[sampling_ixs, 1])
plt.quiver(embedding[sampling_ixs, 0], embedding[sampling_ixs, 1],
           velocity_embedding[:, 0], velocity_embedding[:, 1] ,color='red')

delta_embedding2 = np.loadtxt(open("/Users/guangyuwang/OneDrive - Houston Methodist/Work/cellDancer/data/loom/OneDrive_1_12-27-2021/vlm_delta_embedding.csv", "rb"), delimiter=",")

plt.scatter(embedding[sampling_ixs, 0], embedding[sampling_ixs, 1])
plt.quiver(embedding[sampling_ixs, 0], embedding[sampling_ixs, 1],
           delta_embedding2[sampling_ixs, 0], delta_embedding2[sampling_ixs, 1] ,color='red')








