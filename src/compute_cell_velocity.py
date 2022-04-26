from turtle import color
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import sys
import numpy as np


if __name__ == "__main__":
    sys.path.append('.')
    from sampling import *
    from colormap import *
else:
    try:
        from .sampling import *
        from .colormap import *
    except ImportError:
        from sampling import *
        from colormap import *

####### organize code


def compute_cell_velocity(load_cellDancer,
        gene_list=None,
        n_neighbors=200,
        step=(60,60),
        transfer_mode=None,
        mode=None,
        pca_n_components=None,
        umap_n=None,
        umap_n_components=None,
        use_downsampling=True):

    # mode: [mode='embedding', mode='gene']

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
            c_matrix[i, :] = corr_coeff(cell_matrix, velocity_matrix, i)[0, :]
        np.fill_diagonal(c_matrix, 0)
        return c_matrix


    def velocity_projection(cell_matrix, velocity_matrix, embedding, knn_embedding):
        '''
        cell_matrix: np.ndarray (ngenes, ncells)
            gene expression matrix
        velocity_matrix: np.ndarray (ngenes, ncells)
        '''
        # cell_matrix = np_s0[:,sampling_ixs]
        # velocity_matrix = np_dMatrix[:,sampling_ixs]
        sigma_corr = 0.05
        cell_matrix[np.isnan(cell_matrix)] = 0
        velocity_matrix[np.isnan(velocity_matrix)] = 0
        corrcoef = velocity_correlation(cell_matrix, velocity_matrix)
        probability_matrix = np.exp(corrcoef / sigma_corr)*knn_embedding.A
        probability_matrix /= probability_matrix.sum(1)[:, None]
        unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)
            np.fill_diagonal(unitary_vectors[0, ...], 0)
            np.fill_diagonal(unitary_vectors[1, ...], 0)
        velocity_embedding = (probability_matrix * unitary_vectors).sum(2)
        velocity_embedding -= (knn_embedding.A * unitary_vectors).sum(2) / \
            knn_embedding.sum(1).A.T  # embedding_knn.A *
        velocity_embedding = velocity_embedding.T
        return velocity_embedding


    if gene_list is None:
        gene_choice=load_cellDancer.gene_name.drop_duplicates()
    else:
        gene_choice=gene_list

    load_cellDancer_input = load_cellDancer[load_cellDancer.gene_name.isin(gene_choice)]

    data_df = load_cellDancer_input[['gene_name', 'u0', 's0', 'cellID','embedding1', 'embedding2']]
    # random.seed(10)
    embedding_downsampling, sampling_ixs, knn_embedding = downsampling_embedding(data_df,
                                                                                    para='neighbors',
                                                                                    target_amount=0,
                                                                                    step=step,
                                                                                    n_neighbors=n_neighbors,
                                                                                    mode=mode,
                                                                                    transfer_mode=transfer_mode,
                                                                                    pca_n_components=pca_n_components,
                                                                                    umap_n=umap_n,
                                                                                    umap_n_components=umap_n_components)
    

    np_s0_all,np_dMatrix_all= data_reshape(load_cellDancer)

    print(np_dMatrix_all.shape)
    print(np_s0_all.shape)

    one_gene = load_cellDancer.gene_name[0]
    embedding = load_cellDancer[load_cellDancer.gene_name == one_gene][['embedding1', 'embedding2']]
    embedding = embedding.to_numpy()
    
    # mode only provides neighborlist, use embedding(from raw data) to compute cell velocity
    velocity_embedding = velocity_projection(
            np_s0_all[:, sampling_ixs], 
            np_dMatrix_all[:, sampling_ixs], 
            embedding[sampling_ixs, :], 
            knn_embedding)

    index_gene_choice = load_cellDancer_input[load_cellDancer_input.cellIndex.isin(sampling_ixs)].index
    load_cellDancer.loc[index_gene_choice,'velocity1'] = np.tile(velocity_embedding[:,0], len(gene_choice))
    load_cellDancer.loc[index_gene_choice,'velocity2'] = np.tile(velocity_embedding[:,1], len(gene_choice))

def corr_coeff(ematrix, vmatrix, i):
        '''
        Calculate the correlation between the predict velocity (velocity_matrix[:,i])
        and the displacement between a cell and every other (cell_matrix - cell_matrix[:, i])
        '''
        # ematrix = cell_matrix
        # vmatrix = velocity_matrix
        ematrix = ematrix.T
        vmatrix = vmatrix.T
        ematrix = ematrix - ematrix[i, :]
        vmatrix = vmatrix[i, :][None, :]
        ematrix_m = ematrix - ematrix.mean(1)[:, None]
        vmatrix_m = vmatrix - vmatrix.mean(1)[:, None]

        # Sum of squares across rows
        ematrix_ss = (ematrix_m**2).sum(1)
        vmatrix_ss = (vmatrix_m**2).sum(1)
        cor = np.dot(ematrix_m, vmatrix_m.T) / \
            np.sqrt(np.dot(ematrix_ss[:, None], vmatrix_ss[None]))
        
        return cor.T

def data_reshape(load_cellDancer): # pengzhi version
    '''
    load detail file
    return expression matrix and velocity (ngenes, ncells)
    '''
    psc = 1
    gene_names = load_cellDancer['gene_name'].drop_duplicates().to_list()
    # PZ uncommented this.
    cell_number = load_cellDancer[load_cellDancer['gene_name']==gene_names[0]].shape[0]
    load_cellDancer['index'] = np.tile(range(cell_number),len(gene_names))

    s0_reshape = load_cellDancer.pivot(
        index='gene_name', values='s0', columns='index')
    s1_reshape = load_cellDancer.pivot(
        index='gene_name', values='s1', columns='index')
    dMatrix = s1_reshape-s0_reshape
    np_s0_reshape = np.array(s0_reshape)
    np_dMatrix = np.array(dMatrix)
    np_dMatrix2 = np.sqrt(np.abs(np_dMatrix) + psc) * \
        np.sign(np_dMatrix)
    return(np_s0_reshape, np_dMatrix2)

