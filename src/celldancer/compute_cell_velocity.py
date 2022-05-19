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
else:
    try:
        from .sampling import *
    except ImportError:
        from sampling import *

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
    

    #no na
    is_NaN = load_cellDancer[['alpha','beta']].isnull()
    row_has_NaN = is_NaN. any(axis=1)
    load_cellDancer = load_cellDancer[~row_has_NaN].reset_index(drop=True)
    

    # load_cellDancer_copy['s0Max'] = load_cellDancer_copy.groupby('gene_name')['s0'].transform(lambda x: x.max())
    # load_cellDancer_copy.loc[:,'s0'] = (load_cellDancer_copy.s0/load_cellDancer_copy.s0Max)*4.5
    # load_cellDancer_copy.loc[:,'s1'] = (load_cellDancer_copy.s1/load_cellDancer_copy.s0Max)*4.5

    if gene_list is None:
        gene_list=load_cellDancer.gene_name.drop_duplicates()
    else:
        print("Selected genes: ", gene_list)


    # This creates a new dataframe
    load_cellDancer_input = load_cellDancer[
            load_cellDancer.gene_name.isin(gene_list)].reset_index(drop=True)

    np_s0_all, np_dMatrix_all= data_reshape(load_cellDancer_input)
    print("(genes, cells): ", end="")
    print(np_s0_all.shape)
    n_genes, n_cells = np_s0_all.shape

    # This creates a new dataframe
    data_df = load_cellDancer_input.loc[:, 
            ['gene_name', 'u0', 's0', 'cellID','embedding1', 'embedding2']]
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
    

    # mode only provides neighborlist, use embedding(from raw data) to compute cell velocity
    embedding = load_cellDancer_input[load_cellDancer_input.gene_name == 
            gene_list[0]][['embedding1', 'embedding2']]
    embedding = embedding.to_numpy()
    velocity_embedding = velocity_projection(
            np_s0_all[:, sampling_ixs], 
            np_dMatrix_all[:, sampling_ixs], 
            embedding[sampling_ixs, :], 
            knn_embedding)

    if set(['velocity1','velocity2']).issubset(load_cellDancer.columns):
        print("Caution! Overwriting the \'velocity\' columns.") 
        load_cellDancer.drop(['velocity1','velocity2'], axis=1, inplace=True)

    sampling_ixs_all_genes = load_cellDancer_input[load_cellDancer_input.cellIndex.isin(sampling_ixs)].index
    load_cellDancer_input.loc[sampling_ixs_all_genes,'velocity1'] = np.tile(velocity_embedding[:,0], n_genes)
    load_cellDancer_input.loc[sampling_ixs_all_genes,'velocity2'] = np.tile(velocity_embedding[:,1], n_genes)
    print("After downsampling, there are ", len(sampling_ixs), "cells.")
    return(load_cellDancer_input)

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
        cor = np.dot(ematrix_m, vmatrix_m.T)
        N = np.sqrt(np.dot(ematrix_ss[:, None], vmatrix_ss[None]))
        cor=np.divide(cor, N, where=N!=0)
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

