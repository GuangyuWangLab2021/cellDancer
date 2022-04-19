from velocity_plot import velocityPlot as pl
from turtle import color
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sampling import *
import pandas as pd
from colormap import *
import random
import os

#############################################
############### velocity_embedding ##########
#############################################


def corr_coeff(ematrix, vmatrix, i):
    '''
    Calculate the correlation between the predict velocity (velocity_matrix[:,i])
    and the difference between a cell and every other (cell_matrix - cell_matrix[:, i])
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


def data_reshape(load_cellDancer):
    '''
    load detail file
    return expression matrix and velocity (ngenes, ncells)
    '''
    psc = 1
    gene_names = load_cellDancer['gene_name'].drop_duplicates().to_list()
    # cell_number = load_cellDancer[load_cellDancer['gene_name']==gene_names[0]].shape[0]
    # load_cellDancer['index'] = np.tile(range(cell_number),len(gene_names))
    load_cellDancer['index'] = 0
    for g in gene_names:
        load_cellDancer.loc[load_cellDancer['gene_name'] == g, 'index'] = range(
            load_cellDancer[load_cellDancer['gene_name'] == g].shape[0])
    s0_reshape = load_cellDancer.pivot(
        index='gene_name', values='s0', columns='index')
    s1_reshape = load_cellDancer.pivot(
        index='gene_name', values='s1', columns='index')
    dMatrix = s1_reshape-s0_reshape
    np_s0_reshape = np.array(s0_reshape)
    np_dMatrix = np.array(dMatrix)
    np_dMatrix2 = np.sqrt(np.abs(np_dMatrix) + psc) * \
        np.sign(np_dMatrix)  # (2159, 18140)
    return(np_s0_reshape, np_dMatrix2)


raw_data_path = "/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full.csv"
# raw_data_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Melanoma/data/cellDancer/data/input_data.csv'
load_raw_data = pd.read_csv(raw_data_path, names=[
                            'gene_list', 'u0', 's0', "clusters", 'cellID', 'embedding1', 'embedding2'])

# gene choice
# gene_choice=list(set(load_raw_data.gene_list))

# add_amt=0.006
gene_cost = pd.read_csv(
    '/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/gene_cost.csv')
# gene_choice=gene_cost[(gene_cost.cost>0.075-add_amt) & (gene_cost.cost<0.075+add_amt)] # 512 genes
# gene_choice=gene_choice.gene_name
gene_cost = gene_cost.sort_values(by=['cost'])
gene_cost = gene_cost.reset_index()

n_neighbors_list = [50, 100, 150, 200, 250, 300]
add_amt_gene_list = [200, 400, 600, 800, 1200, 1600, 2000]

detail_result_path = '/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene'
output_path = '/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/velocity_embedding_para_tune'


def run_velocity_embedding(n_neighbors, add_amt_gene, load_raw_data, detail_result_path, output_path):
    gene_choice = gene_cost[100:100+add_amt_gene]['gene_name']

    # end gene choice
    # end tuning

    data_df = load_raw_data[['gene_list', 'u0', 's0', 'cellID',
                             'embedding1', 'embedding2']][load_raw_data.gene_list.isin(gene_choice)]
    random.seed(10)
    embedding_downsampling, sampling_ixs, knn_embedding = downsampling_embedding(data_df,
                                                                                 para='neighbors',
                                                                                 target_amount=0,
                                                                                 step_i=60,
                                                                                 step_j=60,
                                                                                 n_neighbors=n_neighbors)

    print(embedding_downsampling)

    for i in range(301, 312):
        print(i)

        # velocyto
        # load_cellDancer=pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/20220117_adjusted_gene_choice_order/denGyrLr0.001Costv3C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/detail_e'+str(i)+'.csv')

        # mel
        # load_cellDancer=pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/20220120malFirst/malLr0.001Costv3C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/detail_e'+str(i)+'.csv')

        # velocyto - cost=1
        load_cellDancer = pd.read_csv(os.path.join(
            detail_result_path, ('detail_e'+str(i)+'.csv')))
        load_cellDancer = load_cellDancer[load_cellDancer.gene_name.isin(
            gene_choice)]

        np_s0, np_dMatrix = data_reshape(load_cellDancer)  # 2min for 200 genes
        print(np_s0.shape)
        print(np_dMatrix.shape)
        if i == 301:
            np_dMatrix_all = np_dMatrix
            np_s0_all = np_s0
        else:
            np_dMatrix_all = np.vstack((np_dMatrix_all, np_dMatrix))
            np_s0_all = np.vstack((np_s0_all, np_s0))
        print(np_dMatrix_all.shape)
        print(np_s0_all.shape)
    # end Generate and Combine np_dMatrix_all and np_s0_all
    # np_s0_all=np.load('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/s0_all.npy',allow_pickle=True)
    # np_dMatrix_all=np.load('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/dMatrix_all.npy',allow_pickle=True)
    embedding_df = load_raw_data[load_raw_data.gene_list == list(
        load_raw_data.gene_list[0])[0]][['embedding1', 'embedding2']]

    embedding = load_raw_data[load_raw_data.gene_list == list(
        load_raw_data.gene_list[0])[0]][['embedding1', 'embedding2']].to_numpy()
    velocity_embedding = velocity_projection(
        np_s0_all[:, sampling_ixs], np_dMatrix_all[:, sampling_ixs], embedding[sampling_ixs, :], knn_embedding)
    # build velocity embedding dataframe
    name_embedding1 = 'embedding1_n' + \
        str(n_neighbors)+"_"+'gAmt'+str(add_amt_gene)
    name_embedding2 = 'embedding2_n' + \
        str(n_neighbors)+"_"+'gAmt'+str(add_amt_gene)
    velocity_embedding_df = pd.DataFrame(velocity_embedding, columns=[
                                         name_embedding1, name_embedding2])
    velocity_embedding_df.index = sampling_ixs
    velocity_embedding_df = pd.concat(
        [embedding_df.iloc[sampling_ixs, :], velocity_embedding_df], axis=1)
    velocity_embedding_df.to_csv(os.path.join(output_path, ('velocity_embedding_tune'+'_n'+str(
        n_neighbors)+"_"+'gAmt'+str(add_amt_gene)+'.csv')), header=True, index=True)


n_neighbors_list = [50, 100, 150, 200, 250, 300]
add_amt_gene_list = [200, 400, 600, 800, 1200, 1600, 2000]
for n_neighbors in n_neighbors_list:
    print('n_neighbors'+str(n_neighbors))
    for add_amt_gene in add_amt_gene_list:
        print('add_amt_gene'+str(add_amt_gene))
        run_velocity_embedding(n_neighbors, add_amt_gene,
                               load_raw_data, detail_result_path, output_path)

#############################################
############ END velocity_embedding #########
#############################################

# comine the csv for velocity embedding
n_neighbors_list = [50, 100, 150, 200, 250, 300]
add_amt_gene_list = [200, 400, 600, 800, 1200, 1600, 2000]
velocity_embedding_df = pd.DataFrame()
for n_neighbors in n_neighbors_list:
    for add_amt_gene in add_amt_gene_list:
        name_embedding1 = 'embedding1_n' + \
            str(n_neighbors)+"_"+'gAmt'+str(add_amt_gene)
        name_embedding2 = 'embedding2_n' + \
            str(n_neighbors)+"_"+'gAmt'+str(add_amt_gene)
        embedding_path = '/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/velocity_embedding_para_tune/velocity_embedding_tune_n' + \
            str(n_neighbors)+'_gAmt'+str(add_amt_gene)+'.csv'
        load_embedding_data = pd.read_csv(embedding_path)[
            [name_embedding1, name_embedding2]]
        velocity_embedding_df = pd.concat(
            [velocity_embedding_df, load_embedding_data], axis=1)
        # test=pd.read_csv(embedding_path)
        # plt.figure()
        # plt.quiver(test.embedding1, test.embedding2,
        #     test[name_embedding1], test[name_embedding2],color='red')
        # plt.savefig('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/velocity_embedding_para_tune/velocity_embedding_tune_n'+str(n_neighbors)+'_gAmt'+str(add_amt_gene)+'.pdf')

# calculate correlation
n_neighbors_list = [50, 100, 150, 200, 250, 300]
add_amt_gene_list = [200, 400, 600, 800, 1200, 1600, 2000]


def get_corrcoef(n_neighbors_list, add_amt_gene_list, n_neighbors, add_amt_gene, n_neighbors_compare, add_amt_gene_compare):
    n_neighbors_list
    corrcoef_pearson_df = pd.DataFrame(None, index=n_neighbors_list,
                                       columns=add_amt_gene_list)

    for n_neighbors in n_neighbors_list:
        for add_amt_gene in add_amt_gene_list:
            name_embedding1 = 'embedding1_n' + \
                str(n_neighbors)+"_"+'gAmt'+str(add_amt_gene)
            name_embedding2 = 'embedding2_n' + \
                str(n_neighbors)+"_"+'gAmt'+str(add_amt_gene)
            embedding_path = '/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/velocity_embedding_para_tune/velocity_embedding_tune_n' + \
                str(n_neighbors)+'_gAmt'+str(add_amt_gene)+'.csv'
            load_embedding_data = pd.read_csv(embedding_path)[
                [name_embedding1, name_embedding2]].to_numpy()
            load_embedding_data_np = np.append(
                load_embedding_data[:, 0], load_embedding_data[:, 1])

            name_embedding1_compare = 'embedding1_n' + \
                str(n_neighbors_compare)+"_"+'gAmt'+str(add_amt_gene_compare)
            name_embedding2_compare = 'embedding2_n' + \
                str(n_neighbors_compare)+"_"+'gAmt'+str(add_amt_gene_compare)
            embedding_path_compare = '/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/velocity_embedding_para_tune/velocity_embedding_tune_n' + \
                str(n_neighbors_compare)+'_gAmt' + \
                str(add_amt_gene_compare)+'.csv'
            load_embedding_data_compare = pd.read_csv(embedding_path_compare)[
                [name_embedding1_compare, name_embedding2_compare]].to_numpy()
            load_embedding_data_compare_np = np.append(
                load_embedding_data_compare[:, 0], load_embedding_data_compare[:, 1])
            corrcoef_pearson = np.corrcoef(
                load_embedding_data_np, load_embedding_data_compare_np)[0, 1]

            corrcoef_pearson_df.at[n_neighbors,
                                   add_amt_gene] = corrcoef_pearson
    return(corrcoef_pearson_df)


n_neighbors_compare = 300
add_amt_gene_compare = 2000
corrcoef_df = get_corrcoef(n_neighbors_list, add_amt_gene_list,
                           n_neighbors, add_amt_gene, n_neighbors_compare, add_amt_gene_compare)

# draft cell velocity

# velocyto - cost=1
# save_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/cell_velocity.pdf'
# mel
# save_path='output/cell_level_velocity/test2_mal2.pdf'
save_path = None

# end draft cell velocity

# colorful cell velocity
# save_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/cell_velocity_colorful_all_points_'+str(fig_no)+'.pdf'
raw_data_path = "/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full.csv"
# raw_data_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Melanoma/data/cellDancer/data/input_data.csv'
load_raw_data = pd.read_csv(raw_data_path, names=[
                            'gene_list', 'u0', 's0', "clusters", 'cellID', 'embedding1', 'embedding2'])

for n_neighbors in n_neighbors_list:
    for add_amt_gene in add_amt_gene_list:
        # save_path='/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/velocity_embedding_para_tune/velocity_embedding_tune_n'+str(n_neighbors)+'_gAmt'+str(add_amt_gene)+'_colorful_arrow.pdf'
        save_path = '/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/velocity_embedding_para_tune/velocity_embedding_tune_n' + \
            str(n_neighbors)+'_gAmt'+str(add_amt_gene) + \
            '_colorful_grid_curve_arrow.pdf'

        embedding = load_raw_data[load_raw_data.gene_list == list(
            load_raw_data.gene_list[0])[0]][['embedding1', 'embedding2']].to_numpy()
        name_embedding1 = 'embedding1_n' + \
            str(n_neighbors)+"_"+'gAmt'+str(add_amt_gene)
        name_embedding2 = 'embedding2_n' + \
            str(n_neighbors)+"_"+'gAmt'+str(add_amt_gene)
        embedding_path = '/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/velocity_embedding_para_tune/velocity_embedding_tune_n' + \
            str(n_neighbors)+'_gAmt'+str(add_amt_gene)+'.csv'
        load_embedding_velocity_data = pd.read_csv(embedding_path, index_col=0)
        sampling_ixs = load_embedding_velocity_data.index
        velocity_embedding = load_embedding_velocity_data[[
            name_embedding1, name_embedding2]].to_numpy()
        custom_xlim = (-23, 45)
        pl.velocity_cell_map(load_raw_data, embedding, sampling_ixs, velocity_embedding,
                                    save_path=save_path, curve_grid=True, custom_xlim=custom_xlim)
        # pl.velocity_cell_map_draft(embedding,sampling_ixs,velocity_embedding,save_path=save_path)

# end colorful cell velocity
