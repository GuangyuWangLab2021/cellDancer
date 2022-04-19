from turtle import color
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sampling import *
import pandas as pd
from colormap import *

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
    # cell_matrix = np_s0[:,sampling_ixs]
    # velocity_matrix = np_dMatrix[:,sampling_ixs]
    sigma_corr = 0.05
    cell_matrix[np.isnan(cell_matrix)]=0
    velocity_matrix[np.isnan(velocity_matrix)]=0
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
        load_cellDancer.loc[load_cellDancer['gene_name']==g, 'index'] = range(load_cellDancer[load_cellDancer['gene_name']==g].shape[0])
    s0_reshape = load_cellDancer.pivot(index='gene_name', values='s0', columns='index')
    s1_reshape = load_cellDancer.pivot(index='gene_name', values='s1', columns='index')
    dMatrix = s1_reshape-s0_reshape
    np_s0_reshape = np.array(s0_reshape)
    np_dMatrix = np.array(dMatrix)
    np_dMatrix2 = np.sqrt(np.abs(np_dMatrix) + psc) * np.sign(np_dMatrix) # (2159, 18140)
    return(np_s0_reshape, np_dMatrix2)


raw_data_path="/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full.csv" 
# raw_data_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Melanoma/data/cellDancer/data/input_data.csv'
load_raw_data=pd.read_csv(raw_data_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])
gene_choice=list(set(load_raw_data.gene_list))

data_df=load_raw_data[['gene_list', 'u0','s0','cellID','embedding1','embedding2']][load_raw_data.gene_list.isin(gene_choice)]
embedding_downsampling, sampling_ixs, knn_embedding = downsampling_embedding(data_df,
                    para='neighbors',
                    target_amount=0,
                    step_i=60,
                    step_j=60,
                    n_neighbors=100)

np_s0_all=np.load('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/s0_all.npy',allow_pickle=True)
np_dMatrix_all=np.load('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/dMatrix_all.npy',allow_pickle=True)

embedding=load_raw_data[load_raw_data.gene_list==list(load_raw_data.gene_list[0])[0]][['embedding1','embedding2']].to_numpy()

velocity_embedding = velocity_projection(np_s0_all[:,sampling_ixs], np_dMatrix_all[:,sampling_ixs], embedding[sampling_ixs,:], knn_embedding)

################## draft cell velocity
# load_cellDancer['u1'].isnull().values.any()
def cell_velocity_plot_draft():
    plt.figure()
    plt.scatter(embedding[sampling_ixs, 0], embedding[sampling_ixs, 1],s=0.5)
    plt.quiver(embedding[sampling_ixs, 0], embedding[sampling_ixs, 1],
            velocity_embedding[:, 0], velocity_embedding[:, 1] ,color='red')

    # velocyto - cost=1
    plt.savefig('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/cell_velocity.pdf')
    # mel
    plt.savefig('output/cell_level_velocity/test2_mal2.pdf')
################## end draft cell velocity

################### colorful cell velocity
def cell_velocity_plot_colorful():
    colors = {'CA':grove2[7],
    'CA1-Sub':grove2[9],
    'CA2-3-4':grove2[8],
    'Granule':grove2[6],
    'ImmGranule1':grove2[6],
    'ImmGranule2':grove2[6],
    'Nbl1':grove2[5],
    'Nbl2':grove2[5],
    'nIPC':grove2[4],
    'RadialGlia':grove2[3],
    'RadialGlia2':grove2[3],
    'GlialProg' :grove2[2],
    'OPC':grove2[1],
    'ImmAstro':grove2[0]}
    pointsize=20
    pointsize=15
    pointsize=5


    one_gene_raw=list(load_raw_data.gene_list[0])[0]

    step_i=35
    step_j=35
    step_i=25
    step_j=25
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    def gen_Line2D(label,markerfacecolor):
        return Line2D([0], [0], color='w',marker='o', label=label,markerfacecolor=markerfacecolor,  markeredgewidth=0, markersize=5)

    legend_elements=[]
    for i in colors:
        legend_elements.append(gen_Line2D(i,colors[i]))

    fig_no=20
    arrow_idx=sampling_neighbors(embedding[sampling_ixs, :], step_i=step_i,step_j=step_j)

    plt.scatter(embedding[:, 0],
                embedding[:, 1],
                c=load_raw_data[load_raw_data.gene_list==one_gene_raw]['clusters'].map(colors),
                s=pointsize,
                # alpha=1,
                #alpha=0.3,
                alpha=0.05,
                # alpha=0.1,
                edgecolor="none")
    plt.xlim(-23, 45)
    # plt.scatter(embedding[sampling_ixs, 0][arrow_idx], embedding[sampling_ixs, 1][arrow_idx], # sampled circle
    #             color="none",s=pointsize, edgecolor="k",linewidths=0.5)

    # arrow all points
    plt.quiver(embedding[sampling_ixs, 0], 
            embedding[sampling_ixs, 1],
            velocity_embedding[:, 0], 
            velocity_embedding[:, 1] ,
            #color='black',
            color=load_raw_data[load_raw_data.gene_list==one_gene_raw]['clusters'][sampling_ixs].map(colors),
            angles='xy', 
            clim=(0., 1.),
            width=0.002, 
            #    alpha=0.3,
            alpha=1,
            linewidth=.2,
            #    edgecolor='white'
            )

    #fig, ax = plt.subplots()
    plt.legend(handles=legend_elements, loc='right',prop={'size': 6})
    fig_no=fig_no+1
    plt.savefig('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/cell_velocity_colorful_all_points_'+str(fig_no)+'.pdf')

################### end colorful cell velocity

################### calculate_grid_arrows
# Source - https://github.com/velocyto-team/velocyto.py/blob/0963dd2df0ac802c36404e0f434ba97f07edfe4b/velocyto/analysis.py
from scipy.stats import norm as normal


######################## kernel gird plot
def calculate_two_end_gird(embedding, sampling_ixs, velocity_embedding, smooth=0.8, steps=(40, 40), min_mass=2):
    def find_neighbors(data, n_neighbors, gridpoints_coordinates):
        # data  = embedding[sampling_ixs, :]
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=8)
        nn.fit(data)
        dists, neighs = nn.kneighbors(gridpoints_coordinates)
        return(dists, neighs)
    # Prepare the grid
    grs = []
    for dim_i in range(embedding[sampling_ixs, :].shape[1]):
        m, M = np.min(embedding[sampling_ixs, :][:, dim_i]), np.max(embedding[sampling_ixs, :][:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, steps[dim_i])
        grs.append(gr)
        
    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T

    n_neighbors = int(velocity_embedding.shape[0]/3)
    dists_head, neighs_head = find_neighbors(embedding[sampling_ixs, :], n_neighbors, gridpoints_coordinates)
    dists_tail, neighs_tail = find_neighbors(embedding[sampling_ixs, :]+velocity_embedding, n_neighbors, gridpoints_coordinates)
    std = np.mean([(g[1] - g[0]) for g in grs])

    # isotropic gaussian kernel
    gaussian_w_head = normal.pdf(loc=0, scale=smooth * std, x=dists_head)
    total_p_mass_head = gaussian_w_head.sum(1)
    gaussian_w_tail = normal.pdf(loc=0, scale=smooth * std, x=dists_tail)
    total_p_mass_tail = gaussian_w_tail.sum(1)

    UZ_head = (velocity_embedding[neighs_head] * gaussian_w_head[:, :, None]).sum(1) / np.maximum(1, total_p_mass_head)[:, None]  # weighed average
    UZ_tail = (velocity_embedding[neighs_tail] * gaussian_w_tail[:, :, None]).sum(1) / np.maximum(1, total_p_mass_tail)[:, None]  # weighed average

    XY = gridpoints_coordinates

    dists_head2, neighs_head2 = find_neighbors(embedding[sampling_ixs, :], n_neighbors, XY+UZ_head)
    dists_tail2, neighs_tail2 = find_neighbors(embedding[sampling_ixs, :], n_neighbors, XY-UZ_tail)

    gaussian_w_head2 = normal.pdf(loc=0, scale=smooth * std, x=dists_head2)
    total_p_mass_head2 = gaussian_w_head2.sum(1)
    gaussian_w_tail2 = normal.pdf(loc=0, scale=smooth * std, x=dists_tail2)
    total_p_mass_tail2 = gaussian_w_tail2.sum(1)

    UZ_head2 = (velocity_embedding[neighs_head2] * gaussian_w_head2[:, :, None]).sum(1) / np.maximum(1, total_p_mass_head2)[:, None]  # weighed average
    UZ_tail2 = (velocity_embedding[neighs_tail2] * gaussian_w_tail2[:, :, None]).sum(1) / np.maximum(1, total_p_mass_tail2)[:, None]  # weighed average

    mass_filter = total_p_mass_head < min_mass

    # filter dots
    UZ_head_filtered = UZ_head[~mass_filter, :]
    UZ_tail_filtered = UZ_tail[~mass_filter, :]
    UZ_head2_filtered = UZ_head2[~mass_filter, :]
    UZ_tail2_filtered = UZ_tail2[~mass_filter, :]
    XY_filtered = XY[~mass_filter, :]
    return(XY_filtered, UZ_head_filtered, UZ_tail_filtered, UZ_head2_filtered, UZ_tail2_filtered,mass_filter,grs)


XY_filtered, UZ_head_filtered, UZ_tail_filtered, UZ_head2_filtered, UZ_tail2_filtered,mass_filter,grs = calculate_two_end_gird(embedding, sampling_ixs, velocity_embedding, smooth=0.8, steps=(30, 30), min_mass=2)

plt.quiver(XY_filtered[:, 0], XY_filtered[:, 1], UZ_head_filtered[:, 0], UZ_head_filtered[:, 1], zorder=20000, color='blue')
plt.quiver(XY_filtered[:, 0]-UZ_tail_filtered[:, 0], XY_filtered[:, 1]-UZ_tail_filtered[:, 1], UZ_tail_filtered[:, 0], UZ_tail_filtered[:, 1], zorder=20000, color='red')
plt.savefig('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/two_end.pdf')

#######################################################
############ connect two end grid to curve ############
#######################################################
n_curves = XY_filtered.shape[0]
s_vals = np.linspace(0.0, 1.5, 15)
############ get longest distance len and norm ratio ############
XYM=XY_filtered
UVT=UZ_tail_filtered
UVH=UZ_head_filtered
UVT2=UZ_tail2_filtered
UVH2=UZ_head2_filtered

def norm_arrow_display_ratio(XYM,UVT,UVH,UVT2,UVH2,grs,s_vals):
    '''get the longest distance in prediction between the five points,
       and normalize by using the distance between two girds'''
    
    def distance (x, y):
        # calc disctnce list between a set of coordinate
        calculate_square=np.subtract(x[0:-1], x[1:])**2 + np.subtract(y[0:-1], y[1:])**2
        distance_result=(calculate_square)**0.5
        return distance_result

    max_discance=0
    for i in range(n_curves):
        nodes = np.asfortranarray([[XYM[i, 0]-UVT[i, 0]-UVT2[i, 0], XYM[i, 0]-UVT[i, 0], XYM[i, 0], XYM[i, 0]+UVH[i, 0], XYM[i, 0]+UVH[i, 0]+UVH2[i, 0]],
                    [XYM[i, 1]-UVT[i, 1]-UVT2[i, 1],XYM[i, 1]-UVT[i, 1], XYM[i, 1], XYM[i, 1]+UVH[i, 1], XYM[i, 1]+UVH[i, 1]+UVH2[i, 1]]])
        curve = bezier.Curve(nodes, degree=4)
        curve_dots = curve.evaluate_multi(s_vals) # 
        distance_sum=np.sum(distance(curve_dots[0], curve_dots[1]))
        max_discance = max(max_discance,distance_sum)
    distance_gird=(abs(grs[0][0]-grs[0][1]) + abs(grs[1][0]-grs[1][1]))/2
    print(max_discance)
    print(distance_gird)
    norm_ratio=distance_gird/max_discance
    print(norm_ratio)
    return(norm_ratio)

norm_ratio=norm_arrow_display_ratio(XYM,UVT,UVH,UVT2,UVH2,grs,s_vals)
############ end --- get longest distance len and norm ratio ############

############ plot the curve arrow for cell velocity ############
import bezier

XYM=XY_filtered
UVT=UZ_tail_filtered * norm_ratio
UVH=UZ_head_filtered * norm_ratio
UVT2=UZ_tail2_filtered * norm_ratio
UVH2=UZ_head2_filtered * norm_ratio

def plot_cell_velocity_curve(XYM,UVT,UVH,UVT2,UVH2,s_vals,save_path=None):
    plt.axis('equal')
    # TO DO: add 'colorful cell velocity' to here, now there is only curve arrows
    for i in range(n_curves):
        nodes = np.asfortranarray([[XYM[i, 0]-UVT[i, 0]-UVT2[i, 0], XYM[i, 0]-UVT[i, 0], XYM[i, 0], XYM[i, 0]+UVH[i, 0], XYM[i, 0]+UVH[i, 0]+UVH2[i, 0]],
                    [XYM[i, 1]-UVT[i, 1]-UVT2[i, 1],XYM[i, 1]-UVT[i, 1], XYM[i, 1], XYM[i, 1]+UVH[i, 1], XYM[i, 1]+UVH[i, 1]+UVH2[i, 1]]])
        curve = bezier.Curve(nodes, degree=4)
        curve_dots = curve.evaluate_multi(s_vals)
        plt.plot(curve_dots[0],curve_dots[1],'k',linewidth=0.5,color='black',alpha=1)

        # normalize the arrow of the last two points at the tail, to let all arrows has the same size in quiver
        U = curve_dots[0][-1]-curve_dots[0][-2]
        V = curve_dots[1][-1]-curve_dots[1][-2]
        N = np.sqrt( U**2 + V**2 )  
        U1, V1 = U/N*0.5, V/N*0.5 # 0.5 is to let the arrow have a suitable size
        plt.quiver(curve_dots[0][-2],curve_dots[1][-2],U1,V1, units='xy',angles='xy',scale=1,linewidth=0,color='black',alpha=1,minlength=0,width=0.1)

    # used to help identify arrow and line
    # plt.quiver(XY_filtered[:, 0], XY_filtered[:, 1], UZ_head_filtered[:, 0], UZ_head_filtered[:, 1], zorder=20000, color='blue',linewidth=0,alpha=0.2)
    # plt.quiver(XY_filtered[:, 0]-UZ_tail_filtered[:, 0], XY_filtered[:, 1]-UZ_tail_filtered[:, 1], UZ_tail_filtered[:, 0], UZ_tail_filtered[:, 1], zorder=20000, color='red',alpha=0.2)
    if save_path is not None:
        plt.savefig()

save_path=('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/curve.pdf')
plot_cell_velocity_curve(XYM,UVT,UVH,UVT2,UVH2,s_vals,save_path)

############ end --- plot the curve arrow for cell velocity ############


######## smooth line to plot curve
'''
from scipy.interpolate import make_interp_spline, BSpline

# 300 represents number of points to make between T.min and T.max
i =4
points = np.array([[XYM[i, 0]-UVT[i, 0]-UVT2[i, 0], XYM[i, 0]-UVT[i, 0], XYM[i, 0], XYM[i, 0]+UVH[i, 0], XYM[i, 0]+UVH[i, 0]+UVH2[i, 0]],
[XYM[i, 1]-UVT[i, 1]-UVT2[i, 1],XYM[i, 1]-UVT[i, 1], XYM[i, 1], XYM[i, 1]+UVH[i, 1], XYM[i, 1]+UVH[i, 1]+UVH2[i, 1]]])

for i in range(n_curves):
    points = np.array([[XYM[i, 0]-UVT[i, 0]-UVT2[i, 0], XYM[i, 0]-UVT[i, 0], XYM[i, 0], XYM[i, 0]+UVH[i, 0], XYM[i, 0]+UVH[i, 0]+UVH2[i, 0]],[XYM[i, 1]-UVT[i, 1]-UVT2[i, 1],XYM[i, 1]-UVT[i, 1], XYM[i, 1], XYM[i, 1]+UVH[i, 1], XYM[i, 1]+UVH[i, 1]+UVH2[i, 1]]])
    points = points[:,np.argsort(points[0,:])]
    xnew = np.linspace(np.array(points[0]).min(), np.array(points[0]).max(), 300) 
    spl = make_interp_spline(points[0,:], points[1,:])  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew, power_smooth)

plt.savefig('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/curve2.pdf')
'''