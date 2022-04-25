import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import norm as normal
import bezier
import numpy as np
import pandas as pd

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


# todo: add cluster color
# todo: plt.show() for all plot functions
# todo: build map itself    
def scatter_cell(
        load_cellDancer, 
        save_path=None,
        custom_xlim=None,custom_ylim=None,
        colors=None, alpha=0.5,
        velocity=False,
        #add_amt_gene=2000, gene_name=None,
        #mode='embedding',pca_n_components=4,file_name_additional_info='',umap_n=10,transfer_mode=None,umap_n_components=None,
        min_mass=2,grid_steps=(30,30)): 


    """Cell velocity plot.
    TO DO: load_cellDancer contains the cluster information, needs improve
    
    .. image:: https://user-images.githubusercontent.com/31883718/67709134-a0989480-f9bd-11e9-8ae6-f6391f5d95a0.png
    

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    density: `float` (default: 1)
        Amount of velocities to show - 0 none to 1 all
    arrow_size: `float` or triple `headlength, headwidth, headaxislength` (default: 1)
        Size of arrows.
    arrow_length: `float` (default: 1)
        Length of arrows.
    scale: `float` (default: 1)
        Length of velocities in the embedding.
    {scatter}

    Returns
    -------
    `matplotlib.Axis` if `show==False`
    """

    # colors = dict() --> specified colormap constructed based on the dict
    # colors = 'pseudotime'  --> a defined colormap by us, and pseudotime as c.
    # colors = list(cluster) --> cycle a colormap of size n_clusters
    # colors = None --> grey

    # The colors should be one of those:
    #   a dictionary of {cluster:color}
    #   a string of the column attribute (pseudotime, alpha, beta etc)
    #   a list of clusters 
    #   None

    # if isinstance(colors, dict):
    #     colors = colors
    #     c = xx
    #     cmap = yy
    # elif isinstance(colors, str):
    #     colors = some colormap
    #     c = extract_from_df(load_cellDancer, ['colors'])
    # elif isinstance(colors, list):
    #     colors = cycling a default colormap
    # elif colors is None:
    #     colors=  grey color
    #     # 
    #     # colors = generate_colormap(load_celldancer)
    #     colors = {'CA': grove2[7],
    #     'CA1-Sub': grove2[9],
    #     'CA2-3-4': grove2[8],
    #     'Granule': grove2[6],
    #     'ImmGranule1': grove2[6],
    #     'ImmGranule2': grove2[6],
    #     'Nbl1': grove2[5],
    #     'Nbl2': grove2[5],
    #     'nIPC': grove2[4],
    #     'RadialGlia': grove2[3],
    #     'RadialGlia2': grove2[3],
    #     'GlialProg': grove2[2],
    #     'OPC': grove2[1],
    #     'ImmAstro': grove2[0]}
    pointsize = 5


    # PENGZHI
    # I think it is probably a better idea 
    # to add a column in load_cellDancer to indicate sampled or not.

    embedding = extract_from_df(load_cellDancer, ['embedding1', 'embedding2'])
    embedding_ds = extract_from_df(load_cellDancer, ['embedding_1', 'embedding_2'])
    velocity_embedding= extract_from_df(load_cellDancer, ['velocity_1', 'velocity_2'])


    def gen_Line2D(label, markerfacecolor):
        return Line2D([0], [0], color='w', marker='o', label=label, markerfacecolor=markerfacecolor,  markeredgewidth=0, markersize=5)

    legend_elements = []
    for i in colors:
        legend_elements.append(gen_Line2D(i, colors[i]))

    plt.figure()
    # TODO: color
    plt.scatter(embedding[:, 0],
                embedding[:, 1],
                #c=load_cellDancer[one_gene_idx]['clusters'].map(colors),
                #s=pointsize,
                #alpha=alpha,
                edgecolor="none")

    # calculate_grid_arrows
    # Source - https://github.com/velocyto-team/velocyto.py/blob/0963dd2df0ac802c36404e0f434ba97f07edfe4b/velocyto/analysis.py
    def grid_curve(embedding_ds, velocity_embedding):
        # kernel grid plot

        def calculate_two_end_grid(embedding_ds, velocity_embedding, smooth=None, steps=None, min_mass=None):
            # Prepare the grid
            grs = []
            for dim_i in range(embedding_ds.shape[1]):
                m, M = np.min(embedding_ds[:, dim_i])-0.2, np.max(embedding_ds[:, dim_i])-0.2
                m = m - 0.025 * np.abs(M - m)
                M = M + 0.025 * np.abs(M - m)
                gr = np.linspace(m, M, steps[dim_i])
                grs.append(gr)

            meshes_tuple = np.meshgrid(*grs)
            gridpoints_coordinates = np.vstack(
                [i.flat for i in meshes_tuple]).T

            n_neighbors = int(velocity_embedding.shape[0]/3)
            dists_head, neighs_head = find_neighbors(
                embedding_ds, gridpoints_coordinates, n_neighbors, radius=1)
            dists_tail, neighs_tail = find_neighbors(
                embedding_ds+velocity_embedding, gridpoints_coordinates, n_neighbors, radius=1)
            std = np.mean([(g[1] - g[0]) for g in grs])

            # isotropic gaussian kernel
            gaussian_w_head = normal.pdf(
                loc=0, scale=smooth * std, x=dists_head)
            total_p_mass_head = gaussian_w_head.sum(1)
            gaussian_w_tail = normal.pdf(
                loc=0, scale=smooth * std, x=dists_tail)
            total_p_mass_tail = gaussian_w_tail.sum(1)

            
            UZ_head = (velocity_embedding[neighs_head] * gaussian_w_head[:, :, None]).sum(
                1) / np.maximum(1, total_p_mass_head)[:, None]  # weighed average
            UZ_tail = (velocity_embedding[neighs_tail] * gaussian_w_tail[:, :, None]).sum(
                1) / np.maximum(1, total_p_mass_tail)[:, None]  # weighed average

            XY = gridpoints_coordinates

            dists_head2, neighs_head2 = find_neighbors(
                embedding_ds, XY+UZ_head, n_neighbors, radius=1)
            dists_tail2, neighs_tail2 = find_neighbors(
                embedding_ds, XY-UZ_tail, n_neighbors, radius=1)

            gaussian_w_head2 = normal.pdf(
                loc=0, scale=smooth * std, x=dists_head2)
            total_p_mass_head2 = gaussian_w_head2.sum(1)
            gaussian_w_tail2 = normal.pdf(
                loc=0, scale=smooth * std, x=dists_tail2)
            total_p_mass_tail2 = gaussian_w_tail2.sum(1)

            UZ_head2 = (velocity_embedding[neighs_head2] * gaussian_w_head2[:, :, None]).sum(
                1) / np.maximum(1, total_p_mass_head2)[:, None]  # weighed average
            UZ_tail2 = (velocity_embedding[neighs_tail2] * gaussian_w_tail2[:, :, None]).sum(
                1) / np.maximum(1, total_p_mass_tail2)[:, None]  # weighed average

            mass_filter = total_p_mass_head < min_mass

            # filter dots
            UZ_head_filtered = UZ_head[~mass_filter, :]
            UZ_tail_filtered = UZ_tail[~mass_filter, :]
            UZ_head2_filtered = UZ_head2[~mass_filter, :]
            UZ_tail2_filtered = UZ_tail2[~mass_filter, :]
            XY_filtered = XY[~mass_filter, :]
            return(XY_filtered, UZ_head_filtered, UZ_tail_filtered, UZ_head2_filtered, UZ_tail2_filtered, mass_filter, grs)

        XY_filtered, UZ_head_filtered, UZ_tail_filtered, UZ_head2_filtered, UZ_tail2_filtered, mass_filter, grs = calculate_two_end_grid(
            embedding_ds, velocity_embedding, smooth=0.8, steps=grid_steps, min_mass=min_mass)

        # plt.quiver(XY_filtered[:, 0], XY_filtered[:, 1], UZ_head_filtered[:, 0], UZ_head_filtered[:, 1], zorder=20000, color='blue')
        # plt.quiver(XY_filtered[:, 0]-UZ_tail_filtered[:, 0], XY_filtered[:, 1]-UZ_tail_filtered[:, 1], UZ_tail_filtered[:, 0], UZ_tail_filtered[:, 1], zorder=20000, color='red')

        #######################################################
        ############ connect two end grid to curve ############
        #######################################################
        n_curves = XY_filtered.shape[0]
        s_vals = np.linspace(0.0, 1.5, 15) # TODO check last
        ############ get longest distance len and norm ratio ############
        XYM = XY_filtered
        UVT = UZ_tail_filtered
        UVH = UZ_head_filtered
        UVT2 = UZ_tail2_filtered
        UVH2 = UZ_head2_filtered

        def norm_arrow_display_ratio(XYM, UVT, UVH, UVT2, UVH2, grs, s_vals):
            '''get the longest distance in prediction between the five points,
            and normalize by using the distance between two grids'''

            def distance(x, y):
                # calc disctnce list between a set of coordinate
                calculate_square = np.subtract(
                    x[0:-1], x[1:])**2 + np.subtract(y[0:-1], y[1:])**2
                distance_result = (calculate_square)**0.5
                return distance_result

            max_discance = 0
            for i in range(n_curves):
                nodes = np.asfortranarray([[XYM[i, 0]-UVT[i, 0]-UVT2[i, 0], XYM[i, 0]-UVT[i, 0], XYM[i, 0], XYM[i, 0]+UVH[i, 0], XYM[i, 0]+UVH[i, 0]+UVH2[i, 0]],
                                            [XYM[i, 1]-UVT[i, 1]-UVT2[i, 1], XYM[i, 1]-UVT[i, 1], XYM[i, 1], XYM[i, 1]+UVH[i, 1], XYM[i, 1]+UVH[i, 1]+UVH2[i, 1]]])
                curve = bezier.Curve(nodes, degree=4)
                curve_dots = curve.evaluate_multi(s_vals)
                distance_sum = np.sum(
                    distance(curve_dots[0], curve_dots[1]))
                max_discance = max(max_discance, distance_sum)
            distance_grid = (
                abs(grs[0][0]-grs[0][1]) + abs(grs[1][0]-grs[1][1]))/2
            norm_ratio = distance_grid/max_discance
            return(norm_ratio)

        norm_ratio = norm_arrow_display_ratio(
            XYM, UVT, UVH, UVT2, UVH2, grs, s_vals)
        ############ end --- get longest distance len and norm ratio ############

        ############ plot the curve arrow for cell velocity ############

        XYM = XY_filtered
        UVT = UZ_tail_filtered * norm_ratio
        UVH = UZ_head_filtered * norm_ratio
        UVT2 = UZ_tail2_filtered * norm_ratio
        UVH2 = UZ_head2_filtered * norm_ratio

        def plot_cell_velocity_curve(XYM, UVT, UVH, UVT2, UVH2, s_vals):
            plt.axis('equal')
            # TO DO: add 'colorful cell velocity' to here, now there is only curve arrows
            for i in range(n_curves):
                nodes = np.asfortranarray([[XYM[i, 0]-UVT[i, 0]-UVT2[i, 0], XYM[i, 0]-UVT[i, 0], XYM[i, 0], XYM[i, 0]+UVH[i, 0], XYM[i, 0]+UVH[i, 0]+UVH2[i, 0]],
                                            [XYM[i, 1]-UVT[i, 1]-UVT2[i, 1], XYM[i, 1]-UVT[i, 1], XYM[i, 1], XYM[i, 1]+UVH[i, 1], XYM[i, 1]+UVH[i, 1]+UVH2[i, 1]]])
                curve = bezier.Curve(nodes, degree=4)
                curve_dots = curve.evaluate_multi(s_vals)
                plt.plot(curve_dots[0], curve_dots[1],
                            linewidth=0.5, color='black', alpha=1)

                # normalize the arrow of the last two points at the tail, to let all arrows has the same size in quiver
                U = curve_dots[0][-1]-curve_dots[0][-2]
                V = curve_dots[1][-1]-curve_dots[1][-2]
                N = np.sqrt(U**2 + V**2)
                U1, V1 = U/N*0.5, V/N*0.5  # 0.5 is to let the arrow have a suitable size
                plt.quiver(curve_dots[0][-2], curve_dots[1][-2], U1, V1, units='xy', angles='xy',
                            scale=1, linewidth=0, color='black', alpha=1, minlength=0, width=0.1)

            # used to help identify arrow and line
            # plt.quiver(XY_filtered[:, 0], XY_filtered[:, 1], UZ_head_filtered[:, 0], UZ_head_filtered[:, 1], zorder=20000, color='blue',linewidth=0,alpha=0.2)
            # plt.quiver(XY_filtered[:, 0]-UZ_tail_filtered[:, 0], XY_filtered[:, 1]-UZ_tail_filtered[:, 1], UZ_tail_filtered[:, 0], UZ_tail_filtered[:, 1], zorder=20000, color='red',alpha=0.2)

        plot_cell_velocity_curve(XYM, UVT, UVH, UVT2, UVH2, s_vals)
        ############ end --- plot the curve arrow for cell velocity ############

    if velocity:
        grid_curve(embedding_ds, velocity_embedding)


    if custom_xlim is not None:
        plt.xlim(custom_xlim[0], custom_xlim[1])
    if custom_ylim is not None:
        plt.ylim(custom_ylim[0], custom_ylim[1])
    
    lgd=plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path,('velocity_embedding' + \
        file_name_additional_info+\
        '_colorful_grid_curve_arrow.pdf')),bbox_inches='tight',bbox_extra_artists=(lgd,),)
        

def cell_level_para_plot(load_cellDancer,gene_choice,para_list,cluster_choice=None,save_path=None,pointsize=0.2,alpha=1):
    
    '''plot alpha, beta, gamma, s0, u0 at cell level'''

    if cluster_choice is not None:
        reindexed_one_gene=load_cellDancer[load_cellDancer.gene_name==gene_choice[0]].reset_index()
        embedding=reindexed_one_gene[reindexed_one_gene.clusters.isin(cluster_choice)][['embedding1','embedding2']].to_numpy()
    else:
        embedding=load_cellDancer[load_cellDancer.gene_name==gene_choice[0]][['embedding1','embedding2']].to_numpy()
    
    color_map_zissou2 = LinearSegmentedColormap.from_list("mycmap", zissou2)
    color_map_fireworks3 = LinearSegmentedColormap.from_list("mycmap", fireworks3)
    color_dict = {'alpha':color_map_zissou2,'beta':color_map_zissou2,'gamma':color_map_zissou2,'s0':color_map_fireworks3,'u0':color_map_fireworks3}

    for para in para_list:
        for gene_name in gene_choice:
            #gene_name='Ntrk2'
            one_gene=load_cellDancer[load_cellDancer.gene_name==gene_name].reset_index()
            if cluster_choice is not None: 
                # vmin=min(one_gene[para])
                # vmax=max(one_gene[para])
                # layer=plt.scatter(embedding[:,0],embedding[:,1],s=0.2,c=one_gene[reindexed_one_gene.clusters.isin(cluster_choice)][para],cmap=color_map,vmin=vmin,vmax=vmax)
                layer=plt.scatter(embedding[:,0],embedding[:,1],s=pointsize,alpha=alpha,c=one_gene[reindexed_one_gene.clusters.isin(cluster_choice)][para],cmap=color_dict[para])
            
            else:
                layer=plt.scatter(embedding[:,0],embedding[:,1],s=pointsize,c=one_gene[para],cmap=color_dict[para])
            
            plt.title(gene_name+" "+para)
            plt.colorbar(layer)
            if save_path is not None:
                plt.savefig(os.path.join(save_path,(gene_name+'_'+para+'.pdf')),dpi=300)
            plt.show()


# PENGZHI -> Move this to utilities
def find_neighbors(data, gridpoints_coordinates, n_neighbors, radius=1):
    nn = NearestNeighbors(n_neighbors=n_neighbors, radius=1, n_jobs=-1)
    nn.fit(data)
    dists, neighs = nn.kneighbors(gridpoints_coordinates)
    return(dists, neighs)

def extract_from_df(load_cellDancer, attr_list):
    ''' 
    Extract a single copy of a list of columns from the load_cellDancer data frame
    Returns a numpy array.
    '''
    one_gene_idx = load_cellDancer.gene_name == load_cellDancer.gene_name[0]
    data = load_cellDancer[one_gene_idx][attr_list].dropna()
    return data.to_numpy()
