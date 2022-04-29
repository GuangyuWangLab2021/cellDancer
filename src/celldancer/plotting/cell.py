import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.stats import norm as normal
import bezier
import numpy as np
import pandas as pd
from .colormap import *

if __name__ == "__main__":# developer test
    sys.path.append('..')
    from utilities import find_nn_neighbors, extract_from_df
else:
    from celldancer.utilities import find_nn_neighbors, extract_from_df

def scatter_cell(
        ax,
        load_cellDancer, 
        save_path=None,
        custom_xlim=None,
        custom_ylim=None,
        vmin=None,
        vmax=None,
        colors=None, 
        alpha=0.5, 
        s = 5,
        velocity=False,
        gene_name=None,
        legend='off',
        colorbar='on',
        min_mass=2,
        grid_steps=(30,30)): 


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

    
    #   colors can be one of those:
    #   a dictionary of {cluster:color}
    #   a string of the column attribute (pseudotime, alpha, beta etc)
    #   a list of clusters 
    #   None

    def gen_Line2D(label, markerfacecolor):
        return Line2D([0], [0], color='w', marker='o', label=label,
            markerfacecolor=markerfacecolor, 
            markeredgewidth=0,
            markersize=s)

    if isinstance(colors, list):
        #print("\nbuild a colormap for a list of clusters as input\n")
        colors = build_colormap(colors)
    
    if isinstance(colors, dict):
        attr = 'clusters'
        legend_elements= [gen_Line2D(i, colors[i]) for i in colors]
        if legend is not 'off':
            lgd=ax.legend(handles=legend_elements,
                bbox_to_anchor=(1.01, 1),
                loc='upper left')
            bbox_extra_artists=(lgd,)
            if legend is 'only':
                return lgd
        else:
            bbox_extra_artists=None

        c=np.vectorize(colors.get)(extract_from_df(load_cellDancer, 'clusters', gene_name))
        cmap=ListedColormap(list(colors.keys()))
    elif isinstance(colors, str):
        attr = colors
        if colors in ['alpha', 'beta', 'gamma']:
            assert gene_name, '\nError! gene_name is required!\n'
            cmap = LinearSegmentedColormap.from_list("mycmap", zissou2)
        if colors in ['spliced', 'unspliced']:
            assert gene_name, '\nError! gene_name is required!\n'
            colors = {'spliced':'s0', 'unspliced':'u0'}[colors]
            cmap = LinearSegmentedColormap.from_list("mycmap", fireworks3)
        if colors in ['pseudotime']:
            cmap = 'viridis'
        c = extract_from_df(load_cellDancer, [colors], gene_name)
        
    elif colors is None:
        attr = 'basic'
        cmap = None
        c = 'Grey'
        
    
    embedding = extract_from_df(load_cellDancer, ['embedding1', 'embedding2'], gene_name)
    n_cells = embedding.shape[0]
    sample_cells = load_cellDancer['velocity1'][:n_cells].isna()
    embedding_ds = embedding[~sample_cells]
    
    im=ax.scatter(embedding[:, 0],
                embedding[:, 1],
                c=c,
                cmap=cmap,
                s=s,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                edgecolor="none")
    if colorbar is 'on' and  isinstance(colors, str):
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("top", size="5%", pad="-5%")

        cbar = plt.colorbar(im, cax=cax, orientation="horizontal", shrink=0.1)
        cbar.set_ticks([])

    if velocity:
        velocity_embedding= extract_from_df(load_cellDancer, ['velocity1', 'velocity2'], gene_name)
        grid_curve(ax, embedding_ds, velocity_embedding, grid_steps, min_mass)

    if custom_xlim is not None:
        ax.set_xlim(custom_xlim[0], custom_xlim[1])
    if custom_ylim is not None:
        ax.set_ylim(custom_ylim[0], custom_ylim[1])
    
    
    if save_path is not None:
        file_name_parts = ['embedding', attr, gene_name]
        if velocity:
            file_name_parts.insert(0, 'velocity')
        
        save_file_name = os.path.join(save_path, "_".join(file_name_parts)+'.pdf')
        
        print("saved the file as", save_file_name)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(save_file_name, 
                bbox_inches=extent,
                bbox_extra_artists=bbox_extra_artists)
        
    return im


# Source - https://github.com/velocyto-team/velocyto.py/blob/0963dd2df0ac802c36404e0f434ba97f07edfe4b/velocyto/analysis.py
def grid_curve(ax, embedding_ds, velocity_embedding, grid_steps, min_mass):
# calculate_grid_arrows
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
        dists_head, neighs_head = find_nn_neighbors(
            embedding_ds, gridpoints_coordinates, n_neighbors, radius=1)
        dists_tail, neighs_tail = find_nn_neighbors(
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

        dists_head2, neighs_head2 = find_nn_neighbors(
            embedding_ds, XY+UZ_head, n_neighbors, radius=1)
        dists_tail2, neighs_tail2 = find_nn_neighbors(
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
        # TO DO: add 'colorful cell velocity' to here, now there is only curve arrows
        for i in range(n_curves):
            nodes = np.asfortranarray([[XYM[i, 0]-UVT[i, 0]-UVT2[i, 0], XYM[i, 0]-UVT[i, 0], XYM[i, 0], XYM[i, 0]+UVH[i, 0], XYM[i, 0]+UVH[i, 0]+UVH2[i, 0]],
                                        [XYM[i, 1]-UVT[i, 1]-UVT2[i, 1], XYM[i, 1]-UVT[i, 1], XYM[i, 1], XYM[i, 1]+UVH[i, 1], XYM[i, 1]+UVH[i, 1]+UVH2[i, 1]]])
            curve = bezier.Curve(nodes, degree=4)
            curve_dots = curve.evaluate_multi(s_vals)
            ax.plot(curve_dots[0], curve_dots[1],
                        linewidth=0.5, color='black', alpha=1)

            # normalize the arrow of the last two points at the tail, to let all arrows has the same size in quiver
            U = curve_dots[0][-1]-curve_dots[0][-2]
            V = curve_dots[1][-1]-curve_dots[1][-2]
            N = np.sqrt(U**2 + V**2)
            U1, V1 = U/N*0.5, V/N*0.5  # 0.5 is to let the arrow have a suitable size
            ax.quiver(curve_dots[0][-2], curve_dots[1][-2], U1, V1, units='xy', angles='xy',
                        scale=1, linewidth=0, color='black', alpha=1, minlength=0, width=0.1)

    plot_cell_velocity_curve(XYM, UVT, UVH, UVT2, UVH2, s_vals)
    ############ end --- plot the curve arrow for cell velocity ############

def calculate_para_umap(load_cellDancer,para,umap_n=25):

    import umap
    if set([(para+'_umap1'),(para+'_umap2')]).issubset(load_cellDancer.columns):
        load_cellDancer=load_cellDancer.drop(columns=[(para+'_umap1'),(para+'_umap2')])

    if para=='alpha' or para=='beta' or para=='gamma':
        para_df=load_cellDancer.pivot(index='cellIndex', columns='gene_name', values=para)
    elif para=='alpha_beta_gamma':
        alpha_df=load_cellDancer.pivot(index='cellIndex', columns='gene_name', values='alpha')
        beta_df=load_cellDancer.pivot(index='cellIndex', columns='gene_name', values='beta')
        gamma_df=load_cellDancer.pivot(index='cellIndex', columns='gene_name', values='gamma')
        para_df=pd.concat([alpha_df,beta_df,gamma_df],axis=1)
    else:
        print('para should be set in one of alpha, beta, gamma, or alpha_beta_gamma.')

    def get_umap(df,n_neighbors=umap_n, min_dist=0.1, n_components=2, metric='euclidean'):
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric
        )
        embed = fit.fit_transform(df);
        return(embed)
    umap_para=get_umap(para_df)
    umap_info=pd.DataFrame(umap_para,columns=[(para+'_umap1'),(para+'_umap2')])

    gene_amt=len(load_cellDancer.gene_name.drop_duplicates())
    umap_col=pd.concat([umap_info]*gene_amt)
    umap_col.index=load_cellDancer.index
    load_cellDancer=pd.concat([load_cellDancer,umap_col],axis=1)
    return(load_cellDancer)

def plot_para_umap(para,load_cellDancer,gene_name=None,umap_n=25,cluster_map=None,save_path=None,title=None,legend_annotation=False):
    import numpy as np
    onegene=load_cellDancer[load_cellDancer.gene_name==load_cellDancer.gene_name[0]]
    umap_para=onegene[[(para+'_umap1'),(para+'_umap2')]].to_numpy()
    onegene_cluster_info=onegene.clusters

    if gene_name is None:
        if cluster_map is None:
            from plotting.colormap import build_colormap
            cluster_map=build_colormap(onegene_cluster_info)

        colors = list(map(lambda x: cluster_map.get(x, 'black'), onegene_cluster_info))

        if legend_annotation:
            markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in cluster_map.values()]
            lgd=plt.legend(markers, cluster_map.keys(), numpoints=1,loc='upper left',bbox_to_anchor=(1.01, 1))

        plt.scatter(umap_para[:,0], umap_para[:,1],c=colors,s=15,alpha=0.5,edgecolor="none")
        plt.axis('square')
        plt.axis('off')

    else:
        onegene=load_cellDancer[load_cellDancer.gene_name==gene_name]
        plt.figure()
        plt.scatter(umap_para[:,0], umap_para[:,1],c=np.log(onegene.s0+0.0001),s=15,alpha=1,edgecolor="none")
        plt.axis('square')
        plt.axis('off')
        plt.colorbar(label=gene_name+" s0")

    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight',bbox_extra_artists=(lgd,))
    umap_df=pd.concat([pd.DataFrame({'umap1':umap_para[:,0],'umap2':umap_para[:,1]})],axis=1)