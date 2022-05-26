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
    cellDancer_df, 
    colors=None, 
    custom_xlim=None,
    custom_ylim=None,
    vmin=None,
    vmax=None,
    alpha=0.5, 
    s = 5,
    velocity=False,
    gene=None,
    legend='off',
    colorbar='on',
    min_mass=2,
    arrow_grid=(30,30),
    save_path=None
): 

"""Plot the the cell velocity; or plot the parameters ('alpha', 'beta', 'gamma', 'spliced', 'unspliced', or 'pseudotime') of one gene in embedding level.
    
Arguments
---------
ax: `ax`
    ax of plt.subplots()
cellDancer_df: `pandas.DataFrame`
    Data frame of velocity estimation, cell velocity, and pseudotime results - columns=['cellIndex','gene_name','s0','u0','s1','u1','alpha','beta','gamma','loss','cellID','clusters','embedding1','embedding2','velocity1','velocity2','pseudotime']
colors: `list`, `dict`, or `str`
    `list` -> build a colormap dictionary for a list of cell type as input.
    `dict` -> the customized color map dict of each cell type.
    `str` -> one of {'alpha','beta','gamma','spliced','unspliced','pseudotime'}.
custom_xlim: `float` (default: None)
    Set the x limits of the current axes.
custom_ylim: `float` (default: None)
    Set the y limits of the current axes.
vmin: `float` (default: None)
    Set the minimun color limits of the current image.
vmax: `float` (default: None)
    Set the maximum color limits of the current image.
alpha: `float` (default: 0.5)
    The alpha blending value, between 0 (transparent) and 1 (opaque).
s: `float` (default: 5)
    The marker size in points**2.
velocity: `bool` (default: False)
    True if velocity in cell level is to be plotted.
gene: `str` (default: None)
    The gene been selected for the plot of alpha, beta, gamma, spliced, or unspliced in embedding level.
legend: `str` (default: 'off')
    'off' if the color map of cell legend is not to be plotted.
    'only' if to only plot the cell type legend.
colorbar: `str` (default: 'on')
    'on' if the colorbar of of the plot of alpha, beta, gamma, spliced, or unspliced is to be shown.
min_mass: `float` (default: 2)
    Filter by using the isotropic gaussian kernel to display the grid, The less, the more arrows.
arrow_grid: `tuple` (default: (30,30))
    The size of the grid for the cell velocity to display.
save_path: `str` (default: None)
    Directory to save the plot.

Returns
-------
Returns the ax of the plot.
`im` (ax.scatter())

"""  

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
        if legend != 'off':
            lgd=ax.legend(handles=legend_elements,
                bbox_to_anchor=(1.01, 1),
                loc='upper left')
            bbox_extra_artists=(lgd,)
            if legend == 'only':
                return lgd
        else:
            bbox_extra_artists=None

        c=np.vectorize(colors.get)(extract_from_df(cellDancer_df, 'clusters', gene))
        cmap=ListedColormap(list(colors.keys()))
    elif isinstance(colors, str):
        attr = colors
        if colors in ['alpha', 'beta', 'gamma']:
            assert gene, '\nError! gene is required!\n'
            

            cmap = LinearSegmentedColormap.from_list("mycmap", color_map_single_alpha_beta_gamma)
        if colors in ['spliced', 'unspliced']:
            assert gene, '\nError! gene is required!\n'
            colors = {'spliced':'s0', 'unspliced':'u0'}[colors]
            cmap = LinearSegmentedColormap.from_list("mycmap", color_map_alpha_beta_gamma)
        if colors in ['pseudotime']:
            cmap = 'viridis'
        c = extract_from_df(cellDancer_df, [colors], gene)
        
    elif colors is None:
        attr = 'basic'
        cmap = None
        c = 'Grey'
    
    embedding = extract_from_df(cellDancer_df, ['embedding1', 'embedding2'], gene)
    n_cells = embedding.shape[0]
    
    im=ax.scatter(embedding[:, 0],
                embedding[:, 1],
                c=c,
                cmap=cmap,
                s=s,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                edgecolor="none")
    if colorbar == 'on' and  isinstance(colors, str):
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("top", size="5%", pad="-5%")

        print("   \n ")
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal", shrink=0.1)
       # cbar.set_ticks([])

    if velocity:
        sample_cells = cellDancer_df['velocity1'][:n_cells].dropna().index
        embedding_ds = embedding[sample_cells]
        velocity_embedding= extract_from_df(cellDancer_df, ['velocity1', 'velocity2'], gene)
        grid_curve(ax, embedding_ds, velocity_embedding, arrow_grid, min_mass)

    if custom_xlim is not None:
        ax.set_xlim(custom_xlim[0], custom_xlim[1])
    if custom_ylim is not None:
        ax.set_ylim(custom_ylim[0], custom_ylim[1])
    
    
    if save_path is not None:
        file_name_parts = ['embedding', attr, gene]
        if velocity:
            file_name_parts.insert(0, 'velocity')
        
        save_file_name = os.path.join(save_path, "_".join(file_name_parts)+'.pdf')
        
        print("saved the file as", save_file_name)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(save_file_name, 
                bbox_inches=extent,
                bbox_extra_artists=bbox_extra_artists)
    return im

def grid_curve(
    ax, 
    embedding_ds, 
    velocity_embedding, 
    arrow_grid, 
    min_mass
):
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
            embedding_ds, gridpoints_coordinates, n_neighbors)
        dists_tail, neighs_tail = find_nn_neighbors(
            embedding_ds+velocity_embedding, gridpoints_coordinates,
            n_neighbors)
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
            embedding_ds, XY+UZ_head, n_neighbors)
        dists_tail2, neighs_tail2 = find_nn_neighbors(
            embedding_ds, XY-UZ_tail, n_neighbors)

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
        embedding_ds, velocity_embedding, smooth=0.8, steps=arrow_grid, min_mass=min_mass)

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

    norm_ratio = norm_arrow_display_ratio(XYM, UVT, UVH, UVT2, UVH2, grs, s_vals)
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

def embedding_kinetic_para(
    cellDancer_df,
    kinetic_para,
    umap_n=25
):
    """Calculate the UMAP based on kinetic parameter(s).
        
    Arguments
    ---------
    cellDancer_df: `pandas.Dataframe`
        Data frame of velocity estimation results - columns=['cellIndex','gene_name','s0','u0','s1','u1','alpha','beta','gamma','loss','cellID','clusters','embedding1','embedding2']
    kinetic_para: `str`
        Parameter selected to calculate the embedding by using umap, Could be selected from {'alpha', 'beta', 'gamma', 'alpha_beta_gamma'}.
    umap_n: `int` (default: 25)
        The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation in UMAP.

    Returns
    -------
    Returns the updated cellDancer_df with additional column of UMAP based on kinetic parameter(s).
    `cellDancer_df` (pandas.DataFrame)

    """  
    import umap
    if set([(kinetic_para+'_umap1'),(kinetic_para+'_umap2')]).issubset(cellDancer_df.columns):
        cellDancer_df=cellDancer_df.drop(columns=[(kinetic_para+'_umap1'),(kinetic_para+'_umap2')])

    if kinetic_para=='alpha' or kinetic_para=='beta' or kinetic_para=='gamma':
        para_df=cellDancer_df.pivot(index='cellIndex', columns='gene_name', values=kinetic_para)
    elif kinetic_para=='alpha_beta_gamma':
        alpha_df=cellDancer_df.pivot(index='cellIndex', columns='gene_name', values='alpha')
        beta_df=cellDancer_df.pivot(index='cellIndex', columns='gene_name', values='beta')
        gamma_df=cellDancer_df.pivot(index='cellIndex', columns='gene_name', values='gamma')
        para_df=pd.concat([alpha_df,beta_df,gamma_df],axis=1)
    else:
        print('kinetic_para should be set in one of alpha, beta, gamma, or alpha_beta_gamma.')

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
    umap_info=pd.DataFrame(umap_para,columns=[(kinetic_para+'_umap1'),(kinetic_para+'_umap2')])

    gene_amt=len(cellDancer_df.gene_name.drop_duplicates())
    umap_col=pd.concat([umap_info]*gene_amt)
    umap_col.index=cellDancer_df.index
    cellDancer_df=pd.concat([cellDancer_df,umap_col],axis=1)
    return(cellDancer_df)

def plot_kinetic_para(
    cellDancer_df,
    kinetic_para,
    gene=None,
    color_map=None,
    save_path=None,
    title=None,
    legend=False
):

    """Plot the UMAP calculated by kinetic parameter(s).
        
    Arguments
    ---------
    cellDancer_df: `pandas.Dataframe`
        Data frame of velocity estimation results - columns=['cellIndex','gene_name','s0','u0','s1','u1','alpha','beta','gamma','loss','cellID','clusters','embedding1','embedding2']
    kinetic_para: `str`
        Parameter selected plot, Could be selected from {'alpha', 'beta', 'gamma', 'alpha_beta_gamma'}.
    gene: `str` (default: None)
        If the gene name is set, s0 of this gene in the embeddings based on kinetic parameter(s) will be displayed.
    color_map: `dict` (default: None)
        The color map of each cell tpye.
    save_path: `str` (default: None)
        Directory to save the plot.
    legend: `bool` (default: False)
        Display the color legend of each cell type.

    """    
    onegene=cellDancer_df[cellDancer_df.gene_name==cellDancer_df.gene_name[0]]
    umap_para=onegene[[(kinetic_para+'_umap1'),(kinetic_para+'_umap2')]].to_numpy()
    onegene_cluster_info=onegene.clusters

    if gene is None:
        if color_map is None:
            from plotting.colormap import build_colormap
            color_map=build_colormap(onegene_cluster_info)

        colors = list(map(lambda x: color_map.get(x, 'black'), onegene_cluster_info))

        if legend:
            markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_map.values()]
            lgd=plt.legend(markers, color_map.keys(), numpoints=1,loc='upper left',bbox_to_anchor=(1.01, 1))
                
        plt.figure()
        plt.scatter(umap_para[:,0], umap_para[:,1],c=colors,s=15,alpha=0.5,edgecolor="none")
        plt.axis('square')
        plt.axis('off')

    else:
        onegene=cellDancer_df[cellDancer_df.gene_name==gene]
        plt.figure()
        plt.scatter(umap_para[:,0], umap_para[:,1],c=np.log(onegene.s0+0.0001),s=15,alpha=1,edgecolor="none")
        plt.axis('square')
        plt.axis('off')
        plt.colorbar(label=gene+" s0")

    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight',bbox_extra_artists=(lgd,))
    umap_df=pd.concat([pd.DataFrame({'umap1':umap_para[:,0],'umap2':umap_para[:,1]})],axis=1)
