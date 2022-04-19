import matplotlib.pyplot as plt
import os
import sys

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
################# gene_pseudotime
        
class velocity_plot():
    
    '''This class includes all plot functions'''
    
    def velocity_cell_map_draft(embedding, sampling_ixs, velocity_embedding, save_path=None):
        
        '''Cell velocity plot (draft).'''
        
        plt.figure()
        plt.scatter(embedding[sampling_ixs, 0],
                    embedding[sampling_ixs, 1], s=0.5)
        plt.quiver(embedding[sampling_ixs, 0], embedding[sampling_ixs, 1],
                   velocity_embedding[:, 0], velocity_embedding[:, 1], color='red')
        if save_path is not None:
            plt.savefig(save_path)

    def velocity_cell_map(load_raw_data,load_cellDancer, n_neighbors=200,add_amt_gene=2000,step=(60,60),save_path=None,save_csv=None, gene_list=None, custom_xlim=None,colors=None,mode='embedding',use_downsampling=True):
        from get_embedding import get_embedding

        """Cell velocity plot.

        TO DO: load_raw_data contains the cluster information, needs improve
        
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

        embedding, sampling_ixs, velocity_embedding=get_embedding(load_raw_data,load_cellDancer,gene_list=gene_list,n_neighbors=n_neighbors,step=step,mode=mode,use_downsampling=use_downsampling)
        if colors is not None:
            colors=colors
        else:
            colors = {'CA': grove2[7],
                    'CA1-Sub': grove2[9],
                    'CA2-3-4': grove2[8],
                    'Granule': grove2[6],
                    'ImmGranule1': grove2[6],
                    'ImmGranule2': grove2[6],
                    'Nbl1': grove2[5],
                    'Nbl2': grove2[5],
                    'nIPC': grove2[4],
                    'RadialGlia': grove2[3],
                    'RadialGlia2': grove2[3],
                    'GlialProg': grove2[2],
                    'OPC': grove2[1],
                    'ImmAstro': grove2[0]}
        pointsize = 5

        one_gene_raw = load_raw_data.gene_list[0]

        step_i = 25
        step_j = 25
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        def gen_Line2D(label, markerfacecolor):
            return Line2D([0], [0], color='w', marker='o', label=label, markerfacecolor=markerfacecolor,  markeredgewidth=0, markersize=5)

        legend_elements = []
        for i in colors:
            legend_elements.append(gen_Line2D(i, colors[i]))

        arrow_idx = sampling_neighbors(
            embedding[sampling_ixs, :], step_i=step_i, step_j=step_j)
            
        plt.figure()
        plt.scatter(embedding[:, 0],
                    embedding[:, 1],
                    c=load_raw_data[load_raw_data.gene_list ==
                                    one_gene_raw]['clusters'].map(colors),
                    s=pointsize,
                    # alpha=1,
                    # alpha=0.3,
                    # alpha=0.05,
                    alpha=0.1,
                    edgecolor="none")

        # arrow all points
        plt.quiver(embedding[sampling_ixs, 0],
                    embedding[sampling_ixs, 1],
                    velocity_embedding[:, 0],
                    velocity_embedding[:, 1],
                    # color='black',
                    color=load_raw_data[load_raw_data.gene_list ==
                                        one_gene_raw]['clusters'][sampling_ixs].map(colors),
                    units='xy',
                    angles='xy',
                    scale=1,
                    linewidth=0,
                    width=0.2,
                    alpha=1,
                    )
        # arrow all points - bkup for nature dataset
        # plt.quiver(embedding[sampling_ixs, 0],
        #             embedding[sampling_ixs, 1],
        #             velocity_embedding[:, 0],
        #             velocity_embedding[:, 1],
        #             # color='black',
        #             color=load_raw_data[load_raw_data.gene_list ==
        #                                 one_gene_raw]['clusters'][sampling_ixs].map(colors),
        #             angles='xy',
        #             scale=1,
        #             clim=(0., 1.),
        #             width=0.002,
        #             alpha=1,
        #             linewidth=.2,
        #             )

        
        if custom_xlim is not None:
            plt.xlim(-23, 45)
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')
        if save_path is not None:
            plt.savefig(os.path.join(save_path,('velocity_embedding_tune_n' + \
            str(n_neighbors)+'_gAmt'+str(add_amt_gene) + \
            '_colorful_arrow.pdf')))
        if save_csv is not None:
            cell_velocity_df=pd.DataFrame({'embedding1':embedding[sampling_ixs, 0],
                         'embedding2':embedding[sampling_ixs, 1],
                         ('embedding1_n'+str(n_neighbors)+'_gAmt'+str(add_amt_gene)):velocity_embedding[:, 0],
                         ('embedding2_n'+str(n_neighbors)+'_gAmt'+str(add_amt_gene)):velocity_embedding[:, 1]})
            cell_velocity_df.to_csv(os.path.join(save_path,('velocity_embedding_tune_n' + \
            str(n_neighbors)+'_gAmt'+str(add_amt_gene) + \
            '_colorful_arrow.csv')))
        
    def velocity_cell_map_curve(load_raw_data,load_cellDancer, n_neighbors=200,add_amt_gene=2000,step=(60,60),save_path=None, save_csv=False,gene_list=None, custom_xlim=None,custom_ylim=None,colors=None,mode='embedding',pca_n_components=4,file_name_additional_info='',umap_n=10,transfer_mode=None,umap_n_components=None,min_mass=2,grid_steps=(30,30),alpha_inside=0.5,use_downsampling=True):
        from get_embedding import get_embedding

        """Cell velocity plot.

        TO DO: load_raw_data contains the cluster information, needs improve
        
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

        embedding, sampling_ixs, velocity_embedding=get_embedding(load_raw_data,load_cellDancer,gene_list=gene_list,n_neighbors=n_neighbors,step=step,mode=mode,pca_n_components=pca_n_components,umap_n=umap_n,transfer_mode=transfer_mode,umap_n_components=umap_n_components,use_downsampling=use_downsampling)

        if colors is not None:
            colors=colors
        else:
            colors = {'CA': grove2[7],
                    'CA1-Sub': grove2[9],
                    'CA2-3-4': grove2[8],
                    'Granule': grove2[6],
                    'ImmGranule1': grove2[6],
                    'ImmGranule2': grove2[6],
                    'Nbl1': grove2[5],
                    'Nbl2': grove2[5],
                    'nIPC': grove2[4],
                    'RadialGlia': grove2[3],
                    'RadialGlia2': grove2[3],
                    'GlialProg': grove2[2],
                    'OPC': grove2[1],
                    'ImmAstro': grove2[0]}
        pointsize = 5

        one_gene_raw = load_raw_data.gene_list[0]

        step_i = 25
        step_j = 25
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        def gen_Line2D(label, markerfacecolor):
            return Line2D([0], [0], color='w', marker='o', label=label, markerfacecolor=markerfacecolor,  markeredgewidth=0, markersize=5)

        legend_elements = []
        for i in colors:
            legend_elements.append(gen_Line2D(i, colors[i]))

        arrow_idx = sampling_neighbors(
            embedding[sampling_ixs, :], step_i=step_i, step_j=step_j)
            
        plt.figure()
        plt.scatter(embedding[:, 0],
                    embedding[:, 1],
                    c=load_raw_data[load_raw_data.gene_list ==
                                    one_gene_raw]['clusters'].map(colors),
                    s=pointsize,
                    # alpha=1,
                    # alpha=0.3,
                    # alpha=0.05,
                    alpha=alpha_inside,
                    edgecolor="none")

        # arrow all points

        # calculate_grid_arrows
        # Source - https://github.com/velocyto-team/velocyto.py/blob/0963dd2df0ac802c36404e0f434ba97f07edfe4b/velocyto/analysis.py
        def grid_curve(embedding, sampling_ixs, velocity_embedding):
            from scipy.stats import norm as normal
            import bezier
            # kernel grid plot

            def calculate_two_end_grid(embedding, sampling_ixs, velocity_embedding, smooth=None, steps=None, min_mass=None):
                def find_neighbors(data, n_neighbors, gridpoints_coordinates):
                    # data  = embedding[sampling_ixs, :]
                    nn = NearestNeighbors(
                        n_neighbors=n_neighbors, n_jobs=8)
                    nn.fit(data)
                    dists, neighs = nn.kneighbors(gridpoints_coordinates)
                    return(dists, neighs)
                # Prepare the grid
                grs = []
                for dim_i in range(embedding[sampling_ixs, :].shape[1]):
                    m, M = np.min(embedding[sampling_ixs, :][:, dim_i]) - \
                        0.2, np.max(
                            embedding[sampling_ixs, :][:, dim_i])-0.2
                    m = m - 0.025 * np.abs(M - m)
                    M = M + 0.025 * np.abs(M - m)
                    gr = np.linspace(m, M, steps[dim_i])
                    grs.append(gr)

                meshes_tuple = np.meshgrid(*grs)
                gridpoints_coordinates = np.vstack(
                    [i.flat for i in meshes_tuple]).T

                n_neighbors = int(velocity_embedding.shape[0]/3)
                dists_head, neighs_head = find_neighbors(
                    embedding[sampling_ixs, :], n_neighbors, gridpoints_coordinates)
                dists_tail, neighs_tail = find_neighbors(
                    embedding[sampling_ixs, :]+velocity_embedding, n_neighbors, gridpoints_coordinates)
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
                    embedding[sampling_ixs, :], n_neighbors, XY+UZ_head)
                dists_tail2, neighs_tail2 = find_neighbors(
                    embedding[sampling_ixs, :], n_neighbors, XY-UZ_tail)

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
                embedding, sampling_ixs, velocity_embedding, smooth=0.8, steps=grid_steps, min_mass=min_mass)

            # plt.quiver(XY_filtered[:, 0], XY_filtered[:, 1], UZ_head_filtered[:, 0], UZ_head_filtered[:, 1], zorder=20000, color='blue')
            # plt.quiver(XY_filtered[:, 0]-UZ_tail_filtered[:, 0], XY_filtered[:, 1]-UZ_tail_filtered[:, 1], UZ_tail_filtered[:, 0], UZ_tail_filtered[:, 1], zorder=20000, color='red')
            # plt.savefig('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/two_end.pdf')

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
                print(max_discance)
                print(distance_grid)
                norm_ratio = distance_grid/max_discance
                print(norm_ratio)
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

        grid_curve(embedding, sampling_ixs, velocity_embedding)


        if custom_xlim is not None:
            plt.xlim(custom_xlim[0], custom_xlim[1])
            # plt.xlim(-23, 45) # for neurn dataset
        if custom_ylim is not None:
            plt.ylim(custom_ylim[0], custom_ylim[1])
        
        lgd=plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')
        # plt.show()
        
        if save_path is not None:
            plt.savefig(os.path.join(save_path,('velocity_embedding_tune_n' + \
            str(n_neighbors)+'_gAmt'+str(add_amt_gene) + \
            file_name_additional_info+\
            '_colorful_grid_curve_arrow.pdf')),bbox_inches='tight',bbox_extra_artists=(lgd,),)
        if save_csv:
            cell_velocity_df=pd.DataFrame({'embedding1':embedding[sampling_ixs, 0],
                         'embedding2':embedding[sampling_ixs, 1],
                         ('embedding1_n'+str(n_neighbors)+'_gAmt'+str(add_amt_gene)):velocity_embedding[:, 0],
                         ('embedding2_n'+str(n_neighbors)+'_gAmt'+str(add_amt_gene)):velocity_embedding[:, 1]})
            cell_velocity_df.to_csv(os.path.join(save_path,('velocity_embedding_tune_n' + \
            str(n_neighbors)+'_gAmt'+str(add_amt_gene) + \
            file_name_additional_info + \
            '_colorful_grid_curve_arrow.csv')))
        
        
    def velocity_cell_map_curve_has_embedding(load_raw_data,velocity_embedding, n_neighbors=200,add_amt_gene=2000,step=(60,60),save_path=None, save_csv=None,gene_list=None, custom_xlim=None,custom_ylim=None,colors=None,mode='embedding',pca_n_components=4,file_name_additional_info='',umap_n=10,transfer_mode=None,umap_n_components=None,min_mass=2,grid_steps=(30,30),alpha_inside=0.5,use_downsampling=True):
        from get_embedding import get_embedding

        """Cell velocity plot.

        TO DO: load_raw_data contains the cluster information, needs improve
        
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
        embedding=velocity_embedding.to_numpy()[:,0:2]
        sampling_ixs=range(0,embedding.shape[0])
        velocity_embedding=velocity_embedding.to_numpy()[:,2:4]

        if colors is not None:
            colors=colors
        else:
            colors = {'CA': grove2[7],
                    'CA1-Sub': grove2[9],
                    'CA2-3-4': grove2[8],
                    'Granule': grove2[6],
                    'ImmGranule1': grove2[6],
                    'ImmGranule2': grove2[6],
                    'Nbl1': grove2[5],
                    'Nbl2': grove2[5],
                    'nIPC': grove2[4],
                    'RadialGlia': grove2[3],
                    'RadialGlia2': grove2[3],
                    'GlialProg': grove2[2],
                    'OPC': grove2[1],
                    'ImmAstro': grove2[0]}
        pointsize = 5

        one_gene_raw = load_raw_data.gene_list[0]

        step_i = 25
        step_j = 25
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        def gen_Line2D(label, markerfacecolor):
            return Line2D([0], [0], color='w', marker='o', label=label, markerfacecolor=markerfacecolor,  markeredgewidth=0, markersize=5)

        legend_elements = []
        for i in colors:
            legend_elements.append(gen_Line2D(i, colors[i]))

        arrow_idx = sampling_neighbors(
            embedding[sampling_ixs, :], step_i=step_i, step_j=step_j)
            
        plt.figure()
        plt.scatter(embedding[:, 0],
                    embedding[:, 1],
                    c=load_raw_data[load_raw_data.gene_list ==
                                    one_gene_raw]['clusters'].map(colors),
                    s=pointsize,
                    # alpha=1,
                    # alpha=0.3,
                    # alpha=0.05,
                    alpha=alpha_inside,
                    edgecolor="none")

        # arrow all points

        # calculate_grid_arrows
        # Source - https://github.com/velocyto-team/velocyto.py/blob/0963dd2df0ac802c36404e0f434ba97f07edfe4b/velocyto/analysis.py
        def grid_curve(embedding, sampling_ixs, velocity_embedding):
            from scipy.stats import norm as normal
            import bezier
            # kernel grid plot

            def calculate_two_end_grid(embedding, sampling_ixs, velocity_embedding, smooth=None, steps=None, min_mass=None):
                def find_neighbors(data, n_neighbors, gridpoints_coordinates):
                    # data  = embedding[sampling_ixs, :]
                    nn = NearestNeighbors(
                        n_neighbors=n_neighbors, n_jobs=8)
                    nn.fit(data)
                    dists, neighs = nn.kneighbors(gridpoints_coordinates)
                    return(dists, neighs)
                # Prepare the grid
                grs = []
                for dim_i in range(embedding[sampling_ixs, :].shape[1]):
                    m, M = np.min(embedding[sampling_ixs, :][:, dim_i]) - \
                        0.2, np.max(
                            embedding[sampling_ixs, :][:, dim_i])-0.2
                    m = m - 0.025 * np.abs(M - m)
                    M = M + 0.025 * np.abs(M - m)
                    gr = np.linspace(m, M, steps[dim_i])
                    grs.append(gr)

                meshes_tuple = np.meshgrid(*grs)
                gridpoints_coordinates = np.vstack(
                    [i.flat for i in meshes_tuple]).T

                n_neighbors = int(velocity_embedding.shape[0]/3)
                dists_head, neighs_head = find_neighbors(
                    embedding[sampling_ixs, :], n_neighbors, gridpoints_coordinates)
                dists_tail, neighs_tail = find_neighbors(
                    embedding[sampling_ixs, :]+velocity_embedding, n_neighbors, gridpoints_coordinates)
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
                    embedding[sampling_ixs, :], n_neighbors, XY+UZ_head)
                dists_tail2, neighs_tail2 = find_neighbors(
                    embedding[sampling_ixs, :], n_neighbors, XY-UZ_tail)

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
                embedding, sampling_ixs, velocity_embedding, smooth=0.8, steps=grid_steps, min_mass=min_mass)

            # plt.quiver(XY_filtered[:, 0], XY_filtered[:, 1], UZ_head_filtered[:, 0], UZ_head_filtered[:, 1], zorder=20000, color='blue')
            # plt.quiver(XY_filtered[:, 0]-UZ_tail_filtered[:, 0], XY_filtered[:, 1]-UZ_tail_filtered[:, 1], UZ_tail_filtered[:, 0], UZ_tail_filtered[:, 1], zorder=20000, color='red')
            # plt.savefig('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/cell_velocity/two_end.pdf')

            #######################################################
            ############ connect two end grid to curve ############
            #######################################################
            n_curves = XY_filtered.shape[0]
            s_vals = np.linspace(0.0, 1.5, 15)
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
                print(max_discance)
                print(distance_grid)
                norm_ratio = distance_grid/max_discance
                print(norm_ratio)
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

        grid_curve(embedding, sampling_ixs, velocity_embedding)


        if custom_xlim is not None:
            plt.xlim(custom_xlim[0], custom_xlim[1])
            # plt.xlim(-23, 45) # for neurn dataset
        if custom_ylim is not None:
            plt.ylim(custom_ylim[0], custom_ylim[1])
        
        lgd=plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')
        # plt.show()
        
        if save_path is not None:
            plt.savefig(os.path.join(save_path,('velocity_embedding_tune_n' + \
            str(n_neighbors)+'_gAmt'+str(add_amt_gene) + \
            file_name_additional_info+\
            '_colorful_grid_curve_arrow.pdf')),bbox_inches='tight',bbox_extra_artists=(lgd,),)
        if save_csv is not None:
            cell_velocity_df=pd.DataFrame({'embedding1':embedding[sampling_ixs, 0],
                         'embedding2':embedding[sampling_ixs, 1],
                         ('embedding1_n'+str(n_neighbors)+'_gAmt'+str(add_amt_gene)):velocity_embedding[:, 0],
                         ('embedding2_n'+str(n_neighbors)+'_gAmt'+str(add_amt_gene)):velocity_embedding[:, 1]})
            cell_velocity_df.to_csv(os.path.join(save_path,('velocity_embedding_tune_n' + \
            str(n_neighbors)+'_gAmt'+str(add_amt_gene) + \
            file_name_additional_info + \
            '_colorful_grid_curve_arrow.csv')))
        
        
        
    def velocity_gene(gene,detail,color_scatter="#95D9EF",point_size=120,alpha_inside=0.3,v_min=None,v_max=None,save_path=None,step_i=15,step_j=15,show_arrow=True,cluster_info=None,mode=None,cluster_annot=False,colors=None):
        
        '''Gene velocity plot.
        '''

        plt.figure(None,(6,6))
        u_s= np.array(detail[detail['gene_name']==gene][["u0","s0","u1","s1"]]) # u_s

        max_u_s=np.max(u_s, axis = 0)
        u0_max=max_u_s[0]
        s0_max=max_u_s[1]
        y_max=1.25*u0_max
        x_max=1.25*s0_max

        sampling_idx=sampling_neighbors(u_s[:,0:2], step_i=step_i,step_j=step_j,percentile=15) # Sampling
        u_s_downsample = u_s[sampling_idx,0:4]
        # layer1=plt.scatter(embedding[:, 1], embedding[:, 0],
        #             alpha=alpha_inside, s=point_size, edgecolor="none",c=detail[detail['gene_name'].isin(genelist)].alpha_new, cmap=colormap,vmin=v_min,vmax=v_max)
        #u_s= np.array(detail[detail['gene_name'].isin(gene)][["u0","s0","u1","s1"]])
        #u_s= np.array(detail[detail['gene_name'].isin(gene)][["u0","s0","u1","s1"]])

        plt.xlim(-0.05*s0_max, x_max) 
        plt.ylim(-0.05*u0_max, y_max) 

        title_info=gene
        if (cluster_info is not None) and (mode == 'cluster'):
            
            if colors is not None:
                colors=colors
            else:
                colors = {'CA':grove2[6],
                            'CA1-Sub':grove2[8],
                            'CA2-3-4':grove2[7],
                            'Granule':grove2[5],
                            'ImmGranule1':grove2[5],
                            'ImmGranule2':grove2[5],
                            'Nbl1':grove2[4],
                            'Nbl2':grove2[4],
                            'nIPC':grove2[3],
                            'RadialGlia':grove2[2],
                            'RadialGlia2':grove2[2],
                            'GlialProg' :grove2[2],
                            'OPC':grove2[1],
                            'ImmAstro':grove2[0]}
            custom_map=cluster_info.map(colors)
            title_info=gene
            layer1=plt.scatter(u_s[:, 1], u_s[:, 0],
                alpha=alpha_inside, s=point_size, edgecolor="none",
                #c=pd.factorize(cluster_info)[0], 
                c=custom_map)
            
            if cluster_annot:
                from matplotlib.lines import Line2D
                def gen_Line2D(label, markerfacecolor):
                    return Line2D([0], [0], color='w', marker='o', label=label, markerfacecolor=markerfacecolor,  markeredgewidth=0, markersize=5)
                legend_elements = []
                for i in colors:
                    legend_elements.append(gen_Line2D(i, colors[i]))
                lgd=plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')

            #plt.colorbar(layer1)
        # para in gene velocity # not using
        # para=None,colormap=None
        # elif (colormap is not None) and (para is not None):
        #     title_info=gene+' '+para
        #     layer1=plt.scatter(u_s[:, 1], u_s[:, 0],
        #         alpha=1, s=point_size, edgecolor="none",
        #         c=detail[detail['gene_name']==gene][para],cmap=colormap,vmin=v_min,vmax=v_max)
        #     plt.colorbar(layer1)
        elif color_scatter is not None:
            layer1=plt.scatter(u_s[:, 1], u_s[:, 0],
                    alpha=alpha_inside, s=point_size, edgecolor="none",color=color_scatter,vmin=v_min,vmax=v_max)
        if show_arrow:
            plt.scatter(u_s_downsample[:, 1], u_s_downsample[:, 0], # sampled circle
                        color="none",s=point_size, edgecolor="k")
            pcm1 = plt.quiver(
            u_s_downsample[:, 1], u_s_downsample[:, 0], u_s_downsample[:, 3]-u_s_downsample[:, 1], u_s_downsample[:, 2]-u_s_downsample[:, 0],
            angles='xy', clim=(0., 1.))
        plt.title(title_info)
        if save_path is not None:
            if cluster_annot:
                plt.savefig(save_path,bbox_inches='tight',bbox_extra_artists=(lgd,),)
            else:
                plt.savefig(save_path)

    def vaildation_plot(gene,validation_result,save_path_validation=None):
        '''gene validation plot
        TO DO: will remove validation in the future
        '''
        plt.figure()
        plt.scatter(validation_result.epoch, validation_result.cost)
        plt.title(gene)
        if save_path_validation is not None:
            plt.savefig(save_path_validation)

    def cell_level_para_plot(read_raw_data,detail,gene_choice,para_list,cluster_choice=None,save_path=None,pointsize=0.2,alpha=1):
        
        '''plot alpha, beta, gamma, s0, u0 in cell level'''

        if cluster_choice is not None:
            reindexed_one_gene=read_raw_data[read_raw_data.gene_list==gene_choice[0]].reset_index()
            embedding=reindexed_one_gene[reindexed_one_gene.clusters.isin(cluster_choice)][['embedding1','embedding2']].to_numpy()
        else:
            embedding=read_raw_data[read_raw_data.gene_list==gene_choice[0]][['embedding1','embedding2']].to_numpy()
        
        color_map_zissou2 = LinearSegmentedColormap.from_list("mycmap", zissou2)
        color_map_fireworks3 = LinearSegmentedColormap.from_list("mycmap", fireworks3)
        color_dict = {'alpha':color_map_zissou2,'beta':color_map_zissou2,'gamma':color_map_zissou2,'s0':color_map_fireworks3,'u0':color_map_fireworks3}

        for para in para_list:
            for gene_name in gene_choice:
                #gene_name='Ntrk2'

                one_gene=detail[detail.gene_name==gene_name].reset_index()
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
