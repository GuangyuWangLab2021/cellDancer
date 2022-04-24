import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

if __name__ == "__main__":
    sys.path.append('..')
    from colormap import *
else:
    try:
        from .colormap import *
        print('.colormap')
    except ImportError:
        from colormap import *
        print('.ImportError')

################# gene_pseudotime
def gene_pseudotime(gene,load_cellDancer,cell_time,colors=None,save_path=None):
    
    cell_time_time_sort=cell_time.sort_values('pseudotime')
    cell_time_time_sort.columns=['index','pseudotime']
    
    plt.figure()
    onegene=load_cellDancer[load_cellDancer.gene_name==gene]
    merged=pd.merge(cell_time_time_sort,onegene,left_on='index', right_on='cellIndex') # TODO: NOT cellIndex in the future
    plt.title(gene)
    
    cluster_info=onegene.clusters
    
    # build color map
    if colors is None:
        color_list=grove2.copy()
        cluster_list=onegene.clusters.drop_duplicates()
        from itertools import cycle
        colors = dict(zip(cluster_list, cycle(color_list)) if len(cluster_list) > len(color_list) else zip(cycle(cluster_list), color_list))
    
    # plot legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    def gen_Line2D(label, markerfacecolor):
        return Line2D([0], [0], color='w', marker='o', label=label, markerfacecolor=markerfacecolor,  markeredgewidth=0, markersize=5)
    legend_elements = []
    for i in colors:
        legend_elements.append(gen_Line2D(i, colors[i]))
    lgd=plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    custom_map=merged.clusters.map(colors)
    plt.scatter(merged.pseudotime,merged.s0,c=custom_map,s=3)
    
    if save_path is not None:
        plt.savefig(save_path+gene+'.pdf',bbox_inches='tight',bbox_extra_artists=(lgd,),)
    plt.show()


def gene_list_pseudotime(gene_list,load_cellDancer,cell_time,colors=None,save_path=None):
    for gene in gene_list:
        gene_pseudotime(gene,load_cellDancer,cell_time,colors=colors,save_path=save_path)

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
