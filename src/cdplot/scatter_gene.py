import matplotlib.pyplot as plt
import os
import sys

if __name__ == "__main__":
    sys.path.append('..')
    from colormap import *
else:
    try:
        from ..colormap import *
    except ImportError:
        from colormap import *

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

