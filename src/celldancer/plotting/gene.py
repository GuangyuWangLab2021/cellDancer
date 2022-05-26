import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from .colormap import *
from ..sampling import sampling_neighbors
from ..utilities import extract_from_df

def scatter_gene(
    ax=None,
    x=None,
    y=None,
    cellDancer_df=None,
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
    arrow_grid = (15,15),
    save_path=None):

"""Plot the plot of spliced-unspliced of a gene, or plot the parameter ('alpha', 'beta', 'gamma', 'spliced', 'unspliced') in pseudotime, or customize the parameters in x-axis and y-axis of a gene.
    
Arguments
---------
ax: `ax of plt.subplots()`
    ax to add subplot.
x: `str`
    one of {'s0','u0','s1','u1','alpha','beta','gamma','pseudotime}
y: `str`
    one of {'s0','u0','s1','u1','alpha','beta','gamma','pseudotime}
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
    True if velocity in gene level is to be plotted.
gene: `str` (default: None)
    The gene been selected for the plot of alpha, beta, gamma, spliced, or unspliced in embedding level.
legend: `str` (default: 'off')
    'off' if the color map of cell tyoe legend is not to be plotted.
    'only' if to only plot the cell type legend.
arrow_grid: `tuple` (default: (15,15))
    The size of the grid for the gene velocity to display.
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

        c=np.vectorize(colors.get)(extract_from_df(cellDancer_df, 'clusters'))
        cmap=ListedColormap(list(colors.keys()))

    elif isinstance(colors, str):
        attr = colors
        if colors in ['alpha', 'beta', 'gamma']:
            assert gene, '\nError! gene is required!\n'
            cmap = ListedColormap(zissou2)
        if colors in ['spliced', 'unspliced']:
            assert gene, '\nError! gene is required!\n'
            colors = {'spliced':'s0', 'unspliced':'u0'}[colors]
            cmap = ListedColormap(fireworks3)
        if colors in ['pseudotime']:
            cmap = 'viridis'
        else:
            cmap = 'viridis'

        c = extract_from_df(cellDancer_df, [colors], gene)
    elif colors is None:
        attr = 'basic'
        cmap = None
        c = '#95D9EF'
    
    
    if x in ['spliced', 'unspliced']:
        x = {'spliced':'s0', 'unspliced':'u0'}[x]
    if y in ['spliced', 'unspliced']:
        y = {'spliced':'s0', 'unspliced':'u0'}[y]


    assert gene, '\nError! gene is required!\n'
    xy = extract_from_df(cellDancer_df, [x, y], gene)
    ax.scatter(xy[:, 0],
               xy[:, 1],
               c=c,
               cmap=cmap,
               s=s,
               alpha=alpha,
               vmin=vmin,
               vmax=vmax,
               edgecolor="none")

    if custom_xlim is not None:
        ax.set_xlim(custom_xlim[0], custom_xlim[1])
    if custom_ylim is not None:
        ax.set_ylim(custom_ylim[0], custom_ylim[1])

                                 
    if velocity:
        assert (x,y) in [('u0', 's0'), ('s0', 'u0')]
        u_s = extract_from_df(cellDancer_df, ['u0','s0','u1','s1'], gene)
        sampling_idx=sampling_neighbors(u_s[:,0:2], step=arrow_grid, percentile=15) # Sampling
        u_s_downsample = u_s[sampling_idx,0:4]

        plt.scatter(u_s_downsample[:, 1], u_s_downsample[:,0], color="none", s=s, edgecolor="k")
        plt.quiver(u_s_downsample[:, 1], u_s_downsample[:, 0], 
                   u_s_downsample[:, 3]-u_s_downsample[:, 1], 
                   u_s_downsample[:, 2]-u_s_downsample[:, 0],
                   angles='xy', clim=(0., 1.))
            
    if save_path is not None:
        file_name_parts = [y+'-'+x, 'coloring-'+attr, gene]
        if velocity:
            file_name_parts.insert(0, 'velocity')

        save_file_name = os.path.join(save_path, "_".join(file_name_parts)+'.pdf')

        print("saved the file as", save_file_name)
        plt.savefig(save_file_name,
                bbox_inches='tight',
                bbox_extra_artists=bbox_extra_artists)

    return ax

