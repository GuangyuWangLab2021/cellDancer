# functions borrowed from scv; 
# TO DO: Change code style for every functions!!!!!!

import numpy as np
from scipy.sparse import csr_matrix

def find_neighbors(adata, n_pcs=30, n_neighbors=30):
    '''Find neighbors by using pca on UMAP'''
    from scanpy import Neighbors
    import warnings

    neighbors = Neighbors(adata)
    with warnings.catch_warnings():  # ignore numba warning (umap/issues/252)
        warnings.simplefilter("ignore")
        neighbors.compute_neighbors(
            n_neighbors=n_neighbors,
            knn=True,
            n_pcs=n_pcs,
            method="umap",
            use_rep="X_pca",
            random_state=0,
            metric="euclidean",
            metric_kwds={},
            write_knn_indices=True,
        )

    adata.obsp["distances"] = neighbors.distances
    adata.obsp["connectivities"] = neighbors.connectivities
    adata.uns["neighbors"]["connectivities_key"] = "connectivities"
    adata.uns["neighbors"]["distances_key"] = "distances"

    if hasattr(neighbors, "knn_indices"):
        adata.uns["neighbors"]["indices"] = neighbors.knn_indices
        adata.uns["neighbors"]["params"] = {
            "n_neighbors": n_neighbors,
            "method": "umap",
            "metric": "euclidean",
            "n_pcs": n_pcs,
            "use_rep": "X_pca",
        }

def moments(adata):
    '''Calculate moments'''
    connect = adata.obsp['connectivities'] > 0
    connect.setdiag(1)
    connect = connect.multiply(1.0 / connect.sum(1))
    #pd.DataFrame(connect.todense(), adata.obs.index.tolist(), adata.obs.index.tolist())
    #pd.DataFrame(connect.multiply(1.0 / connect.sum(1)).dot(adata.layers['unspliced'].todense()), 
    #adata.obs.index.tolist(), adata.var.index.tolist())
    adata.layers["Mu"] = csr_matrix.dot(connect, csr_matrix(adata.layers["unspliced"])).astype(np.float32).A
    adata.layers["Ms"] = csr_matrix.dot(connect, csr_matrix(adata.layers["spliced"])).astype(np.float32).A

def set_rcParams(fontsize=12): 
    try:
        import IPython
        ipython_format = ["png2x"]
        IPython.display.set_matplotlib_formats(*ipython_format)
    except:
        pass

    from matplotlib import rcParams

    # dpi options (mpl default: 100, 100)
    rcParams["figure.dpi"] = 100
    rcParams["savefig.dpi"] = 150

    # figure (mpl default: 0.125, 0.96, 0.15, 0.91)
    rcParams["figure.figsize"] = (6, 4)
    rcParams["figure.subplot.left"] = 0.18
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.91

    # lines (defaults:  1.5, 6, 1)
    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]

    fontsize = fontsize
    labelsize = 0.92 * fontsize

    # fonsizes (mpl default: 10, medium, large, medium)
    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = labelsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = labelsize

    # legend (mpl default: 1, 1, 2, 0.8)
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4

    # axes
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks (mpl default: k, k, medium, medium)
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = labelsize
    rcParams["ytick.labelsize"] = labelsize

    # axes grid (mpl default: False, #b0b0b0)
    rcParams["axes.grid"] = False
    rcParams["grid.color"] = ".8"

    # color map
    rcParams["image.cmap"] = "RdBu_r"



if __name__ == "__main__":
    import sys
    sys.path.append('.')