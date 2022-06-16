import pandas as pd 
import numpy as np
import os

import networkx as nx
from datashader.layout import forceatlas2_layout
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, bundle_graph
from holoviews.element.graphs import layout_nodes
hv.extension('bokeh')

from .colormap import *
if __name__ == "__main__":# developer test
    sys.path.append('..')
    from utilities import extract_from_df
else:
    from celldancer.utilities import extract_from_df

def graph(
        cellDancer_df,
        node_layout='forceatlas2',
        use_edge_bundling=True,
        node_colors=None,
        node_size=5,
        edge_length=None,
        save_path=None):

    """ Graph visualization of selected cells reflecting their orders in pseudotime 
        
    Arguments
    ---------
    cellDancer_df: `pandas.DataFrame`
        Data frame of velocity estimation, cell velocity, and pseudotime results. 
        Columns=['cellIndex', 'gene_name', 
        'unsplice', 'splice', 
        'unsplice_predict', 'splice_predict', 
        'alpha', 'beta', 'gamma', 
        'loss', 'cellID', 'clusters', 'embedding1', 'embedding2', 
        'velocity1', 'velocity2', 'pseudotime']
    node_layout: optional, `str` (default: forceatlas2)
         Layout for the graph. Currently only supports the forceatlas2 and
         embedding. 

         - `'forceatlas2'` or `'forcedirected'`: treat connections as forces
         between connected nodes.
         - `'embedding'`: use the embedding as positions of the nodes.

    use_edge_bundling: optional, `bool` (default: `True`)
        `True` if bundle the edges (computational demanding). 
        Edge bundling allows edges to curve and groups nearby ones together 
        for better visualization of the graph structure. 
    node_colors: optional, `str` (default: `None`)
        The node colors. 
        Possible values:

            - 'clusters': color according to the clusters information of the
              respective cells.
            - 'pseudotime': colors according to the pseudotime of the 
              repspective cells.
            - A single color format string.

    edge_length: optional, `float` (default: None)
        The distance cutoff in the embedding between two nodes to determine 
        whether an edge should be formed (edge is formed when r < *edge_length*).
        By default, the mean of all the cell
        

    node_size: optional, `float` (default: 5)
        The size of the nodes.
    save_path: optional, `str` (default: None)
        The directory to save the plot. `"."` for the current directory.

    Returns
    -------
    graph_plot
        holoviews *DynamicMap* object for the graph.

    """  


    nodes, edges = create_nodes_edges(cellDancer_df, edge_length)
    graph = hv.Graph((edges, nodes))
    kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
    opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))

    if node_layout in ['forceatlas2', 'forcedirected']:
        layout_graph = layout_nodes(
                graph,
                layout=forceatlas2_layout,
                kwargs={'k':0.01,
                        'x':'x',
                        'y':'y',
                        'weight':'weight',
                        'iterations':1000,
                        'seed':0})
    elif node_layout in ['embedding']:
        layout_graph = graph
    else:
        layout_graph = graph

    if use_edge_bundling:
        bundled = bundle_graph(layout_graph)
    else:
        bundled = layout_graph

    if node_colors in ['clusters']:
        cmap = colormap_neuro
    elif node_colors in ['pseudotime']:
        cmap = 'viridis'
    elif node_colors is None:
        cmap = None


    graph_plot=(datashade(bundled, normalization='linear') * bundled.nodes).opts( 
           opts.Nodes(
               color='clusters',
               size=node_size,
               width=800,
               cmap=cmap,
               legend_position='right'))
    if save_path is not None:
        hv.save(graph_plot, save_path+'graph_plot.html')
    return graph_plot

def create_nodes_edges(data, radius):
    def create_KNN_based_graph():
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(radius = radius)
        neigh.fit(embedding_ds)
        nn_graph = neigh.radius_neighbors_graph(embedding_ds, mode='connectivity')
        nn_array = nn_graph.toarray()

        # nn_array is effectively the edge list
        # Keep track of cells of 0 timeshift.
        node_list = [(i, {'pseudotime': pseudotime_ds[i,0], 'clusters':clusters_ds[i]})
                     for i in range(len(embedding_ds))]

        dtime = pseudotime_ds[:,0] - pseudotime_ds
        INF = 1./np.min(np.abs(dtime[dtime != 0]))

        # upper triangle of the knn array (i<j and nn_array[i,j] = 1)
        edge_filter = np.triu(nn_array, k=1)
        (i,j) = np.where(edge_filter != 0)

        # for forcedirected layouts,
        # edge length is positively correlated with weight.
        # hence 1/dtime here as the weight
        # Created for directed graph
        edge_list = list()
        for a,b,w in zip(i,j, dtime[i,j]):
            if w>0:
                edge_list.append((a, b, 1/w))
            elif w<0:
                edge_list.append((a, b, -1/w))
            else:
                edge_list.append((a, b, INF))

        G = nx.Graph()
        G.add_nodes_from(node_list)
        G.add_weighted_edges_from(edge_list)
        return G

    embedding = extract_from_df(data, ['embedding1', 'embedding2'])
    n_cells = embedding.shape[0]
    sample_cells = data['velocity1'][:n_cells].dropna().index
    clusters = extract_from_df(data, ['clusters'])
    pseudotime = extract_from_df(data, ['pseudotime'])

    embedding_ds = embedding[sample_cells]
    pseudotime_ds = pseudotime[sample_cells]
    clusters_ds = clusters[sample_cells]

    G = create_KNN_based_graph()

    # Gosh! holowviews.Graph
    index = np.array(range(len(embedding_ds)), dtype=int)[:,None]
    nodes = pd.DataFrame(np.hstack((embedding_ds, index, pseudotime_ds, clusters_ds)),
                         columns=['x','y','index','pseudotime','clusters'])

    edges = pd.DataFrame([(i[0], i[1], G.edges[i]['weight']) for i in G.edges],
                         columns=['source', 'target', 'weight'])
    return nodes, edges
