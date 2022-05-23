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
        load_cellDancer,
        node_layout='forceatlas2',
        edge_bundling=True,
        node_colors=None,
        s=5,
        edge_length=5,
        legend='off',
        save_path=None):

    nodes, edges = create_nodes_edges(load_cellDancer, edge_length)
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

    if edge_bundling:
        bundled = bundle_graph(layout_graph)
    else:
        bundled = layout_graph

    graph_plot=(datashade(bundled, normalization='linear') * bundled.nodes).opts( 
           opts.Nodes(
               color='clusters',
               size=5,
               width=800,
               cmap=colormap_neuro,
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
        node_list = [(i, {'ptime': ptime_ds[i,0], 'cluster':clusters_ds[i]})
                     for i in range(len(embedding_ds))]

        dtime = ptime_ds[:,0] - ptime_ds
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
    ptime = extract_from_df(data, ['pseudotime'])

    embedding_ds = embedding[sample_cells]
    ptime_ds = ptime[sample_cells]
    clusters_ds = clusters[sample_cells]

    G = create_KNN_based_graph()

    # Gosh! holowviews.Graph
    index = np.array(range(len(embedding_ds)), dtype=int)[:,None]
    nodes = pd.DataFrame(np.hstack((embedding_ds, index, ptime_ds, clusters_ds)),
                         columns=['x','y','index','ptime','clusters'])

    edges = pd.DataFrame([(i[0], i[1], G.edges[i]['weight']) for i in G.edges],
                         columns=['source', 'target', 'weight'])
    return nodes, edges
