#!conda install -c pyviz holoviews bokeh -y
import pandas as pd 
import numpy as np
import os
import networkx as nx

import datashader as ds
import datashader.transfer_functions as tf
from datashader.layout import forceatlas2_layout
from datashader.bundling import connect_edges, hammer_bundle
from colormap import *

detail_result_path = '/Users/pengzhizhang/Documents/test_data/neuro/velocity_result'
output_path = detail_result_path
output_name = 'pseudo_time_neuro_combined.csv'
celltime = pd.read_csv(os.path.join(output_path, output_name))
sub_celltime = celltime[celltime['sample']]

raw_data_path = '/Users/pengzhizhang/Documents/test_data/neuro/raw_data'
embedding_filename = 'denGyr_full.csv'
load_raw_data = pd.read_csv(os.path.join(raw_data_path, embedding_filename))

nsamples = len(sub_celltime)
cellembedding = load_raw_data[load_raw_data.gene_list == 
                          load_raw_data.gene_list[0]][['embedding1', 'embedding2', 'clusters']]

sub_cellembedding = cellembedding[celltime['sample']]
sub_data = pd.concat([sub_celltime, sub_cellembedding], axis=1)

def create_graph_KNN_based(data, n_neighbors, neighborhood_radius):
    from sklearn.neighbors import NearestNeighbors
    cell_embedding = data[['embedding1', 'embedding2']]
    neigh = NearestNeighbors(n_neighbors = n_neighbors, radius = neighborhood_radius)
    neigh.fit(cell_embedding)
    nn_graph = neigh.radius_neighbors_graph(cell_embedding)
    nn_array = nn_graph.toarray()
    edge_filter = np.triu(nn_array, k=1)
    
    # nn_array is effectively the edge list
    # Keep track of cells of 0 timeshift.
    ptime = data['pseudotime'].to_numpy()
    dtime = ptime - ptime[:,np.newaxis]
    
    INF = 1./np.min(np.abs(dtime[dtime!=0]))
    
    node_list = [(i, dict(ptime=data['pseudotime'].iloc[i], 
                          traj_cluster=data['traj_cluster'].iloc[i],
                          cluster=data['clusters'].iloc[i])) for i in range(nsamples)]
    
    # upper triangle of the knn array (i<j and nn_array[i,j] = 1)
    (i,j) = np.where(edge_filter != 0)
     
    # for forcedirected layouts, edge length is positively correlated with weight.
    # hence 1/dtime here as the weight
    # Created for directed graph
    edge_list = list()
    for a,b,w in zip(i,j, dtime[i,j]):
        if w>0:
            edge_list.append((a, b, 1/w))
        elif w<0:
            edge_list.append((a, b, -1/w))
        else:
            #print(a,b)
            edge_list.append((a, b, INF))
            
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_weighted_edges_from(edge_list)
    
    return G


def layout_graph(data, n_neighbors, neighborhood_radius):
    G = create_graph_KNN_based(data, n_neighbors, neighborhood_radius)
    nodes = sub_data[['embedding1', 'embedding2', 'pseudotime', 'clusters']].rename(
        {'embedding1': 'x', 'embedding2':'y'}, axis=1)
    nodes.reset_index(level=0, drop=True,inplace=True)
    nodes['index']=range(len(nodes))
    
    
    # NOTE!!!
    # the third column has to be "index"
    nodes = nodes[['x', 'y', 'index', 'clusters', 'pseudotime']]
    edges = pd.DataFrame([(i[0], i[1], G.edges[i]['weight']) for i in G.edges], 
                         columns=['source', 'target', 'weight'])
    
    forcedirected = forceatlas2_layout(nodes, edges, weight='weight', iterations=200, seed=10)
    fig, ax = plt.subplots(figsize=(10,10))
    plt.scatter(forcedirected.x, forcedirected.y, c=data['clusters'].map(colors), s = 20, zorder=2)
    plt.axis('off')
    plt.show()
    return forcedirected, nodes, edges

# %%
def nodesplot(nodes, name=None, canvas=None, cat=None):
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    aggregator=None if cat is None else ds.count_cat(cat)
    agg=canvas.points(nodes,'x','y',aggregator)
    return tf.spread(tf.shade(agg), px=1, name=name)

def edgesplot(edges, name=None, canvas=None):
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    return tf.shade(canvas.line(edges, 'x','y', agg=ds.count()), name=name)
    
def graphplot(nodes, edges, name="", canvas=None, cat=None):
    if canvas is None:
        xr = nodes.x.min(), nodes.x.max()
        yr = nodes.y.min(), nodes.y.max()
        canvas = ds.Canvas(**cvsopts)
        
    np = nodesplot(nodes, name + " nodes", canvas, cat)
    ep = edgesplot(edges, name + " edges", canvas)
    return tf.stack(ep, np, how="over", name=name)

# %%
cvsopts = dict(plot_height=400, plot_width=400)
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

forcedirected, nodes, edges = layout_graph(sub_data, 10, 5)

# %% [markdown]
# # Add edges
# # origin

# %%
%%script echo for all cells skipping
fd_d = graphplot(forcedirected, connect_edges(forcedirected,edges), "Force-directed")
fd_b = graphplot(forcedirected, hammer_bundle(forcedirected,edges), "Force-directed, bundled") 

tf.Images(fd_d,fd_b).cols(2)

# %%
%%script echo for all cells skipping
all_data = pd.concat([celltime, cellembedding], axis=1)

# Caution: this roughly takes 20+ minutes for 18140 cells.
forcedirected_all, nodes_all, edges_all = layout_graph(all_data, 10, 5)

# %%
%%script echo for all cells skipping
tf.Images(nodesplot(forcedirected_all, "ForceAtlas2 layout"))

# %%
%%script echo for all cells skipping
fig, ax = plt.subplots(figsize=(10,10))
#plt.plot(HB.x, HB.y, lw=0.1, c='lightblue', alpha = 1, zorder=1)
plt.scatter(forcedirected_all.x, forcedirected_all.y, c=all_data['clusters'].map(colors), s = 20, zorder=2)
plt.axis('off')
plt.show()

# %%
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
colormap = list(sub_data['clusters'].map(colors))

# %%
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

kwargs = dict(width=1000, height=1000, xaxis=None, yaxis=None)
opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))

# %%
from holoviews.element.graphs import layout_nodes
layout_nodes(_graph, layout=nx.spring_layout, 
             kwargs={'weight':'weight', 'iterations':200, 'seed':100})

# %%
graph = hv.Graph((edges, forcedirected), label="neuro cell velocity")
graph.opts(node_size=3, edge_line_width=0.1,
            node_line_color='gray', 
            cmap=colors, node_color='clusters',
            edge_alpha=0.1, node_alpha=1)

# %%
%%time 
from holoviews.operation.datashader import datashade, bundle_graph
bundled = bundle_graph(graph)
bundled

# %%
overlay = datashade(bundled, width=800, height=800) * bundled.select(pseudotime=(0,0.05))
overlay.opts(opts.Graph(node_size=10))

# %%
from bokeh.plotting import figure, output_notebook,reset_output, show
from bokeh.models import ColumnDataSource, Arrow, OpenHead, NormalHead, VeeHead

cds = ColumnDataSource(data=dict(x_start=[0,1, 2], y_start=[0,1, 2], x_end=[1,3, 5], y_end=[0,5, 8], line_width=[1]*3, color=['red','blue','yellow']))
arr = Arrow(end=NormalHead(), source=cds, line_color='color', line_width='line_width', x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end')

