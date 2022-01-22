# when importing meet error https://www.jianshu.com/p/216b88d5d213
# solution: conda install numba=0.51.0

# https://github.com/canzarlab/JVis-learn 
# https://github.com/canzarlab/JVis_paper/blob/master/sc_velocity/j-UMAP/CITE_seq-JUMAP.ipynb
from Jvis import JUMAP, JTSNE 
import numpy as np


# Create a toy example from a random distribution (n_cells = 500)
rna_rand = np.random.rand(500, 100)
adt_rand = np.random.rand(500, 15)
data = {'rna': rna_rand, 'adt': adt_rand} # create a dictionary of modalities.

# Run joint TSNE of the two "random" modalities.
embedding_jtsne = JTSNE(n_components=2).fit_transform(data)

# Run joint UMAP of the two "random" modalities.
embedding_jumap = JUMAP(n_neighbors=20,
                        min_dist=0.3,
                        metric='correlation').fit_transform(data)