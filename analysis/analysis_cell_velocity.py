import pandas as pd
import numpy as np
import warnings
from scipy.sparse import coo_matrix, csr_matrix, issparse

#tools/utils.py
def get_iterative_indices(
    indices,
    index,
    n_recurse_neighbors=2,
    max_neighs=None,
):
    def iterate_indices(indices, index, n_recurse_neighbors):
        if n_recurse_neighbors > 1:
            index = iterate_indices(indices, index, n_recurse_neighbors - 1)
        ix = np.append(index, indices[index])  # direct and indirect neighbors
        if np.isnan(ix).any():
            ix = ix[~np.isnan(ix)]
        return ix.astype(int)

    indices = np.unique(iterate_indices(indices, index, n_recurse_neighbors))
    if max_neighs is not None and len(indices) > max_neighs:
        indices = np.random.choice(indices, max_neighs, replace=False)
    return indices

#tools/utils.py
def norm(A):
    """computes the L2-norm along axis 1
    (e.g. genes or embedding dimensions) equivalent to np.linalg.norm(A, axis=1)
    """
    if issparse(A):
        return np.sqrt(A.multiply(A).sum(1).A1)
    else:
        return np.sqrt(np.einsum("ij, ij -> i", A, A) if A.ndim > 1 else np.sum(A * A))

#tools/utils.py
def vector_norm(x):
    """computes the L2-norm along axis 1
    (e.g. genes or embedding dimensions) equivalent to np.linalg.norm(A, axis=1)
    """
    return np.sqrt(np.einsum("i, i -> ", x, x))

#tools/utils.py
def sum_var(A):
    """summation over axis 1 (var) equivalent to np.sum(A, 1)"""
    if issparse(A):
        return A.sum(1).A1
    else:
        return np.sum(A, axis=1) if A.ndim > 1 else np.sum(A)

#tools/utils.py
def cosine_correlation(dX, Vi):
    dx = dX - dX.mean(-1)[:, None]
    Vi_norm = vector_norm(Vi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if Vi_norm == 0:
            result = np.zeros(dx.shape[0])
        else:
            result = np.einsum("ij, j", dx, Vi) / (norm(dx) * Vi_norm)[None, :]
    return result

#tools/utils.py
def get_indices_from_csr(conn):
    # extracts indices from connectivity matrix, pads with nans
    ixs = np.ones((conn.shape[0], np.max((conn > 0).sum(1)))) * np.nan
    for i in range(ixs.shape[0]):
        cell_indices = conn[i, :].indices
        ixs[i, : len(cell_indices)] = cell_indices
    return ixs

#tools/utils.py
def get_indices(dist, n_neighbors=None, mode_neighbors="distances"):
    D = dist.copy()
    D.data += 1e-6

    n_counts = sum_var(D > 0)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(), n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = D.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[n_neighbors:]
        dat[rm_idx] = 0
    D.eliminate_zeros()

    D.data -= 1e-6
    if mode_neighbors == "distances":
        indices = D.indices.reshape((-1, n_neighbors))
    elif mode_neighbors == "connectivities":
        knn_indices = D.indices.reshape((-1, n_neighbors))
        knn_distances = D.data.reshape((-1, n_neighbors))
        _, conn = compute_connectivities_umap(
            knn_indices, knn_distances, D.shape[0], n_neighbors
        )
        indices = get_indices_from_csr(conn)
    return indices, D

#preprocessing/neighbors.py
def get_neighs(adata, mode="distances"):
    if hasattr(adata, "obsp") and mode in adata.obsp.keys():
        return adata.obsp[mode]
    elif "neighbors" in adata.uns.keys() and mode in adata.uns["neighbors"]:
        return adata.uns["neighbors"][mode]
    else:
        raise ValueError("The selected mode is not valid.")

#preprocessing/neighbors.py
def compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_obs,
    n_neighbors,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """\
    This is from umap.fuzzy_simplicial_set [McInnes18]_.
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """
    from umap.umap_ import fuzzy_simplicial_set

    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):  # umap returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    distances = get_csr_from_indices(knn_indices, knn_dists, n_obs, n_neighbors)

    return distances, connectivities.tocsr()

#preprocessing/neighbors.py
def get_csr_from_indices(knn_indices, knn_dists, n_obs, n_neighbors):
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # we didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()

#tools/velocity_graph.py
def vals_to_csr(vals, rows, cols, shape, split_negative=False):
    graph = coo_matrix((vals, (rows, cols)), shape=shape)

    if split_negative:
        graph_neg = graph.copy()

        graph.data = np.clip(graph.data, 0, 1)
        graph_neg.data = np.clip(graph_neg.data, -1, 0)

        graph.eliminate_zeros()
        graph_neg.eliminate_zeros()

        return graph.tocsr(), graph_neg.tocsr()

    else:
        return graph.tocsr()

#tools/velocity_graph.py
class VelocityGraph:
    def __init__(
        self,
        adata,
        vkey="velocity",
        xkey="Ms",
        tkey=None,
        basis=None,
        n_neighbors=None,
        sqrt_transform=None,
        n_recurse_neighbors=None,
        random_neighbors_at_max=None,
        gene_subset=None,
        approx=None,
        report=False,
        compute_uncertainties=None,
        mode_neighbors="distances",
    ):

        subset = np.ones(adata.n_vars, bool)
        if gene_subset is not None:
            var_names_subset = adata.var_names.isin(gene_subset)
            subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
        elif f"{vkey}_genes" in adata.var.keys():
            subset &= np.array(adata.var[f"{vkey}_genes"].values, dtype=bool)

        xkey = xkey if xkey in adata.layers.keys() else "spliced"

        X = np.array(
            adata.layers[xkey].A[:, subset]
            if issparse(adata.layers[xkey])
            else adata.layers[xkey][:, subset]
        )
        V = np.array(
            adata.layers[vkey].A[:, subset]
            if issparse(adata.layers[vkey])
            else adata.layers[vkey][:, subset]
        )

        nans = np.isnan(np.sum(V, axis=0))
        if np.any(nans):
            X = X[:, ~nans]
            V = V[:, ~nans]

        if approx is True and X.shape[1] > 100:
            X_pca, PCs, _, _ = pca(X, n_comps=30, svd_solver="arpack", return_info=True)
            self.X = np.array(X_pca, dtype=np.float32)
            self.V = (V - V.mean(0)).dot(PCs.T)
            self.V[V.sum(1) == 0] = 0
        else:
            self.X = np.array(X, dtype=np.float32)
            self.V = np.array(V, dtype=np.float32)
        self.V_raw = np.array(self.V)

        self.sqrt_transform = sqrt_transform
        if self.sqrt_transform is None and f"{vkey}_params" in adata.uns.keys():
            self.sqrt_transform = adata.uns[f"{vkey}_params"]["mode"] == "stochastic"
        if self.sqrt_transform:
            self.V = np.sqrt(np.abs(self.V)) * np.sign(self.V)
        self.V -= np.nanmean(self.V, axis=1)[:, None]

        self.n_recurse_neighbors = n_recurse_neighbors
        if self.n_recurse_neighbors is None:
            if n_neighbors is not None or mode_neighbors == "connectivities":
                self.n_recurse_neighbors = 1
            else:
                self.n_recurse_neighbors = 2

        #if "neighbors" not in adata.uns.keys():
        #    neighbors(adata)
        if np.min((get_neighs(adata, "distances") > 0).sum(1).A1) == 0:
            raise ValueError(
                "Your neighbor graph seems to be corrupted. "
                "Consider recomputing via pp.neighbors."
            )
        if n_neighbors is None: #or n_neighbors <= get_n_neighs(adata)
            self.indices = get_indices(
                dist=get_neighs(adata, "distances"),
                n_neighbors=n_neighbors,
                mode_neighbors=mode_neighbors,
            )[0]
        else:
            if basis is None:
                basis_keys = ["X_pca", "X_tsne", "X_umap"]
                basis = [key for key in basis_keys if key in adata.obsm.keys()][-1]
            elif f"X_{basis}" in adata.obsm.keys():
                basis = f"X_{basis}"

            if isinstance(approx, str) and approx in adata.obsm.keys():
                from sklearn.neighbors import NearestNeighbors

                neighs = NearestNeighbors(n_neighbors=n_neighbors + 1)
                neighs.fit(adata.obsm[approx])
                self.indices = neighs.kneighbors_graph(
                    mode="connectivity"
                ).indices.reshape((-1, n_neighbors + 1))
            else:
                from .. import Neighbors

                neighs = Neighbors(adata)
                neighs.compute_neighbors(
                    n_neighbors=n_neighbors, use_rep=basis, n_pcs=10
                )
                self.indices = get_indices(
                    dist=neighs.distances, mode_neighbors=mode_neighbors
                )[0]

        self.max_neighs = random_neighbors_at_max

        gkey, gkey_ = f"{vkey}_graph", f"{vkey}_graph_neg"
        self.graph = adata.uns[gkey] if gkey in adata.uns.keys() else []
        self.graph_neg = adata.uns[gkey_] if gkey_ in adata.uns.keys() else []

        if tkey in adata.obs.keys():
            self.t0 = adata.obs[tkey].copy()
            init = min(self.t0) if isinstance(min(self.t0), int) else 0
            self.t0.cat.categories = np.arange(init, len(self.t0.cat.categories))
            self.t1 = self.t0.copy()
            self.t1.cat.categories = self.t0.cat.categories + 1
        else:
            self.t0 = None

        self.compute_uncertainties = compute_uncertainties
        self.uncertainties = None
        self.self_prob = None
        self.report = report
        self.adata = adata

    def compute_cosines(self):
        vals, rows, cols, uncertainties, n_obs = [], [], [], [], self.X.shape[0]
        #progress = logg.ProgressReporter(n_obs)

        #if self.compute_uncertainties:
        #    m = get_moments(self.adata, np.sign(self.V_raw), second_order=True)

        for i in range(n_obs):
            if self.V[i].max() != 0 or self.V[i].min() != 0:
                neighs_idx = get_iterative_indices(
                    self.indices, i, self.n_recurse_neighbors, self.max_neighs
                )

                if self.t0 is not None:
                    t0, t1 = self.t0[i], self.t1[i]
                    if t0 >= 0 and t1 > 0:
                        t1_idx = np.where(self.t0 == t1)[0]
                        if len(t1_idx) > len(neighs_idx):
                            t1_idx = np.random.choice(
                                t1_idx, len(neighs_idx), replace=False
                            )
                        if len(t1_idx) > 0:
                            neighs_idx = np.unique(np.concatenate([neighs_idx, t1_idx]))

                dX = self.X[neighs_idx] - self.X[i, None]  # 60% of runtime
                if self.sqrt_transform:
                    dX = np.sqrt(np.abs(dX)) * np.sign(dX)
                val = cosine_correlation(dX, self.V[i])  # 40% of runtime #ylq: cosine

                if self.compute_uncertainties:
                    dX /= norm(dX)[:, None]
                    uncertainties.extend(np.nansum(dX ** 2 * m[i][None, :], 1))

                vals.extend(val)
                rows.extend(np.ones(len(neighs_idx)) * i)
                cols.extend(neighs_idx)
                #if self.report:
                #    progress.update()
        #if self.report:
        #    progress.finish()

        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0

        self.graph, self.graph_neg = vals_to_csr(
            vals, rows, cols, shape=(n_obs, n_obs), split_negative=True
        )
        if self.compute_uncertainties:
            uncertainties = np.hstack(uncertainties)
            uncertainties[np.isnan(uncertainties)] = 0
            self.uncertainties = vals_to_csr(
                uncertainties, rows, cols, shape=(n_obs, n_obs), split_negative=False
            )
            self.uncertainties.eliminate_zeros()

        confidence = self.graph.max(1).A.flatten()
        self.self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)

#tools/velocity_graph.py
def velocity_graph(
    adata,
    vkey="velocity",
    xkey="Ms",
    tkey=None,
    basis=None,
    n_neighbors=None,
    n_recurse_neighbors=None,
    random_neighbors_at_max=None,
    sqrt_transform=None,
    variance_stabilization=None,
    gene_subset=None,
    compute_uncertainties=None,
    approx=None,
    mode_neighbors="distances",
):

    if 'velocity_graph' in adata._uns.keys():
        del adata._uns['velocity_graph']
    if 'velocity_graph_neg' in adata._uns.keys():   
        del adata._uns['velocity_graph_neg']
    if 'velocity_params' in adata._uns.keys():     
        del adata._uns['velocity_params']

    vgraph = VelocityGraph(
        adata,
        vkey=vkey,
        xkey=xkey,
        tkey=tkey,
        basis=basis,
        n_neighbors=n_neighbors,
        approx=approx,
        n_recurse_neighbors=n_recurse_neighbors,
        random_neighbors_at_max=random_neighbors_at_max,
        sqrt_transform=sqrt_transform,
        gene_subset=gene_subset,
        compute_uncertainties=compute_uncertainties,
        report=True,
        mode_neighbors=mode_neighbors,
    )
    vgraph.compute_cosines()

    adata.uns[f"{vkey}_graph"] = vgraph.graph
    adata.uns[f"{vkey}_graph_neg"] = vgraph.graph_neg

    if vgraph.uncertainties is not None:
        adata.uns[f"{vkey}_graph_uncertainties"] = vgraph.uncertainties

    adata.obs[f"{vkey}_self_transition"] = vgraph.self_prob

    if f"{vkey}_params" in adata.uns.keys():
        if "embeddings" in adata.uns[f"{vkey}_params"]:
            del adata.uns[f"{vkey}_params"]["embeddings"]
    else:
        adata.uns[f"{vkey}_params"] = {}
    adata.uns[f"{vkey}_params"]["mode_neighbors"] = mode_neighbors
    adata.uns[f"{vkey}_params"]["n_recurse_neighbors"] = vgraph.n_recurse_neighbors

    return adata

#read detail
def read_detail(file):
    return pd.read_csv(file)

def filter_velocity(adata, gene_list):

    nan_array = np.empty((adata.n_obs,adata.n_vars))
    nan_array[:] = np.NaN
    velocity = pd.DataFrame(nan_array, adata.obs.index.tolist(), adata.var.index.tolist())
    velocity_u = velocity.copy()

    all_velocity = pd.DataFrame(adata.layers['velocity'], adata.obs.index.tolist(), adata.var.index.tolist())
    all_velocity_u =pd.DataFrame(adata.layers['velocity_u'], adata.obs.index.tolist(), adata.var.index.tolist())

    for gene_name in gene_list:
        velocity.loc[:, gene_name] = all_velocity.loc[:, gene_name]
        velocity_u.loc[:, gene_name] = all_velocity_u.loc[:, gene_name]

    adata.layers['velocity'] = np.array(velocity)
    adata.layers['velocity_u'] = np.array(velocity_u)  

# Read velocity from our data
def velocity(adata, detail, dt=0.5):
    nan_array = np.empty((adata.n_obs,adata.n_vars))
    nan_array[:] = np.NaN
    velocity = pd.DataFrame(nan_array, adata.obs.index.tolist(), adata.var.index.tolist())
    velocity_u = velocity.copy()

    gene_list = np.unique(detail['gene_name'])

    for gene_name in gene_list:
        velocity.loc[:, gene_name] = np.array((detail[detail['gene_name']==gene_name].s1 - detail[detail['gene_name']==gene_name].s0)/dt)
        velocity_u.loc[:, gene_name] = np.array((detail[detail['gene_name']==gene_name].u1 - detail[detail['gene_name']==gene_name].u0)/dt)
    #velocity.loc[:, 'Abcc8']
    #velocity_u.loc[:, 'Abcc8']
    adata.layers['velocity'] = np.array(velocity)
    adata.layers['velocity_u'] = np.array(velocity_u)

# Show velocity of a subset of genes 
def show_velocity(data, gene_list=None,name=None):
    adata = data.copy()
    if gene_list is not None:
        filter_velocity(adata, gene_list)
    velocity_graph(adata) #calculate velocity graph
    scv.pl.velocity_embedding_stream(adata) 
    #scv.pl.velocity_embedding(adata)
    #scv.pl.velocity_embedding_grid(adata)

if __name__ == "__main__":
    from utilities import *
    import scvelo as scv
    set_rcParams()

    # Prepare the Data
    adata = scv.datasets.pancreas()
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    find_neighbors(adata, n_pcs=30, n_neighbors=30)
    moments(adata)
    adata_scvelo = adata.copy()
    adata_ours = adata.copy()


    ##########################
    # Gene velocity calculation by using scVelo dynamical model
    scv.tl.recover_dynamics(adata_scvelo, n_jobs=16)
    scv.tl.velocity(adata_scvelo, mode='dynamical')

    ##########################
    # Gene velocity calculation by using our model
    detail = read_detail(file = "/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/veloNN_gw/veloNN-main/src/velonn/output/detailcsv/adj_e/detail_e500.csv")

    detail = read_detail(file = "/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/veloNN-main_github/src/velonn/hpc_job/detailcsv/adj_e/20211217_batch2/scvLr0.001Costv2C1r0.8C2cf0.3Downinverse1000_0_0Ratio0.5OAdam/detail_e500.csv")
    detail = read_detail(file = "/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/veloNN-main_github/src/velonn/hpc_job/detailcsv/adj_e/20211217_batch2/scvLr0.001Costv2C1r0.8C2cf0.3Downneighbors0_100_100Ratio0.5OAdam/detail_e500.csv")
    detail = read_detail(file = "/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/veloNN-main_github/src/velonn/hpc_job/detailcsv/adj_e/20211217_batch2_0/scvLr0.001Costv2C1r0.8C2cf0.3Downneighbors0_100_100Ratio0.5OAdam/detail_e500.csv")
    velocity(adata_ours, detail)

    ##########################
    # Show cell velocity
    show_velocity(adata_scvelo)

    show_velocity(adata_ours, np.unique(detail['gene_name']),name="cell dancer")
    show_velocity(adata_scvelo, np.unique(detail['gene_name']),name="scvelo")
    show_velocity(adata_scvelo, adata_scvelo.var.index,name="scvelo")

    show_velocity(adata_ours, ["Abcc8", "Cpe", "Sulf2", "Wfdc15b", "Cdk1", "Actn4", "Gng12"],name="cell dancer")
    show_velocity(adata_scvelo, ["Abcc8", "Cpe", "Sulf2", "Wfdc15b", "Cdk1", "Actn4", "Gng12"],name="scvelo")

    show_velocity(adata_ours, ["Cpe", "Wfdc15b"],name="cell dancer")
    show_velocity(adata_scvelo, ["Cpe", "Wfdc15b"],name="scvelo")
    
    show_velocity(adata_ours, ["Sulf2","Abcc8"],name="cell dancer")
    show_velocity(adata_scvelo, ["Sulf2","Abcc8"],name="scvelo")
    