
import scvelo as scv
from torch.utils.data import *
from utilities import *
import random
import scipy.stats
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

def adata_scv_pancreas():
    adata = scv.datasets.pancreas()
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    return adata

def load_adata(loom_file="/Users/lingqunye/Desktop/bin/veloNN/tests/data/CH-RNA_10GenSC101.loom"):
    adata = sc.read_loom(loom_file)
    adata.var_names_make_unique()
    return adata

def init_adata(adata):
    # cluster, umap, etc..
    sc.pl.highest_expr_genes(adata, n_top=20, )

    adata.var['mt'] = adata.var.index.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
    adata.var['rp'] = adata.var.index.str.startswith('Rp')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['rp'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True)
    sc.pl.violin(adata, ['pct_counts_mt', 'pct_counts_rp'], jitter=0.4, multi_panel=True)

    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')


    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pl.highly_variable_genes(adata)

    #adata.raw = adata


    #adata = adata[:, adata.var.highly_variable]
    #adata = adata[:, adata.var.highly_variable&(adata.var['rp']==False)&(adata.var['mt'])==False]

    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_rp']) 
    sc.pp.scale(adata, max_value=10)

    # pca
    sc.tl.pca(adata, svd_solver='arpack')

    # Computing the neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    # Embedding the neighborhood graph

    sc.tl.umap(adata, random_state=2)

    # #cluster the neighborhood graph
    sc.tl.leiden(adata, key_added = "leiden") # default resolution in 1.0
    sc.tl.leiden(adata, resolution = 0.6, key_added = "leiden_0.8")
    sc.pl.umap(adata, color="leiden", use_raw=False)


    return adata

def smoothing1(adata):
    find_neighbors(adata, n_pcs=30, n_neighbors=30)
    moments(adata)
    return adata

def knn_distance_matrix(data: np.ndarray, metric: str=None, k: int=40, mode: str='connectivity', n_jobs: int=4):
    nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, )
    nn.fit(data)
    return nn.kneighbors_graph(X=None, mode=mode)
    
def connectivity_to_weights(mknn: sparse.csr.csr_matrix, axis: int=1) -> sparse.lil_matrix:
    """Convert a binary connectivity matrix to weights ready to be multiplied to smooth a data matrix
    """
    if type(mknn) is not sparse.csr.csr_matrix:
        mknn = mknn.tocsr()
    return mknn.multiply(1. / sparse.csr_matrix.sum(mknn, axis=axis))

def convolve_by_sparse_weights(data: np.ndarray, w: sparse.csr_matrix) -> np.ndarray:
    """Use the wights learned from knn to convolve any data matrix

    NOTE: A improved implementation could detect wich one is sparse and wich kind of sparse and perform faster computation
    """
    w_ = w.T
    assert np.allclose(w_.sum(0), 1), "weight matrix need to sum to one over the columns"
    return sparse.csr_matrix.dot(data, w_)

def smoothing2(adata, k=500, n_pca_dims=19, diag: float=1):
    #find_neighbors(adata, n_pcs=30, n_neighbors=30)
    #moments(adata)
    space = adata.obsm['X_pca'][:, :n_pca_dims]
    knn = knn_distance_matrix(space, metric='euclidean', k=k, mode="distance", n_jobs=8)
    connectivity = (knn > 0).astype(float)
    knn_smoothing_w = connectivity_to_weights(connectivity)
    connectivity.setdiag(diag)

    adata.layers['Ms'] = convolve_by_sparse_weights(adata.layers['spliced'].T, knn_smoothing_w).T.todense()
    adata.layers['Mu'] = convolve_by_sparse_weights(adata.layers['unspliced'].T, knn_smoothing_w).T.todense()
    return adata

class realDataset(Dataset):
    def __init__(self, adata=None, loom_file="../../data/loom/DentateGyrus.loom", gene_list=None, smoothing=True, k=500, n_pca_dims=19, pooling=False, pooling_method="binning", pooling_scale=2): #point_number=600 for training
        if adata is not None:
            self.adata = adata
        else:
            self.adata = load_adata(loom_file=loom_file)

        if 'X_pca' not in dict(self.adata.obsm):
            raise RuntimeError("PCA need to be performed before velocity calculation")

        if gene_list is not None:
            self.adata = self.adata[:, self.adata.var.index.isin(gene_list)]

        if smoothing:
            self.adata = smoothing2(self.adata, k=k, n_pca_dims=n_pca_dims)

        self.pooling = pooling
        self.pooling_method = pooling_method
        self.pooling_scale = pooling_scale

    def __len__(self):# name cannot be changed 
        return len(self.adata.var)

    def __getitem__(self, idx):# name cannot be changed 
        adata = self.adata[:, idx]
        u0max = np.max(adata.layers['Mu']).copy().astype(np.float32)
        s0max = np.max(adata.layers['Ms']).copy().astype(np.float32)

        u0 = adata.layers['Mu'][:,0].copy().astype(np.float32)
        s0 = adata.layers['Ms'][:,0].copy().astype(np.float32)

        gene_name = adata.var.index[0]
        
        if self.pooling:
            if self.pooling_method == "binning":
                u0 = np.round(u0/u0max, self.pooling_scale)*u0max
                s0 = np.round(s0/s0max, self.pooling_scale)*s0max
                upoints = np.unique(np.array([u0, s0]), axis=1)
                u0 = upoints[0]
                s0 = upoints[1]
            
            if self.pooling_method == "x^2+y^2=1":
                values = np.vstack([u0,s0])
                kernel = scipy.stats.gaussian_kde(values)
                p = kernel(values)
                idx = np.arange(values.shape[1])
                r = scipy.stats.rv_discrete(values=(idx, (1/p)/sum(1/p)), seed=0)
                pp = r.rvs(size=500)
                u0 = values[0, pp]
                s0 = values[1, pp]

            if self.pooling_method == "y=1/x":
                values = np.vstack([u0,s0])
                kernel = scipy.stats.gaussian_kde(values)
                p = kernel(values)
                idx = np.arange(values.shape[1])
                tmp_p = np.square((1-(p/(max(p)))**2))+0.0001
                # tmp_p = np.square((1-(((p+0.4*max(p))*4-2*max(p+0.4*max(p)))/(2*max(p+0.4*max(p))))**2))+0.0001
                p2 = tmp_p/sum(tmp_p)
                r = scipy.stats.rv_discrete(values=(idx, p2), seed=0)
                pp = r.rvs(size=500)
                u0 = values[0, pp]
                s0 = values[1, pp]

        u1 = np.float32(0)
        s1 = np.float32(0)
        alpha = np.float32(0)
        beta = np.float32(0)
        gamma = np.float32(0)

        type = "real"

        return u0, s0, u1, s1, alpha, beta, gamma, gene_name, type, u0max, s0max

def test_pooling(u0, s0):
    import matplotlib.mlab as mlab
    import scipy.stats
    from sklearn.neighbors import KernelDensity
    from scipy.stats import rv_discrete

    values = np.vstack([u0,s0])
    kernel = scipy.stats.gaussian_kde(values)
    p = kernel(values)
    idx = np.arange(values.shape[1])
    r = scipy.stats.rv_discrete(values=(idx, (1/p)/sum(1/p)))
    pp = r.rvs(size=100)
    new_u0 = values[0, pp]
    new_s0 = values[1, pp]

    plt.scatter(values[0, :], values[1, :], c=p.T)
    plt.show()
    plt.scatter(values[0, pp], values[1, pp], c=p.T[pp])
    plt.show()

    return new_u0, new_s0

def test_smoothie():
    gene_list=["Abcc8"]
    for k in [1, 30, 100, 1000]:
        data = realDataset(adata = adata_scv_pancreas(), k=k, gene_list=gene_list)
        for i in range(len(gene_list)):
            u0, s0, u1, s1, alpha, beta, gamma, gene_name, type1, u0max, s0max = data.__getitem__(i)
            print(u0, s0)
            plt.scatter(s0, u0, s=1)
            plt.title(gene_name)
            plt.show()

    gene_list=["Abcc8", "Rbfox3", "Sulf2", "Wfdc15b"]
    for k in [1]:
        data = realDataset(adata = adata_scv_pancreas(), k=k, gene_list=gene_list)
        for i in range(len(gene_list)):
            u0, s0, u1, s1, alpha, beta, gamma, gene_name, type1, u0max, s0max = data.__getitem__(i)
            print(u0, s0)
            plt.scatter(s0, u0, s=1)
            plt.title(gene_name)
            plt.show()

def test_sampling():
    gene_list=["Rbfox3"]
    for pooling_method in ["binning", "x^2+y^2=1", "y=1/x"]:
        data = realDataset(adata = adata_scv_pancreas(), k=100, pooling=True, pooling_method=pooling_method, gene_list=gene_list)
        for i in range(len(gene_list)):
            u0, s0, u1, s1, alpha, beta, gamma, gene_name, type1, u0max, s0max = data.__getitem__(i)
            print(u0, s0)
            plt.scatter(s0, u0, s=1)
            plt.title(gene_name)
            plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utilities import set_rcParams
    set_rcParams()

    gene_list = ["Pdgfra", "Igfbpl1", "Ntrk2", "Syngr1"]
    #adata0 = load_adata(loom_file="../../data/loom/DentateGyrus.loom")
    #adata = init_adata(adata0)
    #adata.write_loom(filename="../../data/loom/DentateGyrus_pca.loom", write_obsm_varm=True)
    adata = load_adata(loom_file="../../data/loom/DentateGyrus_pca.loom")
    data = realDataset(adata = adata, k=500, gene_list=gene_list)
    #data = realDataset(loom_file="../../data/loom/DentateGyrus_pca.loom", k=1000, gene_list=gene_list)
    for i in range(len(gene_list)):
        u0, s0, u1, s1, alpha, beta, gamma, gene_name, type1, u0max, s0max = data.__getitem__(i)
        plt.scatter(s0, u0, s=1)
        plt.title(gene_name)
        plt.show()

    #-------------------

    #["Pcsk2", "Gng12", "Gnao1", "Top2a", "Cpe", "Adk", "Actn4", "Rap1b"]
    gene_list=["Abcc8", "Cdk1", "Nfib", "Rbfox3", "Sulf2", "Wfdc15b"]
    data = realDataset(adata = adata_scv_pancreas(), k=100, gene_list=gene_list)
    for i in range(len(gene_list)):
        u0, s0, u1, s1, alpha, beta, gamma, gene_name, type1, u0max, s0max = data.__getitem__(i)
        plt.scatter(s0, u0, s=1)
        plt.title(gene_name)
        plt.show()
