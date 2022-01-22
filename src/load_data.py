# organize to model_pl4 or seperate module by module

import scvelo as scv
from torch._C import TracingState
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
    def __init__(self, data_fit=None, data_predict=None,datastatus="predict_dataset", sampling_ratio=1): #point_number=600 for training
        self.data_fit=data_fit
        self.data_predict=data_predict
        self.datastatus=datastatus
        self.sampling_ratio=sampling_ratio
        self.gene_list=list(set(data_fit.gene_list))


    def __len__(self):# name cannot be changed 
        return len(self.gene_list) # gene count

    def __getitem__(self, idx):# name cannot be changed
        gene_name = self.gene_list[idx]
        data_pred=self.data_predict[self.data_predict.gene_list==gene_name] # u0 & s0 for cells for one gene
        #print('gene_name: '+gene_name)
        #print(data_pred)
        # ASK: 在random sampling前还是后,max 决定alpha0，beta0，and gamma0；所以1个gene最好用统一alpha0，beta0，and gamma0
        # 未来可能存在的问题：训练cell，和predict cell的u0和s0重大，不match，若不match？（当前predict cell 里是包含训练cell的，所以暂定用predict的u0max和s0max，如果不包含怎么办？还是在外面算好再传参？）
        u0max = np.float32(max(data_pred["u0"]))
        s0max = np.float32(max(data_pred["s0"]))

        if self.datastatus=="fit_dataset":
            data_fitting=self.data_fit[self.data_fit.gene_list==gene_name] # u0 & s0 for cells for one gene
            # random sampling ratio selection
            if self.sampling_ratio==1:
                data=data_fitting
            if self.sampling_ratio>1:
                print("Please set the ratio to be 1 or less than 1.")
            if self.sampling_ratio<1:
                data=data_fitting.sample(frac=self.sampling_ratio)
        elif self.datastatus=="predict_dataset":
            data=data_pred

        # set u0 array and s0 array
        u0 = np.array(data.u0.copy().astype(np.float32))
        s0 = np.array(data.s0.copy().astype(np.float32))

        # below will be deprecated later since this is for the use of realdata
        u1 = np.float32(0)
        s1 = np.float32(0)
        alpha = np.float32(0)
        beta = np.float32(0)
        gamma = np.float32(0)
        type = "real"

        # add embedding (Guangyu)
        embedding1 = np.array(data.embedding1.copy().astype(np.float32))
        embedding2 = np.array(data.embedding2.copy().astype(np.float32))
        # print(embedding1)

        return u0, s0, u1, s1, alpha, beta, gamma, gene_name, type, u0max, s0max, embedding1, embedding2


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
        data = realDataset(data = adata_scv_pancreas(), k=k, gene_list=gene_list)
        for i in range(len(gene_list)):
            u0, s0, u1, s1, alpha, beta, gamma, gene_name, type1, u0max, s0max = data.__getitem__(i)
            print(u0, s0)
            plt.scatter(s0, u0, s=1)
            plt.title(gene_name)
            plt.show()

    gene_list=["Abcc8", "Rbfox3", "Sulf2", "Wfdc15b"]
    for k in [1]:
        data = realDataset(data = adata_scv_pancreas(), k=k, gene_list=gene_list)
        for i in range(len(gene_list)):
            u0, s0, u1, s1, alpha, beta, gamma, gene_name, type1, u0max, s0max = data.__getitem__(i)
            print(u0, s0)
            plt.scatter(s0, u0, s=1)
            plt.title(gene_name)
            plt.show()

def test_sampling():
    gene_list=["Rbfox3"]
    for pooling_method in ["binning", "x^2+y^2=1", "y=1/x"]:
        data = realDataset(data = adata_scv_pancreas(), k=100, pooling=True, pooling_method=pooling_method, gene_list=gene_list)
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

    #data= realDataset(data_fit = data_df_downsampled, data_predict=data_df,datastatus="fit_dataset", sampling_ratio=1)

    #u0, s0, u1, s1, alpha, beta, gamma, gene_name, type1, u0max, s0max = data.__getitem__(0)
    #plt.scatter(s0, u0, s=1)
    #plt.show()



    gene_list = ["Pdgfra", "Igfbpl1", "Ntrk2", "Syngr1"]
    #adata0 = load_adata(loom_file="../../data/loom/DentateGyrus.loom")
    #adata = init_adata(adata0)
    #adata.write_loom(filename="../../data/loom/DentateGyrus_pca.loom", write_obsm_varm=True)
    data = load_adata(loom_file="../../data/loom/DentateGyrus_pca.loom")
    data = realDataset(adata = data, k=500, gene_list=gene_list)
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
