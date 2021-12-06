# Plots for celldancer

from matplotlib.colors import Colormap
from matplotlib.pyplot import title
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

# backup of gene, celltype
gene_plot_l_28_alphabet=['Abcc8', 'Actn4', 'Adk', 'Ank', 'Anxa4', 
                    'Btbd17', 'Cdk1', 'Cpe', 'Dcdc2a', 'Gnao1', 
                    'Gng12', 'Map1b', 'Mapre3', 'Nfib', 'Nnat', 
                    'Pak3', 'Pcsk2', 'Pim2', 'Ppp3ca', 'Rap1b', 
                    'Rbfox3', 'Smoc1', 'Sulf2', 'Tcp11', 'Tmem163', 
                    'Top2a', 'Tspan7', 'Wfdc15b'] #28 alphabet order
gene_plot_l_28=["Ank","Abcc8","Tcp11","Nfib","Ppp3ca",# 28 genes after did the re_clustering
            "Rbfox3","Cdk1","Gng12","Map1b","Cpe",
            "Gnao1","Pcsk2","Tmem163","Pak3","Wfdc15b",
            "Nnat","Anxa4","Actn4","Btbd17","Dcdc2a",
            "Adk","Smoc1","Mapre3","Pim2","Tspan7",
            "Top2a","Rap1b","Sulf2"]
gene_plot_l_3=["Wfdc15b","Abcc8","Sulf2"] # 3 picked sample genes
gene_plot_l_all=adata.var_names # TO DO: If we output the heatmap, there contains some NAN when we check the nalist in the heatmap function. This problem needs to be figured out later.
cell_type=["Ductal",
            "Ngn3 low EP",
            "Ngn3 high EP",
            "Pre-endocrine",
            "Alpha",
            "Beta",
            "Delta",
            "Epsilon"]
list_e=[0,5,10,50,100,200,210,225,250,275,300,400,500,1000]



# Heatmap for alpha/ expression/ others
# TO DO: Add the info of each cell
list_e=[500]
para="alpha_new"
savepath_para_heat="output/heatmap/alpha_allcell/alpha_e"
para="s0" #=exp
savepath_para_heat="output/heatmap/exp_allcell/exp_e"
cell_type=cell_type
g_list=gene_plot_l_28

def heatmap(data,para,file_path,cell_type,g_list):
    detail = pd.read_csv (file_path,index_col=False)
    detail["alpha_new"]=detail["alpha"]/detail["beta"]
    detail["beta_new"]=detail["beta"]/detail["beta"]
    detail["gamma_new"]=detail["gamma"]/detail["beta"]
    celllist_rep=pd.concat([data.obs.clusters]*len(g_list), ignore_index=True)
    detail = pd.concat([detail, celllist_rep], axis=1)
    detail_para=detail.loc[:,["gene_name",para,"clusters"]]
    detail_para_sorted = detail_para.sort_values(["clusters", "gene_name"], ascending = (False, True))

    fin_matrix=pd.DataFrame()
    for ct in cell_type:
        df = list()
        for g in g_list:
            df_v3=detail_para_sorted[(detail_para_sorted.clusters==ct) & (detail_para_sorted.gene_name==g)]
            df.append(df_v3[para])
        reshaped_array = np.reshape(df, (len(g_list),len(df_v3)))
        reshaped_array_pd = pd.DataFrame(reshaped_array)
        fin_matrix=pd.concat([fin_matrix,reshaped_array_pd],axis=1)
    fin_matrix=pd.concat([pd.DataFrame({'col':g_list}),fin_matrix],axis=1)
    return(fin_matrix)

# TO DO:
for e_num in list_e:
    file_path="output/detailcsv/adj_e/detail_e"+str(e_num)+".csv"
    heatmap500=heatmap(adata,para,file_path,cell_type,g_list)
    #heatmap.to_csv(savepath_para_heat+str(e_num)+".csv",index=False)



# Velocity
# TO DO: modify the code to a diff style!!!!!!!!!!
# TO DO: build Class
# ref: https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
import numpy as np
from sklearn.neighbors import NearestNeighbors
def sampling_neighbors(embedding,step_i=20,step_j=20):
    def gaussian_kernel(X, mu = 0, sigma=1):
        return np.exp(-(X - mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    steps = step_i, step_j
    grs = []
    for dim_i in range(embedding.shape[1]-3):
        m, M = np.min(embedding[:, dim_i]), np.max(embedding[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, steps[dim_i])
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
    gridpoints_coordinates = gridpoints_coordinates + norm.rvs(loc=0, scale=0.15, size=gridpoints_coordinates.shape)
    
    np.random.seed(10) # set random seed
    nn = NearestNeighbors()
    nn.fit(embedding[:,0:2])
    dist, ixs = nn.kneighbors(gridpoints_coordinates, 20)
    ix_choice = ixs[:,0].flat[:]
    ix_choice = np.unique(ix_choice)

    nn = NearestNeighbors()
    nn.fit(embedding[:,0:2])
    dist, ixs = nn.kneighbors(embedding[ix_choice, 0:2], 20)
    density_extimate = gaussian_kernel(dist, mu=0, sigma=0.5).sum(1)
    bool_density = density_extimate > np.percentile(density_extimate, 25)
    ix_choice = ix_choice[bool_density]
    return(embedding[ix_choice,0:4])

def velocity_plot(detail,genelist,detail_i,color_scatter,point_size,alpha_inside,colormap,v_min,v_max):
    from scipy.stats import norm
    plt.figure(None,(6,6))
    embedding= np.array(detail[detail['gene_name'].isin(genelist)][["u0","s0","u1","s1", 'true_cost']])


    sampled_coordinates=sampling_neighbors(embedding) # Sampling

    layer1=plt.scatter(embedding[:, 1], embedding[:, 0],
                alpha=alpha_inside, s=point_size, edgecolor="none",c=detail[detail['gene_name'].isin(genelist)].alpha_new, cmap=colormap,vmin=v_min,vmax=v_max)
    plt.colorbar(layer1)
    plt.scatter(sampled_coordinates[:, 1], sampled_coordinates[:, 0],
                color="none",s=point_size, edgecolor="k")
    
    pcm1 = plt.quiver(
    sampled_coordinates[:, 1], sampled_coordinates[:, 0], sampled_coordinates[:, 3]-sampled_coordinates[:, 1], sampled_coordinates[:, 2]-sampled_coordinates[:, 0],
    angles='xy', clim=(0., 1.))
    #plt.savefig("output/velo_plot_adj_e/"+genelist[0]+"_"+detail_i+".pdf")


list_e=[500]
pointsize=120
pointsize=50
color_scatter="#95D9EF" #blue
alpha_inside=0.3

#color_scatter="#DAC9E7" #light purple
color_scatter="#8D71B3" #deep purple
alpha_inside=0.2
g_list=gene_plot_l_3

for e_num in list_e:
    file_path="output/detailcsv/adj_e/detail_e"+str(e_num)+".csv"
    detail = pd.read_csv (file_path,index_col=False)
    detail["alpha_new"]=detail["alpha"]/detail["beta"]
    detail["beta_new"]=detail["beta"]/detail["beta"]
    detail["gamma_new"]=detail["gamma"]/detail["beta"]
    detailfinfo="e"+str(e_num)

    #color_map="Spectral"
    #color_map="PiYG"
    #color_map="RdBu"
    color_map="coolwarm"
    # color_map="bwr"
    alpha_inside=0.3
    alpha_inside=1
    vmin=0
    vmax=1
    for i in g_list:
        velocity_plot(detail, [i],detailfinfo,color_scatter,pointsize,alpha_inside,color_map,vmin,vmax) # from cell dancer
        #plt_alp(detail,[i],detailfinfo)
    print("color map:" +color_map)

