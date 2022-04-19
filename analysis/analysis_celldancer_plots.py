# Plots for celldancer - heatmap, velocity, box, violin
# Read the results in detail file

from matplotlib.colors import Colormap
from matplotlib.pyplot import title
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import scvelo as scv
import numpy as np

def vaildation_plot(gene,validation_result,save_path_validation):
    plt.figure()
    plt.scatter(validation_result.epoch, validation_result.cost)
    plt.title(gene)
    plt.savefig(save_path_validation)
    
### backup of gene, celltype ####
# gene_plot_l_28_alphabet=['Abcc8', 'Actn4', 'Adk', 'Ank', 'Anxa4', 
#                     'Btbd17', 'Cdk1', 'Cpe', 'Dcdc2a', 'Gnao1', 
#                     'Gng12', 'Map1b', 'Mapre3', 'Nfib', 'Nnat', 
#                     'Pak3', 'Pcsk2', 'Pim2', 'Ppp3ca', 'Rap1b', 
#                     'Rbfox3', 'Smoc1', 'Sulf2', 'Tcp11', 'Tmem163', 
#                     'Top2a', 'Tspan7', 'Wfdc15b'] #28 alphabet order
# gene_plot_l_28=["Ank","Abcc8","Tcp11","Nfib","Ppp3ca",# 28 genes after did the re_clustering
#             "Rbfox3","Cdk1","Gng12","Map1b","Cpe",
#             "Gnao1","Pcsk2","Tmem163","Pak3","Wfdc15b",
#             "Nnat","Anxa4","Actn4","Btbd17","Dcdc2a",
#             "Adk","Smoc1","Mapre3","Pim2","Tspan7",
#             "Top2a","Rap1b","Sulf2"]
# gene_plot_l_3=["Wfdc15b","Abcc8","Sulf2"] # 3 picked sample genes
# adata = scv.datasets.pancreas()
# gene_plot_l_all=adata.var_names # TO DO: If we output the heatmap, there contains some NAN when we check the nalist in the heatmap function. This problem needs to be figured out later.
# cell_type_5=["Ductal",
#             "Ngn3 low EP",
#             "Ngn3 high EP",
#             "Pre-endocrine",
#             "Endocrine"]
# cell_type=["Ductal",
#             "Ngn3 low EP",
#             "Ngn3 high EP",
#             "Pre-endocrine",
#             "Alpha",
#             "Beta",
#             "Delta",
#             "Epsilon"]
# colorlist=['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']
# epoches=[0,5,10,50,100,200,210,225,250,275,300,400,500,1000]
# epoches=[0,5,10,50,100,500]
# epoches=[200]


#### Heatmap for alpha/ expression/ others ####


def heatmap(data,para,detail,cell_type,g_list):
    
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

# adata = scv.datasets.pancreas()
# for e_num in epoches:
#     file_path="output/detailcsv/adj_e/detail_e"+str(e_num)+".csv"
#     detail = pd.read_csv (file_path,index_col=False)
#     heat=heatmap(adata,para,detail,cell_type,gene_choice)
#     heat.to_csv(savepath_para_heat+str(e_num)+".csv",index=False)

# TO DO: Add the info of each cell
# epoches=[0,5,10,50,100,500]

# para="alpha_new"
# savepath_para_heat="output/heatmap/alpha_allcell/alpha_e"
# #para="s0" #=exp
# #savepath_para_heat="output/heatmap/exp_allcell/exp_e"
# cell_type=cell_type
# gene_choice=gene_plot_l_28


#### Velocity ####
# TO DO: modify the code to a diff style!!!!!!!!!!
# TO DO: build Class
# ref: https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sampling import sampling_neighbors
from colormap import *

def velocity_plot(gene,detail,para=None,colormap=None,color_scatter="#95D9EF",point_size=120,alpha_inside=0.3,v_min=None,v_max=None,save_path=None,step_i=15,step_j=15,show_arrow=True,custom_map=None):
    plt.figure(None,(6,6))
    u_s= np.array(detail[detail['gene_name']==gene][["u0","s0","u1","s1"]]) # u_s

    max_u_s=np.max(u_s, axis = 0)
    u0_max=max_u_s[0]
    s0_max=max_u_s[1]
    y_max=1.25*u0_max
    x_max=1.25*s0_max

    sampling_idx=sampling_neighbors(u_s[:,0:2], step_i=step_i,step_j=step_j) # Sampling
    u_s_downsample = u_s[sampling_idx,0:4]
    # layer1=plt.scatter(embedding[:, 1], embedding[:, 0],
    #             alpha=alpha_inside, s=point_size, edgecolor="none",c=detail[detail['gene_name'].isin(genelist)].alpha_new, cmap=colormap,vmin=v_min,vmax=v_max)
    #u_s= np.array(detail[detail['gene_name'].isin(gene)][["u0","s0","u1","s1"]])
    #u_s= np.array(detail[detail['gene_name'].isin(gene)][["u0","s0","u1","s1"]])

    plt.xlim(-0.05*s0_max, x_max) 
    plt.ylim(-0.05*u0_max, y_max) 

    title_info=gene
    if custom_map is not None:
        title_info=gene
        layer1=plt.scatter(u_s[:, 1], u_s[:, 0],
               alpha=alpha_inside, s=point_size, edgecolor="none",
               #c=pd.factorize(cluster_info)[0], 
               c=custom_map)
        #plt.colorbar(layer1)
    if (colormap is not None) and (para is not None):
        title_info=gene+' '+para
        layer1=plt.scatter(u_s[:, 1], u_s[:, 0],
               alpha=1, s=point_size, edgecolor="none",
               c=detail[detail['gene_name']==gene][para],cmap=colormap,vmin=v_min,vmax=v_max)
        plt.colorbar(layer1)
    elif color_scatter is not None:
        layer1=plt.scatter(u_s[:, 1], u_s[:, 0],
                alpha=alpha_inside, s=point_size, edgecolor="none",color=color_scatter,vmin=v_min,vmax=v_max)
    if show_arrow:
        plt.scatter(u_s_downsample[:, 1], u_s_downsample[:, 0], # sampled circle
                    color="none",s=point_size, edgecolor="k")
        pcm1 = plt.quiver(
        u_s_downsample[:, 1], u_s_downsample[:, 0], u_s_downsample[:, 3]-u_s_downsample[:, 1], u_s_downsample[:, 2]-u_s_downsample[:, 0],
        angles='xy', clim=(0., 1.))
    plt.title(title_info)
    if save_path is not None:
        plt.savefig(save_path)

# epoches=[10]
# epoches=[0,5,10,50,100,200,210,225,250,275,300,400,500,1000]

pointsize=120
pointsize=50
color_scatter="#95D9EF" #blue
alpha_inside=0.3

#color_scatter="#DAC9E7" #light purple
#color_scatter="#8D71B3" #deep purple
alpha_inside=0.2
gene_choice=['Ntrk2']

#for colormap in colorlist:
# for e_num in epoches:
#     file_path="output/detailcsv/adj_e/detail_e"+str(e_num)+".csv"
#     detail = pd.read_csv (file_path,index_col=False)
#     detail["alpha_new"]=detail["alpha"]/detail["beta"]
#     detail["beta_new"]=detail["beta"]/detail["beta"]
#     detail["gamma_new"]=detail["gamma"]/detail["beta"]
#     detailfinfo="e"+str(e_num)

#     #color_map="Spectral"
#     #color_map="PiYG"
#     #color_map="RdBu"
#     color_map='blue'
#     color_map=cmap1
#     # color_map="bwr"
#     alpha_inside=0.3
#     #alpha_inside=1
#     vmin=0
#     vmax=5
#     step_i=20
#     step_j=20
#     for i in gene_choice:
#         save_path="output/detailcsv/adj_e/"+i+"_"+"e"+str(e_num)+".pdf"# notice: changed
#         velocity_plot(detail, [i],detailfinfo,color_scatter,pointsize,alpha_inside,color_map,vmin,vmax,save_path,step_i,step_j) # from cell dancer


# epoches=[500]
# epoches=[0,5,10,50,100,200,210,225,250,275,300,400,500,1000]

# pointsize=120
# pointsize=50
# color_scatter="#95D9EF" #blue
# alpha_inside=0.3

# #color_scatter="#DAC9E7" #light purple
# color_scatter="#8D71B3" #deep purple
# alpha_inside=0.2
# gene_choice=gene_plot_l_28

# for e_num in epoches:
#     file_path="output/detailcsv/adj_e/detail_e"+str(e_num)+".csv"
#     detail = pd.read_csv (file_path,index_col=False)
#     detail["alpha_new"]=detail["alpha"]/detail["beta"]
#     detail["beta_new"]=detail["beta"]/detail["beta"]
#     detail["gamma_new"]=detail["gamma"]/detail["beta"]
#     detailfinfo="e"+str(e_num)

#     #color_map="Spectral"
#     #color_map="PiYG"
#     #color_map="RdBu"
#     color_map="coolwarm"
#     # color_map="bwr"
#     alpha_inside=0.3
#     alpha_inside=1
#     vmin=0
#     vmax=5
#     for i in gene_choice:
#         save_path="output/velo_plot_adj_e/"+i+"_"+"e"+str(e_num)+".pdf"# notice: changed
#         velocity_plot(detail, [i],detailfinfo,color_scatter,pointsize,alpha_inside,color_map,vmin,vmax,save_path) # from cell dancer



#### Box Plot for alpha_new, u0, s0 ####
# def box_attribute(gene,log,ylim,ymin,ymax,att,save_path,detail):
#     greyset=['grey']*8
#     detail["alpha_new"]=detail["alpha"]/detail["beta"]
#     detail["beta_new"]=detail["beta"]/detail["beta"]
#     detail["gamma_new"]=detail["gamma"]/detail["beta"]
#     g_list=set(detail["gene_name"])
#     celllist_rep=pd.concat([adata.obs.clusters]*len(g_list), ignore_index=True) # match cell type
#     detail = pd.concat([detail, celllist_rep], axis=1)

#     boxlist=detail[detail.gene_name==gene]
#     data=pd.DataFrame()
#     for ct in cell_type:
#         data=pd.concat([data,pd.DataFrame({'col':boxlist[boxlist.clusters==ct][att]})],axis=1,ignore_index=True)

#     plt.figure()
#     if ylim=="ylim":plt.ylim(ymin, ymax)
#     if log=="log":data=np.log10(data)
#     for i,d in enumerate(data):
#         y = data[d]
#         x = np.random.normal(i+1, 0.04, len(y))
#         ax = plt.plot(x, y, mfc = greyset[i], mec='k', ms=3, marker="o", linestyle="None",alpha=0.2,markeredgewidth=0.0)
#     data.boxplot(showfliers=False,grid=False,color=dict(boxes='black', whiskers='black', medians='black', caps='black')).set(title=gene)
#     plt.savefig(save_path)
#     return(data)

# epoches=[500]
# gene_plot_l=gene_plot_l_28
# att="u0"
# att="s0"
# att="alpha_new"
# cell_type=cell_type
# #log="log";ylim="ylim"
# ymin=-3;ymax=1
# log="no";ylim="no"

# for e_num in epoches:
#     for g in gene_plot_l:
#         for att in ["u0","s0","alpha_new"]:
#             save_path="output/dancer_plot_compare/box_"+att+"/velonn_"+g+".pdf"
#             file_path="output/detailcsv/adj_e/detail_e"+str(e_num)+".csv"
#             detail = pd.read_csv (file_path,index_col=False)
#             data_att=box_attribute(g,log,ylim,ymin,ymax,att,save_path,detail)
#             data_att.to_csv("output/dancer_plot_compare/box_"+att+"/data/velonn"+g+".csv",index=False)



#### Violin Plot for alpha_new, u0, s0 ####
# def violin_alpha(gene,log,ylim,ymin,ymax,save_path,detail):
#     detail["alpha_new"]=detail["alpha"]/detail["beta"]
#     detail["beta_new"]=detail["beta"]/detail["beta"]
#     detail["gamma_new"]=detail["gamma"]/detail["beta"]
#     g_list=set(detail["gene_name"])
#     celllist_rep=pd.concat([adata.obs.clusters]*len(g_list), ignore_index=True)
#     detail = pd.concat([detail, celllist_rep], axis=1)

#     #### violin plot
#     sns.set_style('white', {'axes.linewidth': 0.5})
#     plt.rcParams['xtick.bottom'] = True
#     plt.rcParams['ytick.left'] = True
#     sns.set_context("talk", font_scale=1.1)
#     plt.figure(figsize=(4.5,3))
#     plt.figure(figsize=(6,4))

#     if ylim=="ylim":plt.ylim(ymin, ymax)#plt.ylim(-0.6, 2)
#     v_data=detail[detail.gene_name==gene]
#     np.random.seed(10) # set random seed
#     if log=="log":v_data.alpha_new=np.log10(v_data.alpha_new)
#     sns.stripplot(y="alpha_new", 
#                 x="clusters", 
#                 data=v_data,
#             color="black", edgecolor="gray",s=1,alpha=0.5,order=cell_type)
    
#     sns.violinplot(y="alpha_new", 
#                 x="clusters", 
#                 data=v_data,order=cell_type,color="white", width=1 ,inewidth=0.1, inner=None)
#     plt.savefig(save_path)
#     return(v_data)

# e_num=[500]
# gene_plot_l=gene_plot_l_28
# log="log";ylim="ylim"
# ymin=-4;ymax=1
# #log="no";ylim="no"
# for e_num in epoches:
#     file_path="output/detailcsv/adj_e/detail_e"+str(e_num)+".csv"
#     detail = pd.read_csv (file_path,index_col=False)
#     for g in gene_plot_l:
#         save_path="output/alpha_plot/violin/velonn_"+g+".pdf"
#         data_violin=violin_alpha(g,log,ylim,ymin,ymax,save_path,detail)
#         data_violin.to_csv("output/alpha_plot/violin/velonn_"+g+".csv",index=False)

