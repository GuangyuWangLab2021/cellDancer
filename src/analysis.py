# seems only the plot of cost is useful for current version (brief)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def brief_with_different_neighbours(): #cost function plot
    brief = pd.read_csv("/Users/lingqunye/Desktop/onedrive/veloNN/result/result_nopre_10/brief.csv")
    brief_summary = brief.groupby(["gene_name", "type", "epoch"], as_index=False).min()
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = False)
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)

    brief = pd.read_csv("/Users/lingqunye/Desktop/onedrive/veloNN/result/result_nopre_20/brief.csv")
    brief_summary = brief.groupby(["gene_name", "type", "epoch"], as_index=False).min()
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = False)
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)

    brief = pd.read_csv("/Users/lingqunye/Desktop/onedrive/veloNN/result/result_nopre_30/brief.csv")
    brief_summary = brief.groupby(["gene_name", "type", "epoch"], as_index=False).min()
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = False)
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)


    brief = pd.read_csv("/Users/lingqunye/Desktop/onedrive/veloNN/result/result_pre_10/brief.csv")
    brief_summary = brief.groupby(["gene_name", "type", "epoch"], as_index=False).min()
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = False)
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)

    brief = pd.read_csv("/Users/lingqunye/Desktop/onedrive/veloNN/result/result_pre_20/brief.csv")
    brief_summary = brief.groupby(["gene_name", "type", "epoch"], as_index=False).min()
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = False)
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)
  
def show_details_with_model_info( #by lq: not use since we have a new version
    detail,
    model=None,
    gene_name =None,
    type=None,
    true_cost=False,
    cols = 4):
    '''these auguments are lists'''
    if model != None:
        detail = detail[detail["model"].isin(model)]
    if gene_name != None:
        detail = detail[detail["gene_name"].isin(gene_name)]
    if type != None:
        detail = detail[detail["type"].isin(type)]

    detail.index = range(len(detail))
    def trim_axs(axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]
    num = detail.groupby(['model', 'gene_name', 'type']).ngroups
    rows = num//cols
    if num%cols != 0:
        rows = rows + 1
    axs = plt.figure(figsize=(cols*2,rows*2), constrained_layout=True).subplots(rows, cols)
    axs = trim_axs(axs, num)
    #i=0
    for ax, (ids, subdf) in zip(axs, detail.groupby(['model', 'gene_name', 'type'])):
        if true_cost:
            pcm1 = ax.quiver(
                subdf['s0'], subdf['u0'], subdf['s1']-subdf['s0'], subdf['u1']-subdf['u0'], subdf['true_cost']**(1/10), 
                angles='xy', cmap=plt.cm.jet, clim=(0., 1.))
        else:
            pcm1 = ax.quiver(
                subdf['s0'], subdf['u0'], subdf['s1']-subdf['s0'], subdf['u1']-subdf['u0'], subdf['cost']**(1/10), 
                angles='xy', cmap=plt.cm.jet, clim=(0., 1.))
        ax.set_title(ids[1])
        #plt.colorbar(pcm1, cmap=plt.cm.jet, ax=ax)
    plt.show()

# Show details with quiver figure
def show_details(
    detail,
    gene_name =None,
    type=None,
    true_cost=False,
    cols = 4):
    '''these auguments are lists'''
    if gene_name != None:
        detail = detail[detail["gene_name"].isin(gene_name)]
    if type != None:
        detail = detail[detail["type"].isin(type)]

    detail.index = range(len(detail))
    def trim_axs(axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]
    num = detail.groupby(['gene_name', 'type']).ngroups
    rows = num//cols
    if num%cols != 0:
        rows = rows + 1

    if num <= 12:
        colsize = cols*4
        rowsize = rows*4
    else:
        colsize = cols*2
        rowsize = rows*2  

    axs = plt.figure(figsize=(colsize,rowsize), constrained_layout=True).subplots(rows, cols)
    axs = trim_axs(axs, num)
    #i=0
    for ax, (ids, subdf) in zip(axs, detail.groupby(['gene_name', 'type'])):
        if true_cost:
            pcm1 = ax.quiver(
                subdf['s0'], subdf['u0'], subdf['s1']-subdf['s0'], subdf['u1']-subdf['u0'], subdf['true_cost']**(1/10), 
                angles='xy', cmap=plt.cm.jet, clim=(0., 1.))
        else:
            pcm1 = ax.quiver(
                subdf['s0'], subdf['u0'], subdf['s1']-subdf['s0'], subdf['u1']-subdf['u0'], subdf['cost']**(1/10), 
                angles='xy', cmap=plt.cm.jet, clim=(0., 1.)) #angles must be xy
        ax.set_title(ids[0])
        #plt.colorbar(pcm1, cmap=plt.cm.jet, ax=ax)
    plt.show()

# Show details with scatter figure (no arrow) #by lq: not use since we have a new version
def show_details2(
    detail,
    gene_name =None,
    type=None,
    cols = 4):
    '''these auguments are lists'''
    if gene_name != None:
        detail = detail[detail["gene_name"].isin(gene_name)]
    if type != None:
        detail = detail[detail["type"].isin(type)]

    detail.index = range(len(detail))
    def trim_axs(axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]
    num = detail.groupby(['gene_name', 'type']).ngroups
    rows = num//cols
    if num%cols != 0:
        rows = rows + 1

    if num <= 4:
        colsize = cols*4
        rowsize = rows*4
    else:
        colsize = cols*2
        rowsize = rows*2  
    axs = plt.figure(figsize=(colsize,rowsize), constrained_layout=True).subplots(rows, cols)
    axs = trim_axs(axs, num)
    #i=0
    #color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for ax, (ids, subdf) in zip(axs, detail.groupby(['gene_name', 'type'])):
        pcm1 = ax.scatter(subdf['s0'], subdf['u0'], s=1, c=subdf['alpha'])
        ax.set_title(ids[0] +'\n' +r'$\beta$='+str(round(np.mean(subdf['beta']),2))+r', $\gamma$='+str(round(np.mean(subdf['gamma']),2)))
        #pcm1.set_clim(1, 3)
        plt.colorbar(pcm1, cmap=plt.cm.jet, ax=ax)
    plt.show()

# Show details with quive & scatter. Figure 1B #by lq: not use since we have a new version
def show_details_simplify(
    detail,
    gene_name =None,
    title = "",
    scale=0.003,
    color='darkorange',
    seed = 0
    ):
    if gene_name != None:
        detail = detail[detail["gene_name"].isin(gene_name)]

    plt.figure(None,(6,6))
    s0, u0, s1, u1 = detail['s0'], detail['u0'], detail['s1'], detail['u1']
    plt.scatter(s0, u0, c=color, alpha=0.2, s=60, edgecolor="")

    import scipy.stats
    values = np.vstack([u0,s0])
    kernel = scipy.stats.gaussian_kde(values)
    p = kernel(values)
    idx = np.arange(values.shape[1])
    r = scipy.stats.rv_discrete(values=(idx, (1/p)/sum(1/p)), seed=seed)
    pp = r.rvs(size=50)
    u0_sub = values[0, pp]
    s0_sub = values[1, pp]
    values2 = np.vstack([u1,s1])
    u1_sub = values2[0, pp]
    s1_sub = values2[1, pp]

    # from sklearn.neighbors import NearestNeighbors
    # nn = NearestNeighbors()
    # values = np.vstack([u0,s0])
    # nn.fit(values)
    # dist, ixs = nn.kneighbors(gridpoints_coordinates, 20)
    # ix_choice = ixs[:,0].flat[:]
    # ix_choice = np.unique(ix_choice)
    # nn = NearestNeighbors()
    # nn.fit(vlm.embedding)
    # dist, ixs = nn.kneighbors(vlm.embedding[ix_choice], 20)
    # density_extimate = gaussian_kernel(dist, mu=0, sigma=0.5).sum(1)
    # bool_density = density_extimate > np.percentile(density_extimate, 25)
    # ix_choice = ix_choice[bool_density]

    # import scipy.stats
    # values = np.vstack([u0,s0])
    # kernel = scipy.stats.gaussian_kde(values)
    # p = kernel(values)
    # idx = np.arange(values.shape[1])
    # tmp_p = np.square((1-(p/(max(p)))**2))+0.0001
    # # tmp_p = np.square((1-(((p+0.4*max(p))*4-2*max(p+0.4*max(p)))/(2*max(p+0.4*max(p))))**2))+0.0001
    # p2 = tmp_p/sum(tmp_p)
    # r = scipy.stats.rv_discrete(values=(idx, p2), seed=0)
    # pp = r.rvs(size=50)
    # u0_sub = values[0, pp]
    # s0_sub = values[1, pp]
    # values2 = np.vstack([u1,s1])
    # u1_sub = values2[0, pp]
    # s1_sub = values2[1, pp]

    plt.scatter(s0_sub, u0_sub, c=color, alpha=1, s=60, edgecolor="k")

    quiver_kwargs=dict(scale=scale, headaxislength=2, headlength=4, headwidth=4,linewidths=0.1, edgecolors="k", color="k", alpha=1)

    plt.quiver(s0_sub, u0_sub, s1_sub-s0_sub, u1_sub-u0_sub, angles='xy', **quiver_kwargs)
    plt.title(title)


if __name__ == "__main__":
    from utilities import set_rcParams
    set_rcParams()
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.append('.')
    from simulation_pl3 import *

    '''
    brief = pd.read_csv("/Users/lingqunye/Desktop/onedrive/veloNN/result/result_without_pretrain/all_brief.csv")
    brief_summary = brief.groupby(["gene_name", "type", "epoch"], as_index=False).min()
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = False)
    sns.relplot(data=brief_summary, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)


    brief2 = pd.read_csv("/Users/lingqunye/Desktop/onedrive/veloNN/result/result_with_pretrain/all_brief.csv")
    brief_summary2 = brief2.groupby(["gene_name", "type", "epoch"], as_index=False).min()
    sns.relplot(data=brief_summary2, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = False)
    sns.relplot(data=brief_summary2, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)

    detail = pd.read_csv("/Users/lingqunye/Desktop/onedrive/veloNN/result/result_pre_20/detail.csv")
    show_details(detail, model=['early_switch_300.pt', 'degradation_unfinished_300.pt' ], gene_name = ["nm008"], cols=2)
    '''

    '''
    brief = pd.read_csv("../../result/result_pre_g80_e500/brief.csv")
    brief_min = brief.loc[brief.groupby(["gene_name", "type", "epoch"]).cost.idxmin()]
    sns.relplot(data=brief_min, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)
    brief_min.to_csv('../../result/result_pre_g80_e500/brief_min.csv')

    detail = pd.read_csv("../../result/result_pre_g80_e500/detail.csv")
    brief_min_tuple = list(brief_min[brief_min['epoch']==499][['model', 'gene_name']].apply(tuple, axis=1))
    detail_min = detail[detail[['model', 'gene_name']].apply(tuple, axis=1).isin(brief_min_tuple)]
    detail_min_sort = detail_min.sort_values('gene_name')
    detail_min_sort.to_csv("../../result/result_pre_g80_e500/detail_min.csv")
    show_details(detail_min, cols=10)
    show_details(detail_min_sort, true_cost=True, cols=10)
    show_details(detail_min_sort, gene_name=['mp002', 'es002'], true_cost=True, cols=2)
    show_details2(detail_min_sort, cols=10)
    show_details2(detail_min_sort, gene_name=['mp002', 'es002'], cols=2)

    show_details2(detail_min_sort, gene_name=['ac005', 'bw000', 'dc001', 'du000'], cols=2)
    show_details2(detail_min_sort, gene_name=['es000', 'fw000', 'mp001', 'nm004'], cols=2)

    show_details2(detail_min_sort, gene_name=['ac005'], cols=2)
    show_details2(detail_min_sort, gene_name=['mp001'], cols=2)
    '''

    brief = pd.read_csv("../../result/result_pl4/brief.csv")
    sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)
    brief_min = brief.loc[brief.groupby(["gene_name", "type", "epoch"]).cost.idxmin()]
    sns.relplot(data=brief_min, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)

    #brief_min.to_csv('../../result/result_pl4/brief_min.csv')

    detail = pd.read_csv("../../result/result_pl4/detail.csv")
    brief_min_tuple = list(brief_min[brief_min['epoch']==9][['model', 'gene_name']].apply(tuple, axis=1))
    detail_min = detail[detail[['model', 'gene_name']].apply(tuple, axis=1).isin(brief_min_tuple)]
    detail_min_sort = detail_min.sort_values('gene_name')
    #detail_min_sort.to_csv("../../result/result_pl4/detail_min.csv")
    show_details(detail_min, cols=2)
    #show_details2(detail_min_sort, cols=10)
    #show_details(detail_min_sort, gene_name=['es000', 'fw000', 'mp001', 'nm004'], cols=2)
    #show_details(detail_min_sort, gene_name=['es000', 'fw000', 'mp001', 'nm004'], true_cost=True, cols=2)
    #show_details2(detail_min_sort, gene_name=['es000', 'fw000', 'mp001', 'nm004'], cols=2)


    detail = pd.read_csv("../../result/result_pl4/detail.csv")
    show_details(detail, cols=4)
    show_details(detail, cols=4, true_cost=True)
