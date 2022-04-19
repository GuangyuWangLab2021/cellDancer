from matplotlib.colors import Colormap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

load_detail_data=pd.DataFrame()
load_brief_data=pd.DataFrame()

# combine the detail and brief files of dengyr generated from hpc
for i in range(301,312):
    detail='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/detail_e'+str(i)+'.csv'
    detailfinfo='e'+str(300+i)
    detail_data = pd.read_csv (detail,index_col=False)
    load_detail_data=load_detail_data.append(detail_data)

    brief_data='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/brief_e'+str(i)+'.csv'
    brief_data = pd.read_csv (brief_data,index_col=False)
    load_brief_data=load_brief_data.append(brief_data)

##########################################
## generate cost csv file for each gene ##
##########################################
gene_cost=load_detail_data[['gene_name','cost']].drop_duplicates()
gene_cost.sort_values("cost")
gene_cost.to_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/gene_cost.csv',header=True,index=False)
gene_cost=pd.read_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/gene_cost.csv')
load_raw_data=pd.read_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full.csv',names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])
one_gene=load_raw_data[load_raw_data.gene_list==load_raw_data.gene_list.iloc[0]]

########cost 
add_amt=0.006
sns.distplot(gene_cost[(gene_cost.cost>0.05) & (gene_cost.cost<0.09)]['cost'],hist=False,color='black')
sns.distplot(gene_cost[(gene_cost.cost>0.075-add_amt) & (gene_cost.cost<0.075+add_amt)]['cost'],hist=False,color='black')

plt.title('cost')
plt.savefig('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/dist_gene_occupy_ratio.pdf')
####### choose gene by cost
gene_choice=gene_cost[(gene_cost.cost>0.075-add_amt) & (gene_cost.cost<0.075+add_amt)] # 512 genes
gene_choice=gene_choice.gene_name


load_detail_data_gene_choice=load_detail_data[load_detail_data.gene_name.isin(gene_choice)]

# add cluster info to load_detail_data
one_gene_cluster=one_gene.clusters
gene_amount=len(set(load_detail_data_gene_choice.gene_name))
cluster_all_gene=pd.concat([one_gene_cluster]*gene_amount)
load_detail_data_gene_choice=load_detail_data_gene_choice.reset_index()[['gene_name','s0','u0','s1','u1','alpha','beta','gamma','cost']]
load_detail_data_have_cluster_info=pd.concat([load_detail_data_gene_choice, cluster_all_gene.reset_index(),], axis=1)

para_list=['alpha','beta','gamma']
# cluster_choice=['nIPC']
cluster_choice=['RadialGlia','RadialGlia2']

load_detail_data_filtered_cluster=load_detail_data_have_cluster_info[load_detail_data_have_cluster_info.clusters.isin(cluster_choice)]

load_detail_data_filtered_cluster.to_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/512genes_nIPC_area.csv')

for para in para_list:
    para_gene_cell=load_detail_data_filtered_cluster.pivot(index='index', columns='gene_name', values=para).T
    #para_gene_cell.to_csv(os.path.join('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost',('512genes_nIPC_area_'+para+'.csv')))
    para_gene_cell.to_csv(os.path.join('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost',('512genes_RadialGlia_area_'+para+'.csv')))



# cluster the alpha file, then find the identical genes and output, then filter out them
para='alpha'
alpha_gene_cell=load_detail_data_filtered_cluster.pivot(index='index', columns='gene_name', values=para).T

genes_to_filter=pd.read_table('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/gene_to_filter_from_512genes_nIPC_area_alpha_heatmap_cluster_result.txt',header=None)
alpha_gene_cell_filtered=alpha_gene_cell[~(alpha_gene_cell.index.isin(genes_to_filter[0].tolist()))]
alpha_gene_cell_filtered.to_csv(os.path.join('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/gene_cost',('512genes_nIPC_area_alpha_filtered_similar_genes_left466.csv')))


######### filtered and reclustered - alpha
one_gene_embedding=one_gene[['embedding1','embedding2']]
kmeans_result=pd.read_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/512genes_nIPC_area_alpha_2kmeans.csv')
kmeans_result=pd.read_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/512genes_nIPC_area_alpha_filtered_similar_genes_left466_recluster.csv')
filtered_embedding=one_gene_embedding[one_gene_embedding.index.isin(kmeans_result.id)]
filtered_embedding['id']=filtered_embedding.index
merged_inner = pd.merge(left=filtered_embedding, right=kmeans_result, left_on='id', right_on='id')

layer=plt.scatter(merged_inner.embedding1,merged_inner.embedding2,c=merged_inner.k_means_2,cmap='coolwarm')
plt.colorbar(layer)
plt.savefig('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/512genes_nIPC_area_alpha_filtered_similar_genes_left466_recluster.pdf')
genes_to_filter=pd.read_table('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/gene_to_filter_from_512genes_nIPC_area_alpha_heatmap_cluster_result.txt',header=None)

######## load kmeans_result to sort
kmeans_result=pd.read_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/512genes_nIPC_area_alpha_filtered_similar_genes_left466_recluster.csv')
######## beta
beta_gene_cell=pd.read_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/512genes_nIPC_area_beta.csv',index_col='gene_name' )
beta_gene_cell_filtered_gene=beta_gene_cell[~(beta_gene_cell.index.isin(genes_to_filter[0].tolist()))]
filter_lis=list(map(str, kmeans_result['id'].tolist()))
beta_gene_cell_filtered_genes_and_resorted_by_cluster=beta_gene_cell_filtered_gene[filter_lis]
beta_gene_cell_filtered_genes_and_resorted_by_cluster.to_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/beta_gene_cell_filtered_genes_and_resorted_by_cluster.csv')
######## gamma
gamma_gene_cell=pd.read_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/512genes_nIPC_area_gamma.csv',index_col='gene_name' )
gamma_gene_cell_filtered_gene=gamma_gene_cell[~(gamma_gene_cell.index.isin(genes_to_filter[0].tolist()))]
filter_lis=list(map(str, kmeans_result['id'].tolist()))
gamma_gene_cell_filtered_genes_and_resorted_by_cluster=gamma_gene_cell_filtered_gene[filter_lis]
gamma_gene_cell_filtered_genes_and_resorted_by_cluster.to_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/alpha_beta_gamma_heatmap_filtered_by_gene_cost/gamma_gene_cell_filtered_genes_and_resorted_by_cluster.csv')




