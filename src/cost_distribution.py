import pandas as pd
from celldancer_plots import *
def vaildation_plot(gene,validation_result,save_path_validation):
    plt.figure()
    plt.scatter(validation_result.epoch, validation_result.cost)
    plt.title(gene)
    plt.savefig(save_path_validation)

n_gene=10
raw_data_path="/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/veloNN-main_github/src/velonn/data/denGyr_full.csv"
load_raw_data=pd.read_csv(raw_data_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])

load_detail_data=pd.DataFrame()
load_brief_data=pd.DataFrame()

# combine the detail and brief files of dengyr generated from hpc
for i in range(301,312):
    detail='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/20220117_adjusted_gene_choice_order/denGyrLr0.001Costv3C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/detail_e'+str(i)+'.csv'
    detailfinfo='e'+str(300+i)
    detail_data = pd.read_csv (detail,index_col=False)
    load_detail_data=load_detail_data.append(detail_data)

    brief_data='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/20220113_adjusted_identify_shape/denGyrLr0.001Costv3C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/brief_e'+str(i)+'.csv'
    brief_data = pd.read_csv (brief_data,index_col=False)
    load_brief_data=load_brief_data.append(brief_data)

load_detail_data
load_brief_data
load_detail_data["alpha_new"]=load_detail_data["alpha"]

##########################################
## generate cost csv file for each gene ##
##########################################
gene_cost=load_detail_data[['gene_name','cost']].drop_duplicates()
gene_cost.sort_values("cost")
gene_cost.to_csv('output/gene_cost/denGyr/gene_cost.csv',header=True,index=False)

############################
## Dendisty plot for cost ##
############################
gene_cost=pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/gene_cost/denGyr/gene_cost.csv')

import seaborn as sns
sns.distplot(gene_cost[gene_cost.cost<0.1]['cost'],hist=False,color='black')
plt.title('cost')
plt.savefig('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/gene_occupy_ratio/denGyr/dist_gene_occupy_ratio.pdf')

#########################################
## gene velocity plot filtered by cost ##
#########################################
pointsize=120
pointsize=50
color_scatter="#95D9EF" #blue
color_map="coolwarm"
alpha_inside=1
vmin=0
vmax=5
step_i=20
step_j=20
detailfinfo='e301-e311'
output_path='output/costplot'

gene_choice=gene_cost.gene_name[0:n_gene]
gene_choice=gene_cost.gene_name[160:(n_gene+160)]
gene_choice=gene_cost.gene_name[2138:(n_gene+2138)]
gene_choice=gene_cost.gene_name[100:(n_gene+110)]
for i in gene_choice:
    cost_this_gene=gene_cost[gene_cost.gene_name==i]['cost'].iloc[0]
    corelation=np.corrcoef(load_detail_data[load_detail_data.gene_name==i].u0,load_detail_data[load_detail_data.gene_name==i].s0)[0,1]
    save_path=output_path+'/lowcost/'+i+'+'+str(cost_this_gene)[0:5]+'_'+str(corelation)[0:5]+'.pdf'
    velocity_plot(load_detail_data, [i],detailfinfo,color_scatter,pointsize,alpha_inside,color_map,vmin,vmax,save_path,step_i=step_i,step_j=step_j)
    
    # save_path_validation=output_path+'/lowcost/validation_'+i+'+'+str(cost_this_gene)[0:5]+'.pdf'
    # vaildation_plot(gene=i,validation_result=brief[brief["gene_name"]==i],save_path_validation=save_path_validation)

