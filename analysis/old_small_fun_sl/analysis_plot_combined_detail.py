import pandas as pd
from celldancer_plots import *
from colormap import *
def vaildation_plot(gene,validation_result,save_path_validation):
    plt.figure()
    plt.scatter(validation_result.epoch, validation_result.cost)
    plt.title(gene)
    plt.savefig(save_path_validation)

n_gene=10
raw_data_path="/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full.csv"
load_raw_data=pd.read_csv(raw_data_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])


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



# set paras for velocity plot
#color_map="coolwarm"
color_map=None
color_scatter=None
alpha_inside=1
pointsize=30
# pointsize=120
# color_scatter="#95D9EF" #blue
# color_map=None
# alpha_inside=0.3

#color_scatter="#DAC9E7" #light purple
#color_scatter="#8D71B3" #deep purple
#alpha_inside=1
vmin=None
vmax=None
step_i=15
step_j=15
gene_choice=['Rimbp2','Elavl4','Dctn3','Psd3','Dcx','Ntrk2']
gene_choice=['Pfkp']

gene_choice=['Robo2']
gene_choice=['Tox',
            'Fam19a1',
            'Pcdh7',
            'Gabrg3',
            'Nell1',
            'Gpc6',
            'Brinp1',
            'Gabrb1',
            'Fat3',
            'Nav1',
            'Cdh13',
            'Arhgef3',
            'Ppfia2',
            'Kcnq3',
            'Syn2',
            'Dscam',
            'Klf7',
            'Slc4a10',
            'Pafah1b1',
            'Nalcn',
            'Gnao1',
            'Kcnip1',
            'Ncald',
            'Syne1',
            'Raver2']
gene_choice=['Bex4','Mt1','Mt2','Dynll1','Smim18','Slc25a4','Sepw1','Calm1','Celf5','Tmsb10']
gene_choice=['Gstm5','Smim18'] #low performance genes
gene_choice=['Ablim1']
colors = {'CA':grove2[6],
'CA1-Sub':grove2[8],
'CA2-3-4':grove2[7],
'Granule':grove2[5],
'ImmGranule1':grove2[5],
'ImmGranule2':grove2[5],
'Nbl1':grove2[4],
'Nbl2':grove2[4],
'nIPC':grove2[3],
'RadialGlia':grove2[2],
'RadialGlia2':grove2[2],
'GlialProg' :grove2[2],
'OPC':grove2[1],
'ImmAstro':grove2[0]}

from velocity_plot import velocity_plot as vpl
# one_gene_raw=load_raw_data[load_raw_data.gene_list=='Rgs20']
save_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/gene_velocity/celldancer/low_performance/'
for gene in gene_choice:
    # color_scatter=None
    # velocity_plot([gene],load_detail_data,color_scatter,pointsize,alpha_inside,color_map,vmin,vmax,(save_path+'celldancer_'+gene+'_colorful.pdf'),step_i,step_j,show_arrow=True,custom_map=one_gene_raw['clusters'].map(colors))
    color_scatter="#95D9EF"
    vpl.velocity_gene([gene],load_detail_data,color_scatter,pointsize,alpha_inside,color_map,vmin,vmax,(save_path+'celldancer_'+gene+'_blue.pdf'),step_i,step_j,show_arrow=False)

####################################
import pandas as pd
from velocity_plot import velocity_plot as vpl

raw_data_path="/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full.csv"
load_raw_data=pd.read_csv(raw_data_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])
load_detail_data=pd.read_csv ('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/detail_e301.csv',index_col=False)

one_gene_raw=load_raw_data[load_raw_data.gene_list==load_raw_data.gene_list.iloc[0]]
cluster_info=one_gene_raw['clusters']
vpl.velocity_gene('Ablim1',load_detail_data,save_path=None)
vpl.velocity_gene('Ablim1',load_detail_data,save_path=None,cluster_info=cluster_info,mode='cluster')
vpl.velocity_gene('Ablim1',load_detail_data,save_path=None,cluster_info=cluster_info,mode='cluster',cluster_annot=True)

