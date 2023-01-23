import sys
sys.path.append('../src')
from utilities import set_rcParams
#from utilities import *
set_rcParams()
import warnings
import os
import argparse
warnings.filterwarnings("ignore")
from celldancer_plots import *
from sampling import *
import time
from velocity_estimation import *
from constant import *
import pandas as pd
import matplotlib.pyplot as plt
from colormap import *





model_dir = {"Sulf2": '/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Sulf2/Sulf2.pt', 
            "Ntrk2_e500": "/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Ntrk2_e500/Ntrk2_e500.pt"}
raw_path = '/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full.csv'
load_raw_data=pd.read_csv(raw_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])



gene_choice = ['Ntrk2']
# gene_choice = ['Dcx']

data_df=load_raw_data[['gene_list', 'u0','s0','cellID','embedding1','embedding2']][load_raw_data.gene_list.isin(gene_choice)]

embedding_downsampling, sampling_ixs, neighbor_ixs = downsampling_embedding(data_df,
                    para='neighbors',
                    target_amount=0,
                    step_i=200,
                    step_j=200,
                    n_neighbors=30)
gene_downsampling = downsampling(data_df=data_df, gene_choice=gene_choice, downsampling_ixs=sampling_ixs)
_, sampling_ixs_select_model, _ = downsampling_embedding(data_df,
                    para='neighbors',
                    target_amount=0,
                    step_i=20,
                    step_j=20,
                    n_neighbors=30)
gene_downsampling_select_model = downsampling(data_df=data_df, gene_choice= gene_choice, downsampling_ixs=sampling_ixs_select_model)
gene_shape_classify_dict=pd.DataFrame({'gene_name':gene_choice})
gene_shape_classify_dict['model_type']=gene_shape_classify_dict.apply (lambda row: select_initial_net(row.gene_name,gene_downsampling_select_model, data_df), axis=1)
gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Sulf2','model_type_dir']='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/bin/cellDancer-development_20220125/src/model/Sulf2/Sulf2.pt'
gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Ntrk2_e500','model_type_dir']='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/bin/cellDancer-development_20220125/src/model/Ntrk2_e500/Ntrk2_e500.pt'
feed_data = feedData(data_fit = gene_downsampling, data_predict=data_df, sampling_ratio=0.125) # default sampling_ratio=0.5
output_path = '/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/debug'


brief, detail = train(feed_data,
                        model_path=model_dir, 
                        max_epoches=100, 
                        model_save_path=None,
                        result_path=output_path,
                        n_jobs=8,
                        learning_rate=0.001,
                        cost_version=1,
                        cost2_cutoff=0.3,
                        n_neighbors=30,
                        cost1_ratio=0.5,
                        optimizer='Adam',
                        gene_shape_classify_dict=gene_shape_classify_dict,
                        with_trace_cost=True,
                        with_corrcoef_cost=True)

##################################### colorful



# color_map="coolwarm"
color_scatter=None
alpha_inside=1
pointsize=30
# pointsize=120
# color_scatter="#95D9EF" #blue
color_map=None
# alpha_inside=0.3

#color_scatter="#DAC9E7" #light purple
#color_scatter="#8D71B3" #deep purple
#alpha_inside=1
vmin=None
vmax=None
step_i=15
step_j=15

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

for i in gene_choice:
    one_gene_raw=load_raw_data[load_raw_data.gene_list==i]
    save_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/gene_velocity/celldancer/celldancer_colorful_Ntrk2_e1.pdf' # notice: changed
    velocity_plot([i],detail,color_scatter,pointsize,alpha_inside,color_map,vmin,vmax,save_path,step_i=step_i,step_j=step_j,custom_map=one_gene_raw['clusters'].map(colors)) # from cell dancer # from cell dancer
    





##################################### blue
# color_map="coolwarm"
color_scatter=None
alpha_inside=1
pointsize=30
# pointsize=120
color_scatter="#95D9EF" #blue
color_map=None
# alpha_inside=0.3

#color_scatter="#DAC9E7" #light purple
#color_scatter="#8D71B3" #deep purple
#alpha_inside=1
vmin=None
vmax=None
step_i=15
step_j=15

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

for i in gene_choice:
    one_gene_raw=load_raw_data[load_raw_data.gene_list==i]
    save_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/gene_velocity/celldancer/celldancer_blue_Ntrk2_e1.pdf"# notice: changed
    velocity_plot([i],detail,color_scatter,pointsize,alpha_inside,color_map,vmin,vmax,save_path,step_i=step_i,step_j=step_j) # from cell dancer # from cell dancer
    