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


time_start=time.time()

print('\nvelocity_estimate.py version 1.0.0')
print('python velocity_estimate.py')
print('time_start'+str(time_start))
print('')

use_all_gene=False
plot_trigger=True

model_dir = {"Sulf2": '/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Sulf2/Sulf2.pt', 
            "Ntrk2_e500": "/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Ntrk2_e500/Ntrk2_e500.pt"}
config = pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/config/config_test.txt', sep=';',header=None)
data_source = config.iloc[0][0]
platform = config.iloc[0][1]
epoches=[int(config.iloc[0][2])]
num_jobs=int(config.iloc[0][3])
learning_rate=float(config.iloc[0][4])
cost_version=int(config.iloc[0][5])
cost1_ratio=float(config.iloc[0][6])
cost2_cutoff=float(config.iloc[0][7])
downsample_method=config.iloc[0][8]
downsample_target_amount=int(config.iloc[0][9])
step_i=int(config.iloc[0][10])
step_j=int(config.iloc[0][11])
sampling_ratio=float(config.iloc[0][12])
n_neighbors=int(config.iloc[0][13])
optimizer=config.iloc[0][14] #["SGD","Adam"]  # set to Adam
raw_path=config.iloc[0][18]
out_path=config.iloc[0][19]



#### mkdir for output_path with parameters(naming)
folder_name=(data_source+
    "Lr"+str(learning_rate)+
    "Costv"+str(cost_version)+
    "C1r"+str(cost1_ratio)+
    "C2cf"+str(cost2_cutoff)+
    "Down"+downsample_method+str(downsample_target_amount)+"_"+str(step_i)+"_"+str(step_j)+
    "Ratio"+str(sampling_ratio)+
    "N"+str(n_neighbors)+
    "O"+optimizer)

output_path=(out_path+folder_name+"/")
if os.path.isdir(output_path):pass
else:os.mkdir(output_path)


######################################################
############             Guangyu          ############
######################################################
#raw_data_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/data/simulation_data/one_gene.csv'

load_raw_data=pd.read_csv(raw_path)

if use_all_gene: 
    gene_choice=list(set(load_raw_data.gene_list))
    gene_choice.sort()
    gene_choice=gene_choice[full_start: full_end]
    print('---gene_choice---')
    print(gene_choice)
else:
    gene_choice = select_gene_set(data_source)

data_df=load_raw_data[['gene_list', 'u0','s0','embedding1','embedding2']][load_raw_data.gene_list.isin(gene_choice)]

#!!!!!!!!!!! data_df=load_raw_data[['gene_list', 'u0','s0','cellID','embedding1','embedding2']][load_raw_data.gene_list.isin(gene_choice)]
embedding_downsampling, sampling_ixs, neighbor_ixs = downsampling_embedding(data_df,
                    para=downsample_method,
                    target_amount=downsample_target_amount,
                    step_i=step_i,
                    step_j=step_j,
                    n_neighbors=n_neighbors)
gene_downsampling = downsampling(data_df=data_df, gene_choice=gene_choice, downsampling_ixs=sampling_ixs)

_, sampling_ixs_select_model, _ = downsampling_embedding(data_df,
                    para=downsample_method,
                    target_amount=downsample_target_amount,
                    step_i=20,
                    step_j=20,
                    n_neighbors=n_neighbors)
gene_downsampling_select_model = downsampling(data_df=data_df, gene_choice=gene_choice, downsampling_ixs=sampling_ixs_select_model)


gene_shape_classify_dict=pd.DataFrame({'gene_name':gene_choice})
gene_shape_classify_dict['model_type']=gene_shape_classify_dict.apply (lambda row: select_initial_net(row.gene_name,gene_downsampling_select_model, data_df), axis=1)

gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Sulf2','model_type_dir']='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/bin/cellDancer-development_20220125/src/model/Sulf2/Sulf2.pt'
gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Ntrk2_e500','model_type_dir']='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/bin/cellDancer-development_20220125/src/model/Ntrk2_e500/Ntrk2_e500.pt'

# set fitting data, data to be predicted, and sampling ratio in fitting data
feed_data = feedData(data_fit = gene_downsampling, data_predict=data_df, sampling_ratio=sampling_ratio) # default sampling_ratio=0.5

model_save_path=None
#model_dir=None
for epoch in epoches: # delete
    #############################################
    ###########  Fitting and Predict ############
    #############################################
    print('-------epoch----------------')
    print(epoch)
    #epoch = 1
    # cost_version=1
    brief, detail = train(feed_data,
                            model_path=model_dir, 
                            max_epoches=epoch, 
                            model_save_path=model_save_path,
                            result_path=output_path,
                            n_jobs=num_jobs,
                            learning_rate=learning_rate,
                            cost_version=cost_version,
                            cost2_cutoff=cost2_cutoff,
                            n_neighbors=n_neighbors,
                            cost1_ratio=cost1_ratio,
                            optimizer=optimizer,
                            gene_shape_classify_dict=gene_shape_classify_dict,
                            with_trace_cost=True,
                            with_corrcoef_cost=True)
    detail["alpha_new"]=detail["alpha"]/detail["beta"]
    detail["beta_new"]=detail["beta"]/detail["beta"]
    detail["gamma_new"]=detail["gamma"]/detail["beta"]
    detailfinfo="e"+str(epoch)
    ##########################################
    ###########       Plot        ############
    ##########################################
    if plot_trigger:
        #color_map="coolwarm"
        #alpha_inside=1
        #pointsize=50
        pointsize=120
        color_scatter="#95D9EF" #blue
        color_map=None
        alpha_inside=0.3

        #color_scatter="#DAC9E7" #light purple
        #color_scatter="#8D71B3" #deep purple
        #alpha_inside=1
        vmin=0
        vmax=5
        step_i=20
        step_j=20
        frame_invisable=True

        for i in gene_choice:
            save_path=output_path+i+"_"+"e"+str(epoch)+".pdf"# notice: changed
            velocity_plot(detail, [i],detailfinfo,color_scatter,pointsize,alpha_inside,color_map,vmin,vmax,save_path,step_i=step_i,step_j=step_j) # from cell dancer
            save_path_validation=output_path+i+"_validation_"+"e"+str(epoch)+".pdf"
            if epoch>0:vaildation_plot(gene=i,validation_result=brief[brief["gene_name"]==i],save_path_validation=save_path_validation)

time_end=time.time()
print('time spent: ',(time_end-time_start)/60,' min')   
