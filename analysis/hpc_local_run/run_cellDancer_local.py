import sys
# sys.path.append('../src')
sys.path.append('/Users/wanglab/Documents/ShengyuLi/Velocity/bin/celldancer_polish/src')
#from utilities import *

import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import time
import warnings
import pandas as pd
if __name__ == "__main__":
    sys.path.append('.')
    from constant import *
    from sampling import *
    from velocity_estimation import *
    from utilities import set_rcParams
    # from velocity_plot import velocity_plot as vpl
else:
    from .constant import *
    from .sampling import *
    from .sampling import *
    from .velocity_estimation import *
    from .utilities import set_rcParams
    # from .velocity_plot import velocity_plot as vpl

from cdplot.scatter_gene import velocity_gene

set_rcParams()

time_start=time.time()

print('\nvelocity_estimate.py version 1.0.0')
print('python velocity_estimate.py')
print('time_start'+str(time_start))
print('')

use_all_gene=False
plot_trigger=False

# model_dir = {"Sulf2": '/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development_20220128/src/model/Sulf2/Sulf2.pt', 
#             "Ntrk2_e500": "/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Ntrk2_e500/Ntrk2_e500.pt"}
# config = pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/config/config_test20220128.txt', sep=';',header=None)
# config = pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/config/config_test20220128.txt', sep=';',header=None)
# config = pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/velocity_result/celldancer/config_ratio.txt', sep=';',header=None)

# config = pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/Gastrulation/result_detailcsv/config_sample.txt', sep=';',header=None)
# config = pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/Gastrulation/velocity_result/result_detailcsv/Hba-x/config_sample.txt', sep=';',header=None)
# config = pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/Gastrulation/velocity_result/result_detailcsv/Smim1/config_sample.txt', sep=';',header=None)
# config = pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/Gastrulation/velocity_result/result_detailcsv/test_fun_norm_us/config_sample.txt', sep=';',header=None)
#config = pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/Gastrulation/velocity_result/result_detailcsv/2000_genes/config_sample.txt', sep=';',header=None)
config = pd.read_csv('/Users/wanglab/Documents/ShengyuLi/Velocity/data/Gastrulation/velocity_result/result_detailcsv/polish/config_sample.txt', sep=';',header=None)

task_id=config.iloc[0][0]
data_source = config.iloc[0][1]
platform = config.iloc[0][2]
epoch=int(config.iloc[0][3])
check_n_epoch=config.iloc[0][4]
num_jobs=int(config.iloc[0][5])
learning_rate=float(config.iloc[0][6])
cost2_cutoff=float(config.iloc[0][7])
downsample_method=config.iloc[0][8]
downsample_target_amount=int(config.iloc[0][9])
step_i=int(config.iloc[0][10])
step_j=int(config.iloc[0][11])
sampling_ratio=float(config.iloc[0][12])
n_neighbors=int(config.iloc[0][13])
optimizer=config.iloc[0][14] #["SGD","Adam"]  # set to Adam
trace_cost_ratio=float(config.iloc[0][15])
corrcoef_cost_ratio=float(config.iloc[0][16])
raw_path=config.iloc[0][17]
out_path=config.iloc[0][18]

# new feature
n_neighbors_downsample=int(config.iloc[0][19])
auto_downsample=bool(int(config.iloc[0][20]))
auto_norm_u_s=bool(int(config.iloc[0][21]))

cost_type=config.iloc[0][22]
average_cost_window_size=int(config.iloc[0][23])
patience=int(config.iloc[0][24])
smooth_weight=float(config.iloc[0][25])
binning=bool(int(config.iloc[0][26]))


model_save_path=None
print(raw_path)

# startpoint=int(config.iloc[0][22])
# endpoint=int(config.iloc[0][23])

#### mkdir for output_path with parameters(naming)
folder_name=(data_source+
    "epoch"+str(epoch)+
    "check_n"+str(check_n_epoch)+
    "Lr"+str(learning_rate)+
    "C2cf"+str(cost2_cutoff)+
    "Down"+downsample_method+str(downsample_target_amount)+"_"+str(step_i)+"_"+str(step_j)+
    "Ratio"+str(sampling_ratio)+
    "N"+str(n_neighbors)+
    "O"+optimizer+
    "traceR"+str(trace_cost_ratio)+
    "corrcoefR"+str(corrcoef_cost_ratio)+
    "nD"+str(n_neighbors_downsample)+
    "autoD"+str(auto_downsample)+
    "autoN"+str(auto_norm_u_s)+
    "costT"+str(cost_type)+
    "avgCWin"+str(average_cost_window_size)+
    "p"+str(patience)+
    "smoW"+str(smooth_weight)+
    'bin'+str(binning))

output_path=(out_path+folder_name+"/")
if os.path.isdir(output_path):pass
else:os.mkdir(output_path)


######################################################
############             Guangyu          ############
######################################################
#raw_data_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/data/simulation_data/one_gene.csv'

# load_raw_data=pd.read_csv(raw_path)
# load_raw_data=pd.read_csv(raw_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])
load_raw_data=pd.read_csv(raw_path)

# burning
gene_choice=['Sulf2']
brief, detail = train(load_raw_data,
                      downsample_method=downsample_method,
                      n_neighbors_downsample=n_neighbors_downsample,
                      auto_downsample=auto_downsample,
                      auto_norm_u_s=auto_norm_u_s,
                        max_epoches=epoch,
                        check_n_epoch=check_n_epoch,
                        model_save_path=model_save_path,
                        result_path=output_path,
                        n_jobs=num_jobs,
                        learning_rate=learning_rate,
                        cost2_cutoff=cost2_cutoff,
                        n_neighbors=n_neighbors,
                        optimizer=optimizer,
                        trace_cost_ratio=trace_cost_ratio,
                        corrcoef_cost_ratio=corrcoef_cost_ratio,
                      gene_choice=gene_choice,
                     cost_type=cost_type,
                      average_cost_window_size=average_cost_window_size,
                     patience=patience,
                     smooth_weight=smooth_weight,
                     binning=binning,
                     sampling_ratio=sampling_ratio)

# END burning
if use_all_gene: 
    gene_choice=load_raw_data.gene_list.drop_duplicates()[0:2]
    # gene_choice.sort()
    # gene_choice=gene_choice
    # gene_choice=['Hba-x']
    # gene_choice=['H2afv']
    # gene_choice=['Smim1','Hba-x']
    gene_choice=['Smim1','Hba-x']
    print('---gene_choice---')
    print(gene_choice)
else:
    gene_choice = select_gene_set(data_source)

# gene_choice=None

model_save_path=None
#model_dir=None

#############################################
###########  Fitting and Predict ############
#############################################
print('-------epoch----------------')
print(epoch)
# cost_version=1
brief, detail = train(load_raw_data,
                      downsample_method=downsample_method,
                      n_neighbors_downsample=n_neighbors_downsample,
                      auto_downsample=auto_downsample,
                      auto_norm_u_s=auto_norm_u_s,
                        max_epoches=epoch,
                        check_n_epoch=check_n_epoch,
                        model_save_path=model_save_path,
                        result_path=output_path,
                        n_jobs=num_jobs,
                        learning_rate=learning_rate,
                        cost2_cutoff=cost2_cutoff,
                        n_neighbors=n_neighbors,
                        optimizer=optimizer,
                        trace_cost_ratio=trace_cost_ratio,
                        corrcoef_cost_ratio=corrcoef_cost_ratio,
                      gene_choice=gene_choice,
                     cost_type=cost_type,
                      average_cost_window_size=average_cost_window_size,
                     patience=patience,
                     smooth_weight=smooth_weight,
                     binning=binning,
                     sampling_ratio=sampling_ratio)
detailfinfo="e"+str(epoch)
##########################################
###########       Plot        ############
##########################################
if plot_trigger:
    for i in gene_choice:
        save_path=output_path+i+"_"+"e"+str(epoch)+".pdf"# notice: changed

time_end=time.time()
print('time spent: ',(time_end-time_start)/60,' min')   
