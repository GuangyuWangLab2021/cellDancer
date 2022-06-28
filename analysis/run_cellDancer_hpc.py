import sys
#sys.path.append('../src')

#from utilities import *

import warnings
import os
import argparse
warnings.filterwarnings("ignore")
import time

import pandas as pd

if __name__ == "__main__":
    sys.path.append('.')
    from constant import *
    from sampling import *
    from sampling import *
    from velocity_estimation import *
    from utilities import set_rcParams
    from velocity_plot import velocity_plot as vpl
else:
    from .constant import *
    from .sampling import *
    from .sampling import *
    from .velocity_estimation import *
    from .utilities import set_rcParams
    from .velocity_plot import velocity_plot as vpl

set_rcParams()

time_start=time.time()

print('\nvelocity_estimate.py version 1.0.0')
print('python velocity_estimate.py')
print('time_start'+str(time_start))
print('')

use_all_gene=True
plot_trigger=False

print("---Parameters---")
for i in sys.argv:
    print(i)
print("----------------")

task_id =sys.argv[1]
data_source =sys.argv[2]
platform =sys.argv[3]
epoch =int(sys.argv[4])
check_n_epoch =sys.argv[5]
num_jobs =int(sys.argv[6])
learning_rate =float(sys.argv[7])
cost2_cutoff =float(sys.argv[8])
downsample_method =sys.argv[9]
downsample_target_amount =int(sys.argv[10])
step_i =int(sys.argv[11])
step_j =int(sys.argv[12])
sampling_ratio =float(sys.argv[13])
n_neighbors =int(sys.argv[14])
optimizer =sys.argv[15] #["SGD","Adam"] default->adam
trace_cost_ratio = float(sys.argv[16])
corrcoef_cost_ratio = float(sys.argv[17])
raw_path =sys.argv[18]
out_path =sys.argv[19]

# new feature
n_neighbors_downsample=int(sys.argv[20])
auto_downsample=bool(int(sys.argv[21]))
auto_norm_u_s=bool(int(sys.argv[22]))

cost_type=sys.argv[23]
average_cost_window_size=int(sys.argv[24])
#print(sys.argv[25])
patience=int(sys.argv[25])
smooth_weight=float(sys.argv[26])
binning=bool(int(sys.argv[27]))

startpoint=int(sys.argv[28])
endpoint=int(sys.argv[29])

model_save_path=None
print(raw_path)

#### mkdir for output_path with parameters(naming)
folder_name=(task_id + data_source+
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
    gene_choice=load_raw_data.gene_list.drop_duplicates()
    # gene_choice.sort()
    gene_choice=gene_choice[startpoint:endpoint]
    print('---gene_choice---')
    print(gene_choice)
else:
    gene_choice = select_gene_set(data_source)

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
# if plot_trigger:
#     for i in gene_choice:
#         save_path=output_path+i+"_"+"e"+str(epoch)+".pdf"# notice: changed
#         vpl.velocity_gene(i,detail,save_path=save_path) # from cell dancer # from cell dancer
#         save_path_validation=output_path+i+"_validation_"+"e"+str(epoch)+".pdf"
#         if epoch>0:vpl.vaildation_plot(gene=i,validation_result=brief[brief["gene_name"]==i],save_path_validation=save_path_validation)

time_end=time.time()
print('time spent: ',(time_end-time_start)/60,' min')   
