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


time_start=time.time()

print('\nvelocity_estimate.py version 1.0.0')
print('python velocity_estimate.py')
print('time_start'+str(time_start))
print('')

use_all_gene=True
plot_trigger=True
platform = 'hpc'
if platform == "local":
    #model_dir='model/model2'
    #model_dir='model/Ntrk2_e500'
    model_dir = {"Sulf2": '/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Sulf2/Sulf2.pt', 
                "Ntrk2_e500": "/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Ntrk2_e500/Ntrk2_e500.pt"}
    config = pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/config/config_test.txt', sep=';',header=None)
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
    optimizer=config.iloc[0][14] #["SGD","Adam"]
elif platform == 'hpc':
    model_dir="/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/model2"
    #model_dir="/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/Ntrk2_e500"
    #model_dir = {'Sulf2': '/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/Sulf2', 
    #            'Ntrk2_e500': '/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/Ntrk2_e500'}
    print("---Parameters---")
    for i in sys.argv:
        print(i)
    print("----------------")
    data_source=sys.argv[1]
    platform=sys.argv[2]
    epoches=[int(sys.argv[3])]
    num_jobs=int(sys.argv[4])
    learning_rate=float(sys.argv[5])
    cost_version=int(sys.argv[6])
    cost1_ratio=float(sys.argv[7])
    cost2_cutoff=float(sys.argv[8])
    downsample_method=sys.argv[9]
    downsample_target_amount=int(sys.argv[10])
    step_i=int(sys.argv[11])
    step_j=int(sys.argv[12])
    sampling_ratio=float(sys.argv[13])
    n_neighbors=int(sys.argv[14])
    optimizer=sys.argv[15] #["SGD","Adam"] default->adam
    full_start=int(sys.argv[16])
    full_end =int(sys.argv[17])

# set data_source
if data_source=="mal":
    if platform=="local":
        raw_data_path=""
    elif platform=="hpc":
        raw_data_path_hpc='/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/mal/input_data.csv'        
        raw_data_path=raw_data_path_hpc
    gene_choice=["Ank","Abcc8","Tcp11","Nfib","Ppp3ca",
            "Rbfox3","Cdk1","Gng12","Map1b","Cpe",
            "Gnao1","Pcsk2","Tmem163","Pak3","Wfdc15b",
            "Nnat","Anxa4","Actn4","Btbd17","Dcdc2a",
            "Adk","Smoc1","Mapre3","Pim2","Tspan7",
            "Top2a","Rap1b","Sulf2"]
    #gene_choice=["Sulf2","Top2a","Abcc8"]

if data_source=="scv":
    if platform=="local":
        raw_data_path="/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/data/scv_data_full.csv" #["velocyto/data/denGyr.csv","data/scv_data.csv"]
    elif platform=="hpc":
        raw_data_path_hpc='/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/scv_data_full.csv'        #["/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/velocyto/data/denGyr.csv","/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/data/scv_data.csv"]
        raw_data_path=raw_data_path_hpc
    gene_choice=["Ank","Abcc8","Tcp11","Nfib","Ppp3ca",
            "Rbfox3","Cdk1","Gng12","Map1b","Cpe",
            "Gnao1","Pcsk2","Tmem163","Pak3","Wfdc15b",
            "Nnat","Anxa4","Actn4","Btbd17","Dcdc2a",
            "Adk","Smoc1","Mapre3","Pim2","Tspan7",
            "Top2a","Rap1b","Sulf2"]
    #gene_choice=["Sulf2","Top2a","Abcc8"]

elif data_source=="denGyr":
    if platform=="local":
        raw_data_path="/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/veloNN-main_github/src/velonn/data/denGyr_full.csv" #["data/denGyr.csv","data/scv_data.csv"]
    elif platform=="hpc":
        raw_data_path_hpc="/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/denGyr_full.csv"         #["/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/velocyto/data/denGyr.csv","/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/data/scv_data.csv"]
        raw_data_path=raw_data_path_hpc
    # gene_choice=['Ank','Btbd17','Cdk1','Cpe','Gnao1',
    #             'Gng12','Map1b','Mapre3','Nnat','Ntrk2',
    #             'Pak3','Pcsk2','Ppp3ca','Rap1b','Rbfox3',
    #             'Smoc1','Sulf2','Tmem163','Top2a','Tspan7',
    #             "Pdgfra","Igfbpl1"]
    # gene_choice=['Ank','Btbd17','Cdk1','Cpe','Gnao1',
    #             'Gng12','Map1b','Mapre3','Nnat','Ntrk2',
    #             'Pak3','Pcsk2','Ppp3ca','Rap1b','Rbfox3',
    #             'Smoc1','Sulf2','Tmem163','Top2a','Tspan7',
    #             "Pdgfra","Igfbpl1",#
    #             #Added GENE from page 11 of https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0414-6/MediaObjects/41586_2018_414_MOESM3_ESM.pdf
    #             "Syngr1","Fam210b","Meg3","Fam19a2","Kcnc3","Dscam"]#"Hagh"] time spent:  46.042665135860446  min
    # gene_choice=['Adam23','Arid5b','Blcap','Coch','Dcx',
    #             'Elavl2','Elavl3','Elavl4','Eomes','Eps15',
    #             'Fam210b','Foxk2','Gpc6','Icam5','Kcnd2',
    #             'Pfkp','Psd3','Sult2b1','Thy1','Car2','Clip3','Ntrk2','Nnat'] #21 genes
    gene_choice=['Rimbp2','Dctn3','Psd3','Dcx','Elavl4','Ntrk2']
    # gene_choice=['Nnat','Ntrk2','Gnao1','Cpe','Ank']
    # gene_choice=["Ank"]
    #gene_choice=["Gnao1"]
    #gene_choice=["Dcx",'Elavl4']
    #gene_choice=["Ntrk2"]
    #gene_choice=["Nnat"]
    #gene_choice=["Kcnc3","Dscam"]
    #gene_choice=['Elavl4','Eomes','Dcx','Psd3','Sult2b1','Thy1','Car2']
    #gene_choice=['Elavl4']

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
if platform=="local":
    output_path=("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/"+folder_name+"/")
    if os.path.isdir(output_path):pass
    else:os.mkdir(output_path)
elif platform=="hpc":
    output_path=("/condo/wanglab/tmhsxl98/Velocity/cell_dancer/output/detailcsv/adj_e/"+folder_name+"/")
    if os.path.isdir(output_path):pass
    else:os.mkdir(output_path)

######################################################
############             Guangyu          ############
######################################################
#raw_data_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/data/simulation_data/one_gene.csv'

load_raw_data=pd.read_csv(raw_data_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])
if use_all_gene: 
    gene_choice=list(set(load_raw_data.gene_list))
    gene_choice.sort()
    gene_choice=gene_choice[full_start: full_end]
    print('---gene_choice---')
    print(gene_choice)

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
if platform=="local":
    gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Sulf2','model_type_dir']='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Sulf2/Sulf2.pt'
    gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Ntrk2_e500','model_type_dir']='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Ntrk2_e500/Ntrk2_e500.pt'
elif platform=="hpc":
    gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Sulf2','model_type_dir']='/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/Sulf2/Sulf2.pt'
    gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Ntrk2_e500','model_type_dir']='/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/Ntrk2_e500/Ntrk2_e500.pt'

# set fitting data, data to be predicted, and sampling ratio in fitting data
feed_data = feedData(data_fit = gene_downsampling, data_predict=data_df, sampling_ratio=sampling_ratio) # default sampling_ratio=0.5
#epoches = [5,10,100,300,500]
#epoches = [5]
#model_save_path="model_Ntrk2/Ntrk2_500.pt"
model_save_path=None
#model_dir=None
for epoch in epoches: # delete
    #############################################
    ###########  Fitting and Predict ############
    #############################################
    print('-------epoch----------------')
    print(epoch)
    #epoch = 1
    cost_version=1
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

    #detail.to_csv(output_path+"detail_e"+str(epoch)+".csv")
    #brief.to_csv(output_path+"brief_e"+str(epoch)+".csv")
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

gene_choice=list(set(load_raw_data.gene_list))
raw_data2 = load_raw_data[load_raw_data.gene_list.isin(gene_choice)][['gene_list', 'u0','s0']]
ratio = calculate_occupy_ratio(gene_choice, raw_data2, 30, 30)
ratio.sort_values(by = ['ratio'])