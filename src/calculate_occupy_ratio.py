# output the gene_occupy_ratio.csv of occupy ratio for each gene
# output header: gene_choice,ratio

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import *
import seaborn as sns
import sys

def identify_in_grid(u, s, onegene_u0_s0):
    select_cell =onegene_u0_s0[(onegene_u0_s0[:,0]>u[0]) & (onegene_u0_s0[:,0]<u[1]) & (onegene_u0_s0[:,1]>s[0]) & (onegene_u0_s0[:,1]<s[1]), :]
    if select_cell.shape[0]==0:
        return False
    else:
        return True

def build_grid_list(u_fragment,s_fragment,onegene_u0_s0):
    min_u0 = min(onegene_u0_s0[:,0])
    max_u0 = max(onegene_u0_s0[:,0])
    min_s0 = min(onegene_u0_s0[:,1])
    max_s0 = max(onegene_u0_s0[:,1])
    u0_coordinate=np.linspace(start=min_u0, stop=max_u0, num=u_fragment+1).tolist()
    s0_coordinate=np.linspace(start=min_s0, stop=max_s0, num=s_fragment+1).tolist()
    u0_array = np.array([u0_coordinate[0:(len(u0_coordinate)-1)], u0_coordinate[1:(len(u0_coordinate))]]).T
    s0_array = np.array([s0_coordinate[0:(len(s0_coordinate)-1)], s0_coordinate[1:(len(s0_coordinate))]]).T
    return u0_array, s0_array

def calculate_occupy_ratio(gene_choice,data, u_fragment, s_fragment):
    # data = raw_data2
    ratio = np.empty([len(gene_choice), 1])
    for idx, gene in enumerate(gene_choice):
        print(idx)
        onegene_u0_s0=data[data.gene_list==gene][['u0','s0']].to_numpy()
        u_grid, s_grid=build_grid_list(u_fragment,s_fragment,onegene_u0_s0)
        # occupy = np.empty([1, u_grid.shape[0]*s_grid.shape[0]])
        occupy = 0
        for i, s in enumerate(s_grid):
            for j,u in enumerate(u_grid):
                #print(one_grid)
                if identify_in_grid(u, s,onegene_u0_s0):
                    # print(1)
                    occupy = occupy + 1
        occupy_ratio=occupy/(u_grid.shape[0]*s_grid.shape[0])
        # print('occupy_ratio for '+gene+"="+str(occupy_ratio))
        ratio[idx,0] = occupy_ratio
    ratio2 = pd.DataFrame({'gene_choice': gene_choice, 'ratio': ratio[:,0]})
    return(ratio2)



if __name__ == "__main__":
    from utilities import set_rcParams
    #from utilities import *
    set_rcParams()
    import warnings
    import os
    import sys
    warnings.filterwarnings("ignore")
    from celldancer_plots import *
    from sampling import *

    time_start=time.time()

    print('calculate_occupy_ratio.py')
    print('time_start'+str(time_start))
    print('')

    use_all_gene=True
    plot_trigger=False
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
        optimizer=sys.argv[15] #["SGD","Adam"]
        full_start=int(sys.argv[16])
        full_end =int(sys.argv[17])

    # set data_source
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
        gene_choice=['Ank','Btbd17','Cdk1','Cpe','Gnao1',
                    'Gng12','Map1b','Mapre3','Nnat','Ntrk2',
                    'Pak3','Pcsk2','Ppp3ca','Rap1b','Rbfox3',
                    'Smoc1','Sulf2','Tmem163','Top2a','Tspan7',
                    "Pdgfra","Igfbpl1"]
        gene_choice=['Ank','Btbd17','Cdk1','Cpe','Gnao1',
                    'Gng12','Map1b','Mapre3','Nnat','Ntrk2',
                    'Pak3','Pcsk2','Ppp3ca','Rap1b','Rbfox3',
                    'Smoc1','Sulf2','Tmem163','Top2a','Tspan7',
                    "Pdgfra","Igfbpl1",#
                    #Added GENE from page 11 of https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0414-6/MediaObjects/41586_2018_414_MOESM3_ESM.pdf
                    "Syngr1","Fam210b","Meg3","Fam19a2","Kcnc3","Dscam"]#"Hagh"] time spent:  46.042665135860446  min
        gene_choice=['Adam23','Arid5b','Blcap','Coch','Dcx',
                    'Elavl2','Elavl3','Elavl4','Eomes','Eps15',
                    'Fam210b','Foxk2','Gpc6','Icam5','Kcnd2',
                    'Pfkp','Psd3','Sult2b1','Thy1','Car2','Clip3','Ntrk2','Nnat'] #21 genes
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

    load_raw_data=pd.read_csv(raw_data_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])
    if use_all_gene: 
        # weather to use the full_start and full_end to set the range of genes to be run
        # gene_choice=list(set(load_raw_data.gene_list))[full_start: full_end]
        gene_choice=list(set(load_raw_data.gene_list))

    gene_choice.sort()
    raw_data2 = load_raw_data[load_raw_data.gene_list.isin(gene_choice)][['gene_list', 'u0','s0']]
    
    #################################
    ## Calculate gene occupy ratio ##
    #################################
    gene_occupy_ratio = calculate_occupy_ratio(gene_choice, raw_data2, 30, 30)
    gene_occupy_ratio.sort_values(by = ['ratio'])
    gene_occupy_ratio.to_csv(output_path+'gene_occupy_ratio('+str(full_start)+','+str(full_end)+').csv')

    # generate occupy csv file for each gene
    load_raw_data=pd.read_csv(raw_data_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])
    raw_data2 = load_raw_data[load_raw_data.gene_list.isin(gene_choice)][['gene_list', 'u0','s0']]

    # build csv - The first gene to keep header in the csv
    g=gene_choice[0]
    i=i+1
    print("processing:"+str(i)+"th")
    gene_occupy_ratio = calculate_occupy_ratio([g], raw_data2, 30, 30)
    gene_occupy_ratio.to_csv('output/gene_occupy_ratio/denGyr/gene_occupy_ratio.csv',mode='a',header=True,index=False)

    # build csv - The genes below to append
    for g in gene_choice[1:]:
        i=i+1
        print("processing:"+str(i)+"th")
        gene_occupy_ratio = calculate_occupy_ratio([g], raw_data2, 30, 30)
        gene_occupy_ratio.to_csv('output/gene_occupy_ratio/denGyr/gene_occupy_ratio.csv',mode='a',header=False,index=False)

    #########################################
    ## Dendisty plot for gene occupy ratio ##
    #########################################
    gene_occupy_ratio=pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/gene_occupy_ratio/denGyr/gene_occupy_ratio.csv')

    # plot the density for ratio of each gene
    import seaborn as sns
    sns.distplot(gene_occupy_ratio['ratio'],hist=False,color='black')
    plt.title('occupy ratio')
    plt.savefig('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/gene_occupy_ratio/denGyr/dist_gene_occupy_ratio.pdf')

    time_end=time.time()
    print('time spent: ',(time_end-time_start)/60,' min')   



