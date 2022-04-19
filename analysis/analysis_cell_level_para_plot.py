# https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb

import pandas as pd
from velocity_plot import velocity_plot as vpl
import pandas as pd


# detail = pd.read_csv("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/denGyrLr0.1Costv2C1r0.8C2cf0.3Downneighbors0_200_200Ratio0.125N30OSGD/detail_e500.csv")
# detail = pd.read_csv("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/denGyrLr1e-05Costv3C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/detail_e2000.csv")
# detail = pd.read_csv("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/denGyrLr0.001Costv3C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/detail_e200.csv")
# detail = pd.read_csv("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/denGyrLr0.001Costv3C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/detail_e200.csv")
# detail = pd.read_csv("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/denGyrLr0.001Costv1C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/detail_e200.csv")

detail = pd.read_csv("/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/20220124_costv1_sampled_genes/detail_velocity_plot/detail_e200.csv")
read_raw_data=pd.read_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full_steroboard_genes.csv')

# color_map_list=[color_map1,color_map1,color_map1,color_map4,color_map4]
para_list=['alpha','beta','gamma','s0','u0']

gene_choice=['Ank','Btbd17','Cdk1','Cpe','Gnao1',
            'Gng12','Map1b','Mapre3','Nnat','Ntrk2',
            'Pak3','Pcsk2','Ppp3ca','Rap1b','Rbfox3',
            'Smoc1','Sulf2','Tmem163','Top2a','Tspan7',
            "Pdgfra","Igfbpl1",#
            #Added GENE from page 11 of https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0414-6/MediaObjects/41586_2018_414_MOESM3_ESM.pdf
            "Syngr1","Fam210b","Meg3","Fam19a2","Kcnc3","Dscam","Hagh"] 
gene_choice=['Gnao1']
gene_choice=['Elavl4','Dcx']
gene_choice=['Ntrk2','Dcx']

save_path='/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/20220124_costv1_sampled_genes/20220124_costv1_cell_level_para/NIPC_area'
save_path=None

# plot with fixed clusters
cluster_choice=['nIPC']
vpl.cell_level_para_plot(read_raw_data,detail,gene_choice,para_list,cluster_choice=cluster_choice,save_path=save_path,pointsize=50,alpha=0.3)

# regular plot
vpl.cell_level_para_plot(read_raw_data,detail,gene_choice,para_list,save_path=save_path)




