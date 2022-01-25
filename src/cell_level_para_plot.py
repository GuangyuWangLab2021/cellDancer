# https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colormap import *
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

embedding = np.loadtxt(open("velocyto/vlm_variables/vlm_embedding.csv", "rb"), delimiter=",")


# embedding
# embedding.shape # (18140, 2)
# plt.scatter(embedding[:,0],embedding[:,1])

#detail = pd.read_csv("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/denGyrLr0.1Costv2C1r0.8C2cf0.3Downneighbors0_200_200Ratio0.125N30OSGD/detail_e500.csv")

# detail = pd.read_csv("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/denGyrLr1e-05Costv3C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/detail_e2000.csv")
# detail = pd.read_csv("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/denGyrLr0.001Costv3C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/detail_e200.csv")

# detail = pd.read_csv("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/denGyrLr0.001Costv3C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/detail_e200.csv")
detail = pd.read_csv("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/denGyrLr0.001Costv1C1r0.5C2cf0.3Downneighbors0_200_200Ratio0.125N30OAdam/detail_e200.csv")

gene_choice=['Ank','Btbd17','Cdk1','Cpe','Gnao1',
            'Gng12','Map1b','Mapre3','Nnat','Ntrk2',
            'Pak3','Pcsk2','Ppp3ca','Rap1b','Rbfox3',
            'Smoc1','Sulf2','Tmem163','Top2a','Tspan7',
            "Pdgfra","Igfbpl1",#
            #Added GENE from page 11 of https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0414-6/MediaObjects/41586_2018_414_MOESM3_ESM.pdf
            "Syngr1","Fam210b","Meg3","Fam19a2","Kcnc3","Dscam","Hagh"] 

gene_choice=['Gnao1']

gene_choice=['Elavl4','Dcx']
gene_choice=['Ntrk2']

cmap1 = LinearSegmentedColormap.from_list("mycmap", zissou2)
color_map1=cmap1
cmap1 = LinearSegmentedColormap.from_list("mycmap", beach)
color_map2=cmap1
cmap1 = LinearSegmentedColormap.from_list("mycmap", purpleOrange)
color_map3=cmap1
cmap1 = LinearSegmentedColormap.from_list("mycmap", fireworks3)
color_map4=cmap1


color_map_list=[color_map1,color_map1,color_map1,color_map4,color_map4]
para_list=['alpha','beta','gamma','s0','u0']
# color_map_list=[color_map4]
# para_list=['u0']


for para,color_map in zip(para_list,color_map_list):
    for gene_name in gene_choice:
        #gene_name='Ntrk2'
        one_gene=detail[detail.gene_name==gene_name]
        layer=plt.scatter(embedding[:,0],embedding[:,1],s=0.2,c=one_gene[para],cmap=color_map)
        plt.title(gene_name+" "+para)
        plt.colorbar(layer)
        #plt.savefig(('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/cell_level_para/dengyr/'+gene_name+'_'+para+'.png'),dpi=300)
        plt.savefig(('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/data/velocyto/neuro/cell_level_para/20220124_costv1/'+gene_name+'_'+para+'.png'),dpi=300)
        plt.show()

# np.corrcoef(detail[detail.gene_name=='Nnat'].alpha, detail[detail.gene_name=='Nnat'].u0) # 0.5
# np.corrcoef(detail[detail.gene_name=='Nnat'].beta, detail[detail.gene_name=='Nnat'].s0) # -0.82 sum = -0.32

# np.corrcoef(detail[detail.gene_name=='Ntrk2'].alpha, detail[detail.gene_name=='Ntrk2'].u0) # 0.94224016
# np.corrcoef(detail[detail.gene_name=='Ntrk2'].beta, detail[detail.gene_name=='Ntrk2'].s0) # 0.21682215 sum = 1.15824016

# np.corrcoef(detail[detail.gene_name=='Gnao1'].alpha, detail[detail.gene_name=='Gnao1'].u0) # 0.75263632
# np.corrcoef(detail[detail.gene_name=='Gnao1'].beta, detail[detail.gene_name=='Gnao1'].s0) # -0.74187894 sum = 0.01

# np.corrcoef(detail[detail.gene_name=='Cpe'].alpha, detail[detail.gene_name=='Cpe'].u0) # 0.99524385
# np.corrcoef(detail[detail.gene_name=='Cpe'].beta, detail[detail.gene_name=='Cpe'].s0) # 0.6725009 sum = 1.66

# np.corrcoef(detail[detail.gene_name=='Ank'].alpha, detail[detail.gene_name=='Ank'].u0) # 0.44617186
# np.corrcoef(detail[detail.gene_name=='Ank'].beta, detail[detail.gene_name=='Ank'].s0) # -0.61965973 sum = -0.17348787

# #sperman

# plt.scatter(embedding[:,0],embedding[:,1],c = detail[detail.gene_name=='Ank'].alpha,s=0.5)
# plt.scatter(embedding[:,0],embedding[:,1],c = detail[detail.gene_name=='Ank'].u0,s=0.5)
# plt.scatter(embedding[:,0],embedding[:,1],c = detail[detail.gene_name=='Ank'].beta,s=0.5)
# plt.scatter(embedding[:,0],embedding[:,1],c = detail[detail.gene_name=='Ank'].s0,s=0.5)




# Cell level Range plot

gene='Elavl4'
u_range_1=0.4
u_range_2=0.5
s_range_1=0.5
s_range_2=1.0

raw_data_path="/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/veloNN-main_github/src/velonn/data/denGyr_full.csv" #["data/denGyr.csv","data/scv_data.csv"]
load_raw_data=pd.read_csv(raw_data_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])
onegene=load_raw_data[load_raw_data.gene_list==gene]
onegene['u_range']='red'
onegene['s_range']='red'
onegene.loc[onegene.u0<u_range_1, 'u_range'] = 'blue'
onegene.loc[onegene.u0>u_range_2, 'u_range'] = 'blue'

onegene.loc[onegene.s0<s_range_1, 's_range'] = 'blue'
onegene.loc[onegene.s0>s_range_2, 's_range'] = 'blue'

plt.scatter(onegene.embedding1,onegene.embedding2, c=onegene.u_range,s=0.3)
plt.title(gene+' u_range('+str(u_range_1)+","+str(u_range_2)+')')
plt.savefig(('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/cell_level_para_range/dengyr/'+gene+'_u_range('+str(u_range_1)+","+str(u_range_2)+')''.pdf'))

plt.scatter(onegene.embedding1,onegene.embedding2, c=onegene.s_range,s=0.3)
plt.title(gene+' s_range('+str(s_range_1)+","+str(s_range_2)+')')
plt.savefig(('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/cell_level_para_range/dengyr/'+gene+'_s_range('+str(s_range_1)+","+str(s_range_2)+')''.pdf'))

onegene

for cls,color in zip(set(onegene.clusters),ironMan_2):

#for cls,color in zip(['ImmAstro','nIPC'],['blue','red']):
    print(cls)
    plt.scatter(onegene[onegene.clusters==cls].s0,onegene[onegene.clusters==cls].u0,c=color,s=2,label=color)

