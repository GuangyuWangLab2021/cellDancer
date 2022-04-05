from turtle import pd
import pandas as pd
import matplotlib.pyplot as plt
from colormap import *

# Cell level Range plot - temp task

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

