import matplotlib
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../src')
from celldancer_plots import *

detail=pd.read_csv('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/simulation/data/normal/detail_e200.csv')
detail['alpha_norm']=detail['alpha']/detail['beta']
detail['gamma_norm']=detail['gamma']/detail['beta']

gene='simulation810'
gene_list=list(set(detail.gene_name))
gene_list.sort()
for gene in gene_list[0:100]:
    velocity_plot([gene],detail,colormap='RdYIBu',para='alpha_norm',save_path=None)

colormap_list=['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

for gene in ['simulation1','simulation149']:
    save_path='/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/simulation/data/normal/'+gene+'.pdf'
    velocity_plot(gene,detail,colormap='RdBu',para='alpha_norm',save_path=save_path)
    # velocity_plot(gene,detail,colormap='RdBu',para='gamma_norm',save_path=None,v_min=0.35,v_max=1)
    # velocity_plot(gene,detail,colormap='RdBu',para='gamma_norm',save_path=None,v_min=0.35,v_max=1,alpha_inside=1)

    #velocity_plot(gene,detail,colormap='RdBu',para='gamma_norm',save_path=None,alpha_inside=1)
