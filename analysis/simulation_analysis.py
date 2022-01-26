import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


backward = pd.read_csv('/Users/guangyu/OneDrive - Houston Methodist/Work/cellDancer/data/simulation/simulation_results/backward/detail_e200.csv')
backward_back_ground_true = pd.read_pickle('/Users/guangyu/OneDrive - Houston Methodist/Work/cellDancer/data/simulation/data/data_backward.pkl')

backward = backward.sort_values(by = ['gene_name','id'])


m1 = []
m2 = []

for i in range(1000):
    # i=0
    backward_one = backward[backward['gene_name']=='simulation'+str(i)]
    backward_back_ground_true_one = backward_back_ground_true[backward_back_ground_true['gene_list']=='simulation'+str(i)]

    m1.append(np.mean(backward_one['beta']/backward_one['gamma'].to_list()))
    m2.append(backward_back_ground_true_one['beta'][0]/backward_back_ground_true_one['gamma'][0])

plt.scatter(np.log(m1), np.log(m2)) 
plt.savefig('/Users/guangyu/OneDrive - Houston Methodist/Work/cellDancer/data/simulation/data/data_backward.pdf')

np.corrcoef(m1,m2) # 0.96647943