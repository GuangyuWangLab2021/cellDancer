
import numpy as np
import pandas as pd
import matplotlib.pyplot as pd

# embedding
embedding = np.loadtxt(open("velocyto/vlm_variables/vlm_embedding.csv", "rb"), delimiter=",")

# embedding.shape # (18140, 2)
# plt.scatter(embedding[:,0],embedding[:,1])


detail = pd.read_csv("/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/20220124_costv1_sampled_genes/detail_velocity_plot/detail_e200.csv")


np.corrcoef(detail[detail.gene_name=='Nnat'].alpha, detail[detail.gene_name=='Nnat'].u0) # 0.5
np.corrcoef(detail[detail.gene_name=='Nnat'].beta, detail[detail.gene_name=='Nnat'].s0) # -0.82 sum = -0.32

np.corrcoef(detail[detail.gene_name=='Ntrk2'].alpha, detail[detail.gene_name=='Ntrk2'].u0) # 0.94224016
np.corrcoef(detail[detail.gene_name=='Ntrk2'].beta, detail[detail.gene_name=='Ntrk2'].s0) # 0.21682215 sum = 1.15824016

np.corrcoef(detail[detail.gene_name=='Gnao1'].alpha, detail[detail.gene_name=='Gnao1'].u0) # 0.75263632
np.corrcoef(detail[detail.gene_name=='Gnao1'].beta, detail[detail.gene_name=='Gnao1'].s0) # -0.74187894 sum = 0.01

np.corrcoef(detail[detail.gene_name=='Cpe'].alpha, detail[detail.gene_name=='Cpe'].u0) # 0.99524385
np.corrcoef(detail[detail.gene_name=='Cpe'].beta, detail[detail.gene_name=='Cpe'].s0) # 0.6725009 sum = 1.66

np.corrcoef(detail[detail.gene_name=='Ank'].alpha, detail[detail.gene_name=='Ank'].u0) # 0.44617186
np.corrcoef(detail[detail.gene_name=='Ank'].beta, detail[detail.gene_name=='Ank'].s0) # -0.61965973 sum = -0.17348787

#sperman

plt.scatter(embedding[:,0],embedding[:,1],c = detail[detail.gene_name=='Ank'].alpha,s=0.5)
plt.scatter(embedding[:,0],embedding[:,1],c = detail[detail.gene_name=='Ank'].u0,s=0.5)
plt.scatter(embedding[:,0],embedding[:,1],c = detail[detail.gene_name=='Ank'].beta,s=0.5)
plt.scatter(embedding[:,0],embedding[:,1],c = detail[detail.gene_name=='Ank'].s0,s=0.5)

