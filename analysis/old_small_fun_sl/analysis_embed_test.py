from velocity_plot import velocity_plot as vpl
import pandas as pd

##### load data
gene_list=None
raw_data_path = "/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/raw_data/denGyr_full.csv"
# raw_data_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Melanoma/data/cellDancer/data/input_data.csv'
# load_raw_data = pd.read_csv(raw_data_path, names=[
                            # 'gene_list', 'u0', 's0', "clusters", 'cellID', 'embedding1', 'embedding2'])
detail_result_path = '/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/velocyto/neuro/detailcsv/cost_v1_all_gene/detail_e301.csv'
n_neighbors=200
step=(60,60)

load_cellDancer = pd.read_csv(detail_result_path)
load_raw_data = pd.read_csv(raw_data_path, names=[
                    'gene_list', 'u0', 's0', "clusters", 'cellID', 'embedding1', 'embedding2'])
##### load data-end

vpl.velocity_cell_map(load_raw_data,load_cellDancer, n_neighbors=n_neighbors,step=step,save_path=None, gene_list=None, custom_xlim=None)
vpl.velocity_cell_map_curve(load_raw_data,load_cellDancer, n_neighbors=n_neighbors,step=step,save_path=None, gene_list=None, custom_xlim=None)