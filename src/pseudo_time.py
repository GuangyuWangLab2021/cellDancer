import os
import sys
import time
import importlib

if "diffusion" in sys.modules:
    importlib.reload(sys.modules["diffusion"])

from diffusion import *
from cellDancer.get_embedding import get_embedding
    
def load_velocity(raw_data_file, detail_result_path, n_neighbors, step):
    load_raw_data = pd.read_csv(raw_data_file)
    detail_files = glob.iglob(os.path.join(detail_result_path, '*detail*.csv'))
    lcd = list()
    for f in detail_files:
        load_cellDancer_temp = pd.read_csv(f)
        load_cellDancer_temp.rename(columns = {'Unnamed: 0':'cellIndex'}, inplace = True)
        load_cellDancer_temp = load_cellDancer_temp.sort_values(by = ['gene_name', 'cellIndex'], ascending = [True, True])
        lcd.append(load_cellDancer_temp)
    load_cellDancer = pd.concat(lcd)
    
    gene_choice=list(set(load_cellDancer.gene_name))
    embedding, sampling_ixs, velocity_embedding = get_embedding(
        load_raw_data=load_raw_data,
        load_cellDancer=load_cellDancer,
        gene_list=gene_choice,
        mode="gene",
        n_neighbors=n_neighbors,
        step=step)
    plot_velocity(embedding[sampling_ixs], velocity_embedding)
    
    return load_cellDancer, embedding, sampling_ixs, velocity_embedding

def pseudo_time(embedding, velocity_embedding, sampling_ixs, downsample_step, grid, dt, t_total,
        n_repeats, output_path):

    start_time = time.time()

    cell_embedding, normalized_embedding = embedding_normalization(
        embedding[sampling_ixs], embedding, mode='minmax', NORM_ALL_CELLS=True)

    velocity = velocity_normalization(velocity_embedding, mode='max')

    __ = generate_grid(cell_embedding, normalized_embedding, velocity, steps=grid)
    vel_mesh = __[0] 
    grid_density = __[1]
    cell_grid_idx = __[2] 
    cell_grid_coor = __[3]
    all_grid_idx = __[4] 
    all_grid_coor = __[5]
    
    plot_mesh_velocity(vel_mesh, grid_density)
    
    paths=run_diffusion(cell_embedding, 
                        vel_mesh, 
                        grid_density, 
                        dt=dt, 
                        t_total=t_total,
                        eps=1e-5, 
                        off_cell_init=False, 
                        n_repeats = n_repeats, 
                        n_jobs = mp.cpu_count()-1)
    
    newPaths = truncate_end_state_stuttering(paths, cell_embedding, PLOT=False)
    traj_displacement = np.array([compute_trajectory_displacement(ipath) for ipath in newPaths])

    # sorted from long to short
    order = np.argsort(traj_displacement)[::-1]
    sorted_traj = newPaths[order]
    traj_displacement=traj_displacement[order]


    path_clusters = dict()
    cell_clusters = dict()
    path_clusters, cell_clusters = path_clustering(
        path_clusters, 
        cell_clusters, 
        sorted_traj, 
        similarity_cutoff=0.1, 
        similarity_threshold=0, 
        nkeep=-1)

    # This step could cause dropping of number of path clusters.
    cell_fate = cell_clustering_tuning(cell_embedding, cell_clusters)
    clusters = np.unique(cell_fate)
    #path_clusters = [path_clusters[i] for i in clusters]
    
    outname = 'pseudo_time_neuro_combined'+ \
        '__grid' + str(grid[0])+'x'+str(grid[1])+ \
        '__dt' + str(dt)+ \
        '__ttotal' + str(t_total)+ \
        '__nrepeats' + str(n_repeats) + \
        '.csv'

    all_cell_time, all_cell_fate= compute_all_cell_time(
        normalized_embedding, 
        cell_embedding, 
        path_clusters, 
        cell_fate,
        vel_mesh, 
        cell_grid_idx=cell_grid_idx, 
        grid_density=grid_density, 
        sampling_ixs=sampling_ixs, 
        step=downsample_step,
        dt=dt, 
        t_total=t_total, 
        n_repeats = n_repeats, 
        n_jobs=mp.cpu_count()-1, 
        outfile=os.path.join(output_path, outname))
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return all_cell_time
