#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import random
import multiprocessing as mp
import itertools
import collections

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy import interpolate

import networkx as nx
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap

def embedding_normalization(cell_embedding, embedding=None, mode="minmax", NORM_ALL_CELLS=False):
    '''
    Normalize by the maximum absolute value.
    
    Parameters
    ----------
    embedding: 2D numpy array (n_cells, 2)
    mode: string
          'maxabs', "minmax"
    maxabs is meant for sparse data and/or centered at 0. 
    Note in this program (ML velocity), it is pretty safe to do maxabs normalization
    since the data are free of extreme outliers.
     
    '''
    if mode in ['max', 'maximum', 'maxabs']:
        transformer = preprocessing.MaxAbsScaler().fit(cell_embedding)
    elif mode in ['minmax']:
        transformer = preprocessing.MinMaxScaler().fit(cell_embedding)
    em = transformer.transform(cell_embedding)
    if NORM_ALL_CELLS:
        try:
            em_all = transformer.transform(embedding)
        except ValueError:
            print("ERROR! Missing embedding for all cells.")
            raise
        return em, em_all
    else:
        return em
    
def velocity_normalization(downsampled_vel, all_vel=None, mode="max", NORM_ALL_CELLS=False):
    '''
    Normalize by the maximum absolute value in the downsampled_vel.
    
    Parameters
    ----------
    downsampled_vel: 2D numpy array (n_cells, 2)
    mode: 'maxabs'
    
    maxabs is meant for sparse data and/or centered at 0. 
    
    Note in this program, it is pretty safe to do maxabs normalization
    since the data are free of extreme outliers.
     
    '''
    dummy = 1
    # add v_prime to vel of each cell without changing their directions.
    v_mag = np.linalg.norm(downsampled_vel, axis=1)

    # for 0 velocity cell, nothing changed.
    v_mag[np.where(v_mag==0)]=dummy
    v_prime = np.std(v_mag)
    downsampled_vel = downsampled_vel*(v_prime/v_mag+1)[:,None]

    if mode in ['max', 'maximum', 'maxabs']:
        transformer = preprocessing.MaxAbsScaler().fit(downsampled_vel)
    em = transformer.transform(downsampled_vel)
    if NORM_ALL_CELLS:
        em_all = transformer.transform(all_vel)
        return em, em_all
    else:
        return em
    

def discretize(coordinate, xmin, xmax, steps, capping=False):
    '''
    '''
    grid_size = np.array(xmax) - np.array(xmin)
    grid_size = grid_size / np.array(steps)

    grid_idx = np.int64(np.floor((coordinate-xmin)/grid_size))
    
    if capping:
        grid_idx = np.where(grid_idx > steps, steps, grid_idx)
        grid_idx = np.where(grid_idx <0, 0, grid_idx)
    
    grid_coor = xmin + grid_size * (grid_idx+0.5)
    return grid_idx, grid_coor 


def generate_grid(cell_embedding, embedding, velocity_embedding, steps):
    '''
    the original embedding is used to generate the density
    '''

    def array_to_tuple(arr):
        try:
            return tuple(array_to_tuple(i) for i in arr)
        except TypeError:
            return arr

    xmin = np.min(cell_embedding, axis=0)
    xmax = np.max(cell_embedding, axis=0)
    steps = np.array(steps, dtype=int)

    cell_grid_idx, cell_grid_coor = discretize(cell_embedding, xmin=xmin, xmax=xmax, steps=steps)
    mesh = np.zeros(np.append(steps+1,len(steps)))

    # The actual steps need to allow a leeway +1 in each dimension.
    cnt = np.zeros(steps+1)
    
    for index in range(cell_grid_idx.shape[0]):
        grid_index = cell_grid_idx[index]
        if np.any(grid_index > steps) or np.any(grid_index < 0):
            continue
        grid_index = array_to_tuple(grid_index)
        mesh[grid_index] += velocity_embedding[index]
        cnt[grid_index] += 1
    cnt = cnt[:,:,None]
    mesh = np.divide(mesh, cnt, out=np.zeros_like(mesh), where=cnt!=0)
    
    #sanity check
    #fig, ax = plt.subplots(figsize=(6, 6))
    #plt.scatter(cell_embedding[:,0],cell_embedding[:,1], c='grey', alpha = 0.3)
    #plt.scatter(cell_grid_coor[:,0], cell_grid_coor[:,1], marker='s',
    #        color='none', edgecolor='blue', alpha=0.3)
    #plt.xlim([-0.1,1.1])
    #plt.ylim([-0.1,1.1])
    #plt.show()
    
    # The actual steps need to allow a leeway +1 in each dimension.
    density = np.zeros(steps+1)
    all_grid_idx, all_grid_coor = discretize(embedding, xmin=xmin, xmax=xmax, steps=steps)
    for index in range(all_grid_idx.shape[0]):
        all_grid_index = all_grid_idx[index]
        
        # outside density is not needed.
        if np.any(all_grid_index > steps) or np.any(all_grid_index < 0):
            continue
        all_grid_index = array_to_tuple(all_grid_index)
        density[all_grid_index] += 1
    return mesh, density, cell_grid_idx, cell_grid_coor, all_grid_idx, all_grid_coor


def plot_velocity(embedding, velocity_embedding):
# support 2D only
    fig, ax = plt.subplots(figsize=(6,6))
    plt.quiver(embedding[:, 0],embedding[:, 1],
              velocity_embedding[:,0], velocity_embedding[:,1], 
              color='Blue')
    plt.show()

def plot_mesh_velocity(mesh, grid_density):
# support 2D only
    x=list()
    y=list()
    vx=list()
    vy=list()
    for i in range(mesh.shape[0]):
        for j in range(mesh.shape[1]):
            x.append(i)
            y.append(j)
            vx.append(mesh[i,j][0])
            vy.append(mesh[i,j][1])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(x,y,vx,vy,color='red',scale = 10)
    plt.imshow(grid_density.T, interpolation=None, origin='lower',cmap="Greys")
    plt.show()

def velocity_add_random(velocity, theta):
    '''
    Rotate the velocity according to a randomized kicks on the perpendicular direction.
    The direction is determined by the sign of a random number. 
    The magnitude of the perpendicular kick is determined by the random number 
    from a normal distribution N(0, theta).
    Magnitude of the velocity is kept the same to conserve energy (temperature) of the system.
    
    Parameters
    ----------
    velocity
        velocity of the grid
    theta
        the angular range that the noise could be affecting the direction of the velocity
        
    WARNING
        at a rare chance, the rotation angle (magnitude) could be much larger than theta.
        
    Return
    ------
    Adjusted velocity for the interested cell
        
    '''
    r = np.random.normal(0, theta, 1)
    cosine = np.cos(r)[0]
    sine = np.sin(r)[0]
    
    # Rotation matrix
    R = np.array([[cosine, sine],[-sine, cosine]])
    velocity = np.dot(velocity, R)
    return velocity

def velocity_rotation(velocity, theta):
    '''
    Rotate the velocity clockwise by angle theta
    
    Parameters
    ----------
    velocity
        velocity of the grid
    theta
        the angular range that the noise could be affecting the direction of the velocity
        
    Return
    ------
    Adjusted velocity for the interested cell
        
    '''
    cosine = np.cos(theta)
    sine = np.sin(theta)
    
    # Rotation matrix
    R = np.array([[cosine, sine],[-sine, cosine]])
    velocity = np.dot(velocity, R)
    return velocity


def diffusion_off_grid_wallbound(
        cell_embedding, 
        vel, 
        init, 
        grid_density,
        dt=0.001, 
        t_total=10000, 
        eps = 1e-5):
    
    '''
    Simulate the diffusion of a cell in the velocity field (off grid), the cell's velocity will turn 90 degrees
    if it hits the boundary the next timestep.

    The diffusion is stopped by any of the criteria:
    1. reach t_total
    2. the magnitude of the velocity is less than eps.
    3. the cell goes to places where the cell density = 0 even after turning.
    4. the cell is out of the simulation box

    Parameters
    ----------
    
    cell_embedding: numpy ndarray (n_cells x n_dims)
        embedding coordinate for all the cells (downsampled)

    vel: numpy ndarray (n_grids x n_dims)
        pre-assigned velocity of each grid

    init: numpy ndarray (n_cells x n_dims)
        The initial position (cell_embedding)

    dt: float 
        Step size of each integration time step

    t_total: int
        Total number of time steps

    grid_density: numpy ndarray (n_grids x n_dims)
        Density of cells.

    eps 
        Criterion to stop a trajectory before t_total (v_net < eps)

    
    Return
    ------
        a numpy ndarray of coordinates in the trajectory, shape:
        (real_n_time_steps, n_dims)
    '''
    
    # TODO: recommmend values of dt and timestep
    THETA = np.pi/6
    
    XMIN = np.min(cell_embedding, axis=0)
    XMAX = np.max(cell_embedding, axis=0)
    STEPS=(vel.shape[0]-1,vel.shape[1]-1)

    # lower 5% nonzero density set to 0..
    MAX_IGNORED_DENSITY= np.percentile(grid_density[grid_density>0],5)
    
    def no_cells_around(xcur, xcur_d, vcur):
        xnxt = xcur + vcur*dt
        xnxt_d, dummy = discretize(xnxt, xmin=XMIN, xmax=XMAX, steps=STEPS)
        try:
            density = grid_density[xnxt_d[0], xnxt_d[1]]
        except IndexError:
            return True
        return density <= MAX_IGNORED_DENSITY
   
    x0 = init
    x0_d, dummy = discretize(x0, xmin=XMIN, xmax=XMAX, steps=STEPS)
    v0 = vel[x0_d[0],x0_d[1]]
    v0 = velocity_add_random(v0, THETA)
    trajectory = [x0]
    
    for i in range(int(t_total)):
    
        if np.linalg.norm(v0) < eps:
            #print("Velocity is too small")
            return np.array(trajectory)
        if no_cells_around(x0, x0_d, v0):
            v0_cc = velocity_rotation(v0, np.pi/2)
            v0_c = velocity_rotation(v0, -np.pi/2)
            # nowhere to go but null
            CC = no_cells_around(x0, x0_d, v0_cc) 
            C = no_cells_around(x0, x0_d, v0_c)
            if CC and C:
                return np.array(trajectory)
            elif not C:
                v0 = v0_c
            else:
                v0 = v0_cc
                
        else:
            x = x0 + v0*dt
            x_d, dummy = discretize(x, xmin=XMIN, xmax=XMAX, steps=STEPS)
            try:
                v = vel[x_d[0],x_d[1]]
                density = grid_density[x_d[0],x_d[1]]
                v = velocity_add_random(v, THETA)
            except IndexError:
                break
            
            trajectory.append(x)
            x0 = x
            v0 = v

    return np.array(trajectory)


def diffusion_on_grid_wallbound(
        cell_embedding, 
        vel, 
        init, 
        grid_density,
        dt=0.001, 
        t_total=10000, 
        eps = 1e-5):
    
    '''
    same as diffusion_off_grid_wallbound, however, it returns the coordinates
    of the grid traversed by the cell, instead of the position of the cell.

    The diffusion is stopped by any of the criteria:
    1. reach t_total
    2. the magnitude of the velocity is less than eps.
    3. the cell goes to places where the cell density = 0 even after turning.
    4. the cell is out of the simulation box

    Parameters
    ----------
    
    cell_embedding: numpy ndarray (n_cells x n_dims)
        embedding coordinate for all the cells (downsampled)

    vel: numpy ndarray (n_grids x n_dims)
        pre-assigned velocity of each grid

    init: numpy ndarray (n_cells x n_dims)
        The initial position (cell_embedding)

    dt: float 
        Step size of each integration time step

    t_total: int
        Total number of time steps

    grid_density: numpy ndarray (n_grids x n_dims)
        Density of cells.

    eps 
        Criterion to stop a trajectory before t_total (v_net < eps)

    
    Return
    ------
        a numpy ndarray of coordinates in the trajectory, shape:
        (real_n_time_steps, n_dims)
    '''
    
    THETA = np.pi/6
    
    XMIN = np.min(cell_embedding, axis=0)
    XMAX = np.max(cell_embedding, axis=0)
    STEPS=(vel.shape[0]-1,vel.shape[1]-1)
    
    def no_cells_around(xcur, xcur_d, vcur):
        xnxt = xcur + vcur*dt
        xnxt_d, dummy = discretize(xnxt, xmin=XMIN, xmax=XMAX, steps=STEPS)
        try:
            density = grid_density[xnxt_d[0], xnxt_d[1]]
        except IndexError:
            return True
        return density < 1
   
    x0 = init
    x0_d, x0_d_coor = discretize(x0, xmin=XMIN, xmax=XMAX, steps=STEPS)
    v0 = vel[x0_d[0],x0_d[1]]
    v0 = velocity_add_random(v0, THETA)
    trajectory = [x0_d_coor]
    
    for i in range(int(t_total)):
    
        if np.linalg.norm(v0) < eps:
            print("Velocity is too small")
            return np.array(trajectory)
        if no_cells_around(x0_d_coor, x0_d, v0):
            v0_cc = velocity_rotation(v0, np.pi/2)
            v0_c = velocity_rotation(v0, -np.pi/2)
            # nowhere to go but null
            CC = no_cells_around(x0_d_coor, x0_d, v0_cc) 
            C = no_cells_around(x0_d_coor, x0_d, v0_c)
            if CC and C:
                return np.array(trajectory)
            elif not C:
                v0 = v0_c
            else:
                v0 = v0_cc
                
        else:
            x = x0_d_coor + v0*dt
            x_d, x_d_coor = discretize(x, xmin=XMIN, xmax=XMAX, steps=STEPS)
            try:
                v = vel[x_d[0],x_d[1]]
                v = velocity_add_random(v, THETA)
            except IndexError:
                break
            
            trajectory.append(x_d_coor)
            x0 = x_d
            x0_d_coor = x_d_coor
            v0 = v

    return np.array(trajectory)


def run_diffusion(cell_embedding, vel, grid_density, dt, t_total = 10000, eps = 1e-5, 
                  off_cell_init = False, init_cell = [], n_repeats = 10, n_jobs = 8):    
    '''
    Simulation of diffusion of a cell in the velocity field (on grid), 
    the cell's velocity will turn 90 degrees if it hits the boundary the next timestep.
    Embarrassingly parallel (process) are employed.
    
    Parameters
    ----------
    
    cell_embedding: numpy.ndarray (ncells, 2)
        embedding coordinate for all the cells (downsampled)
        
    vel: numpy.ndarray (ngrid, ngrid, 2)
        pre-assigned velocity of each grid
    
    dt: float
        Step size of each integration time step
    
    t_total: int
        Total number of time steps
    
    eps: float
        Criterion to stop a trajectory before t_total (v_net < eps)
    
    off_cell_init: Boolean
        Whether to spawn initial coordinates from the neighbouring space around a cell
        
    init_cell: list
        List of initial cell indices. If empty list, use all cell indices in the given cell_embedding.
    
    n_repeats: init
        Number of repeats (either on or off the cells)
    
    n_jobs: int
        Number of threads
    
    Return
    ------
        a numpy array of trajectorys,  shape: (num_trajs, *n_time_steps, 2)
    '''
        
    if n_jobs >= mp.cpu_count():
        n_jobs = max(8, mp.cpu_count()-1)
        print("Reducing n_jobs to", n_jobs, "for optimal performance.")

    TASKS = list()
    # Setting up the TASKS
    n_cells = cell_embedding.shape[0]
    
    if not init_cell:
        init_cell = list(range(n_cells))

    embedding_range = cell_embedding.max(axis=0) - cell_embedding.min(axis=0)
    steps = np.array([vel.shape[0], vel.shape[1]])
    grid_size = embedding_range/steps
    
    for i in init_cell:
        for j in range(n_repeats):
            if off_cell_init:
                init_position = cell_embedding[i] + grid_size * np.random.uniform(-0.5,0.5,2)
            else:
                init_position = cell_embedding[i]
            TASKS.append((cell_embedding, vel, init_position, grid_density, dt, t_total))
    
    with mp.Pool(n_jobs) as pool:
        paths = pool.starmap(diffusion_off_grid_wallbound, TASKS)
    return np.array(paths, dtype=object)


def compute_trajectory_displacement(traj):
    traj = np.array(traj)
    return np.linalg.norm(traj[-1,:] - traj[0,:])

def compute_trajectory_length(traj1):
    temp = traj1[:-1,:] - traj1[1:,:]
    length = np.sum(np.sqrt(np.sum(temp**2, axis=1)))
    return length


def compute_trajectory_similarity(traj1, traj2, numPicks=10):
    '''
    Computes the similarity between two curves based on average distance of a selection of closest pairs

    Input: 
    - numpy arrays (nsteps1, 2), (nsteps2, 2); nsteps1 >= nsteps2 
    - numPicks: number of representative points on the shorter curve traj2
    Return: a float number
    '''
    # assert traj1 longer than traj2
    # empty traj2
    if not traj2.size:
        print("empty trajectory here!")
        raise 
    # pick N points evenly from traj2
    idx = np.round(np.linspace(0, len(traj2) - 1, numPicks)).astype(int)
    
    # in the case the shorter trajectory is less than numPicks timesteps
    idx = np.unique(idx)
    
    temp = traj1 - traj2[idx, None]
    A = np.min(np.sum(temp**2, axis=2), axis=1)
    
    return np.mean(A**(1/2))


def compute_similarity_matrix(traj, numPicks=10):
    import itertools
    traj_length = np.array([compute_trajectory_length(np.array(i)) for i in traj])
    traj_order = np.argsort(traj_length) # 1d array
    
    ncells = len(traj_order)
    simMat = np.zeros((ncells,ncells))
    for i,j in itertools.combinations(traj_order, 2):
            # length of traj[i] <= traj[j]
            simMat[i,j] = compute_trajectory_similarity(np.array(traj[j]), np.array(traj[i]), numPicks)
            simMat[j,i] = simMat[i,j]
    return simMat


def truncate_end_state_stuttering(paths, cell_embedding, PLOT=True):
    newPaths = [ipath[:np.int32(np.where(np.linalg.norm(ipath-ipath[-1], axis=1) < 1e-3)[0][0])] for ipath in paths]
    newPaths = [i for i in newPaths if len(i) > 10]
    
    if PLOT:
        # sanity check; end points of all the trajectories
        endState = np.array([(i[-1,0], i[-1,1]) for i in newPaths])
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.scatter(cell_embedding[:,0], cell_embedding[:,1], c='silver', s=10, alpha=0.3)
        plt.scatter(endState[:,0], endState[:,1], s=20, c='orange', alpha=1)
        plt.axis("off")
        plt.title("end states of all trajectories")
        plt.show()
    
    return np.array(newPaths, dtype=object)


def path_clustering(path_clusters, cell_clusters, paths, similarity_cutoff, similarity_threshold, nkeep=10):
    '''
    a clustering method to find representative paths.
    
    Parameters
    ----------
    
    paths: np.ndarray (N, ntimesteps, 2)
        N paths, sorted by their |displacement|, each trajectory is a (ntimestep, 2) array
        
    similarity_threshold: float
        first round clustering to keep trajectories which are within this threshold
    
    After each iteration, a number of trajectories are popped in the traj list 
    returns a list of clusters
    
    Return
    ------
    path_clusters: a dictionary of np.ndarray (ntimesteps, 2)
    cell_clusters: a dictionary of np.ndarray (2, )
    
    '''
    #print("Currently ", len(path_clusters), "clusters:\t", len(paths), "paths to be categorized")
    clusterID = len(path_clusters)
    if not paths.size:
        return path_clusters, cell_clusters
    
    longest = paths[0]
    similarity = np.array([compute_trajectory_similarity(np.array(longest), np.array(ipath), 10) 
                           for ipath in paths])
    
    #print(similarity.shape)
    #print(paths.shape)
    sel = (similarity < similarity_cutoff)
    sel_keep = (similarity_threshold <= similarity)
    cluster = paths[sel & sel_keep][:nkeep]
    if len(cluster) == 0:
        #print("this cluster has no traj to keep")
        cluster = paths[0,None]
    elif not np.array_equal(paths[0], cluster[0]):
        #print("concat", cluster[0].shape, paths[0].shape)
        cluster = np.append(cluster, paths[0,None])
    #print(cluster.shape)
    path_clusters[clusterID] = cluster
    cell_clusters[clusterID] = [ipath[0] for ipath in paths[sel]]
    
    paths = paths[~sel]
    return path_clustering(path_clusters, cell_clusters, paths, similarity_cutoff, similarity_threshold, nkeep)

    
def cell_clustering_tuning(embedding, cell_clusters, n_neighbors=20):
    '''
    Parameters
    ----------
    embedding: numpy ndarray (ncells, 2)
    
    cell_clusters: dictionary of length n_clusters
        A dictionary of starting cell positions (cluster_size, 2) in each cluster.

    n_neighbors: float
        
    Return
    ------
    A numpy array (size: ncells) recording fate of each cell in the order of
    cellIDs.
    '''

    # path time to cell time.
    n_clusters = len(cell_clusters)
    n_cells = len(embedding)

    # initialization
    # each cell has a chance to go through the n_clusters fates
    # according to its match with cells in each cluster
    # defined by [p0, p1, ..., p{n_cluster-1}]

    cell_fate = np.zeros((n_cells, n_clusters))
    cluster_index = 0
    clusterIDs = list()
    for cluster, cell_embeddings in cell_clusters.items():
        temp = cell_embeddings - embedding[:,None]

        # tuple [0] --> cell indices
        # tuple [1] --> traj indices in the cluster
        for i in np.where(np.sum(temp**2, axis = 2) == 0)[0]:
            cell_fate[i][cluster_index] += 1
        cluster_index += 1
        clusterIDs.append(cluster)
    cell_fate_major = np.argmax(cell_fate, axis=1)
    print(cell_fate_major)

    # mapping back to clusterIDs
    # cluster_index is the list index of the clusterIDs
    cell_fate_major = np.array([clusterIDs[i] for i in cell_fate_major],
            dtype=int)

    print(cell_fate_major)

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(embedding)
    A = neigh.kneighbors_graph(embedding)
    B = A.toarray()
    cell_fate_tuned = np.array([collections.Counter(B[i][B[i]!=0]*cell_fate_major[B[i]!=0]).most_common()[0][0] 
                           for i in range(n_cells)], dtype=int)

    return np.array(cell_fate_tuned)


def cell_time_projection_cluster(embedding, rep_path, cluster, cell_fate):
    '''
    Parameters
    ----------
    embedding: numpy ndarray
    
    rep_path: numpy ndarray
        used a reference for cell time projection
    
    clusters: int
        which cluster of cells
        
    cell_fate: numpy 1d array
        cluster number for each cell
        
    Return
    ------
    A dictionary of cell time {cellID: time}
    '''
    cell_time= list()
    #print("Path ", cluster)
    cells = embedding[cell_fate == cluster]
    #print(cells)
    cell_index = np.where(cell_fate == cluster)[0]
    dist = cells[:,None] - rep_path
    cell_time=np.argmin(np.sum(dist**2, axis = 2), axis = 1)
    cell_time_per_cluster = {A: B for A, B in zip(cell_index, cell_time)}
    return cell_time_per_cluster


def plot_cell_clusters(cell_fate, cell_embedding):
    clusters = np.unique(cell_fate)
    n_clusters = len(clusters)

    cmap = ListedColormap(sns.color_palette("colorblind", n_colors = n_clusters))
    fig, ax1 = plt.subplots(figsize=(6, 6))
    img1=ax1.scatter(cell_embedding[:,0], cell_embedding[:,1], c=cell_fate,
            s=1, alpha=1, cmap=cmap)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title("cell fate: majority votes")
    ax1.axis("off")

    bounds = np.linspace(0, n_clusters, n_clusters+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax3 = fig.add_axes([0.9, 0.3, 0.02, 0.3])
    cb = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, spacing='proportional', boundaries=bounds, norm=norm, format='%1i')
    labels = ["cluster "+str(i) for i in range(n_clusters)]

    cb.ax.get_yaxis().set_ticks([])
    for i, label in enumerate(labels):
        cb.ax.text(4.5, i + 0.5 , label, ha='center', va='center')
    plt.show()


def plot_path_clusters(path_clusters, clusters, cell_embedding):    
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(cell_embedding[:,0], cell_embedding[:,1], c='silver', s=10, alpha = 0.3)
    n_clusters = len(clusters)
    
    cmap = ListedColormap(sns.color_palette("colorblind", n_colors = n_clusters))
    colormaps = [ListedColormap(sns.light_palette(cmap.colors[i],
        n_colors=100)) for i in range(n_clusters)]

    # find the nearest cell (terminal cell) to the end point
    neigh = NearestNeighbors(n_neighbors=1, n_jobs=mp.cpu_count()-1)
    neigh.fit(cell_embedding)

    cluster_cnt = 0
    for cluster in clusters:
        cl = colormaps[cluster_cnt]
        leading_path=path_clusters[cluster][0]
        terminal_cell=leading_path[-1]
        A = neigh.kneighbors_graph(np.array([terminal_cell]))
        B = A.toarray()
        terminal_cell = np.matmul(B, cell_embedding)

        plt.text(leading_path[-1,0], leading_path[-1,1], "cluster"+str(cluster), fontsize=12)
        plt.scatter(leading_path[:,0], leading_path[:,1], s=5,
                c=range(len(leading_path)), cmap=colormaps[cluster_cnt])
        plt.scatter(terminal_cell[:,0], terminal_cell[:,1], s=30, color=cmap.colors[cluster_cnt])
        cluster_cnt += 1
    plt.axis('off')
    plt.show()
    

def closest_distance_between_two_paths(path1, path2, cell_embedding):
    '''
    returns the closest distance and the closest cells.
    '''
   
    if path1.size and path2.size:
        temp = path1 - path2[:, None]
        A = np.sum(temp**2, axis=2)
        pair = np.unravel_index(np.argmin(A), A.shape)
        print("The closest distance is ", np.sqrt(A[pair]))
        print("Between dot", pair[1], " from refPath1 and dot", pair[0], "from refPath2.")
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.scatter(cell_embedding[:,0], cell_embedding[:,1], alpha = 0.3)
        plt.scatter(path1[:,0], path1[:,1], c=range(len(path1)), s=5)
        plt.text(path1[-1,0], path1[-1,1], "refPath"+str(1), fontsize=12)
        plt.text(path2[-1,0], path2[-1,1], "refPath"+str(2), fontsize=12)        
        plt.scatter(path2[:,0], path2[:,1], c=range(len(path2)), s=5)
        plt.show()
        return np.sqrt(A[pair]), pair[::-1]
    else:
        return np.Inf,(np.nan,np.nan)
    
    
def recursive_cell_time_assignment_intracluster(
    unresolved_cell_time_cluster, 
    cluster, 
    sorted_refPaths, 
    cell_fate,
    cell_embedding, 
    vel, 
    cell_grid_idx, 
    grid_density, 
    dt=0.001, 
    t_total=10000, 
    n_repeats = 10, 
    n_jobs = mp.cpu_count()-1,
    MAX_ZERO_TIME_CELLS = 0.05):
    '''
    Recursive function to consolidate cell time within a cluster.
    
    Parameters
    ----------
    unresolved_cell_time_cluster: list
        a list of dictionary {cellID : time}
        cellIDs and corresponding unresolved time for a specific cluster
    
    sorted_refPaths: list
        a list of paths in the cluster ordered from longest to shortest in displacement. 
        
    cell_embedding, vel, grid_density: a set of parameters for the diffusion simulations.
    
    Return
    ------
    resolved_cell_time_cluster: list
        a list of dictionaries {cellID : time} 
    
    sorted_refPaths: list
        a list of longest trajectories used for cell time projection
    '''
    
    print("\n\n")
    print("cluster ", cluster)
    ZERO = 0
    if isinstance(MAX_ZERO_TIME_CELLS, float):
        MAX_ZERO_TIME_CELLS = int(MAX_ZERO_TIME_CELLS
                * len(unresolved_cell_time_cluster[cluster]))
    MAX_ZERO_TIME_CELLS = max(MAX_ZERO_TIME_CELLS, 10)
    print("MAX allowed zero time cells are: ", MAX_ZERO_TIME_CELLS)

    zero_time_cells = [cellid for cellid, celltime in unresolved_cell_time_cluster[cluster].items() if celltime <= ZERO]
    if len(zero_time_cells) < MAX_ZERO_TIME_CELLS:
        print("Only ", len(zero_time_cells), " Cells left. Stopping.")
        return unresolved_cell_time_cluster, sorted_refPaths
    else:
        print(len(zero_time_cells), " Cells left.")
    
    # get subsampled cell embedding
    sub_embedding = cell_embedding[zero_time_cells]

    # get subsampled density matrix
    sub_grid_density = np.zeros_like(grid_density)
    for cell in zero_time_cells:
        i = tuple(cell_grid_idx[cell])
        sub_grid_density[i] = grid_density[i]
    
    # sanity check
    cells = [_ for _ in unresolved_cell_time_cluster[cluster]]

    fig, axes = plt.subplots(nrows=1, ncols=2,
            gridspec_kw={'width_ratios':[1,1]}, figsize=(12,6))
    axes[0].scatter(cell_embedding[:,0], cell_embedding[:,1], s=5, alpha=0.1)
    axes[0].scatter(cell_embedding[cells,0],
            cell_embedding[cells,1], s=5, alpha=0.5)
    axes[0].scatter(sub_embedding[:,0], sub_embedding[:,1], s=5, c='k')
    axes[0].title.set_text("spread of the zero time cells")

    cmap = sns.cubehelix_palette(start=2, rot=0., dark=0.2, light=1, as_cmap=True)
    axes[1].imshow(grid_density.T, interpolation=None, origin='lower', cmap="Greys")
    axes[1].imshow(sub_grid_density.T, interpolation=None,
            origin='lower',cmap=cmap, alpha=0.3)
    axes[1].title.set_text("cell density for generating new trajectories")
    plt.show()
    
    # generating new trajectories for the zero-time cells
    print("Sampling new trajectories for zero-time cells in cluster ", cluster, "  ...")
    sub_traj=run_diffusion(cell_embedding, vel=vel,
            grid_density=sub_grid_density, 
            dt=dt, t_total=t_total, eps=1e-3, 
            off_cell_init=False, init_cell=zero_time_cells, 
            n_repeats = n_repeats, n_jobs = n_jobs)
    
    # finding the longest trajectory
    newPaths = truncate_end_state_stuttering(sub_traj, cell_embedding, PLOT=False)
    traj_displacement = np.array([compute_trajectory_displacement(ipath) for ipath in newPaths])
    order = np.argsort(traj_displacement)[::-1]
    sorted_traj = newPaths[order]
    traj_displacement=traj_displacement[order]

    # add the longest trajectory belonging to the zero-time cells to the ref paths.
    ref_path = sorted_traj[0]
    
    # re-assign time for zero time cells
    sub_cell_time = cell_time_projection_cluster(cell_embedding, ref_path, cluster, cell_fate)

    # consolidate 
    dist, (dotA, dotB) = closest_distance_between_two_paths(ref_path,
            sorted_refPaths[-1], cell_embedding)
    
    print("now the cell time is updated for cluster: ", cluster)
    print("shifting ref_path2 by time ", dotA-dotB)
    #print(unresolved_cell_time_cluster[cluster])

    for cell in unresolved_cell_time_cluster[cluster]:
        unresolved_cell_time_cluster[cluster][cell] = sub_cell_time[cell] if \
                cell in zero_time_cells else unresolved_cell_time_cluster[cluster][cell] - dotB + dotA

    sorted_refPaths.append(ref_path)
    
    return recursive_cell_time_assignment_intracluster(
        unresolved_cell_time_cluster, 
        cluster, 
        sorted_refPaths, 
        cell_fate,
        cell_embedding, 
        vel, 
        cell_grid_idx, 
        grid_density, 
        dt=dt, 
        t_total=t_total, 
        n_repeats = n_repeats, 
        n_jobs = n_jobs,
        MAX_ZERO_TIME_CELLS=MAX_ZERO_TIME_CELLS)
    

def cell_time_assignment_intercluster(unresolved_cell_time, cell_fate, cell_embedding, tau = 0.05):
    '''
    Consolidate cell time between clusters according to the intersection
    between cells from any two clusters.

    Assumption: No cyclic behavior between clusters. Else, return None.
    We construct a directed graph: 
    CT --> inter-cluster time gap (transfer)
    to detect cycles.

    Parameters
    ----------
    unresolved_cell_time: list
        a list of dictionary {cellID : time}
        cellIDs and corresponding unresolved time for all cells.

    cell_fate: np.ndarray
        an array of cluster numbers which each cell belongs to.

    Return
    ------
    resolved_cell_time_cluster: list
        a list of dictionaries {cellID : time}

    '''

    clusterIDs = sorted(np.unique(cell_fate))
    cutoff = overlap_crit_intra_cluster(cell_embedding, cell_fate, tau)

    CT = nx.DiGraph()
    for cluster in clusterIDs:
        CT.add_node(cluster)

    # nodes
    nodes = clusterIDs
    n_nodes = len(nodes)
    # paths
    paths = list()
    # weights
    w = list()
    
    # MAX_IGNORED_TIME_SHIFT is set to 50% of the shortest cluster.
    MAX_IGNORED_TIME_SHIFT = 0.5 * min([max(unresolved_cell_time[cluster].values()) for
        cluster in clusterIDs])
    for i,j in itertools.combinations(clusterIDs, 2):
        shiftT, overlap_cells = overlap_intercluster(cell_embedding, cell_fate, 
                unresolved_cell_time, i, j, cutoff)

        if shiftT:
            shiftT = 0 if abs(shiftT) < MAX_IGNORED_TIME_SHIFT else shiftT

            print("shift time is: ", shiftT, ".\n The overlapping cells are:\n",
                    "cell ", overlap_cells[0], " from cluster ", i, " and ", 
                    overlap_cells[1], " from cluster ", j)

            if shiftT > 0:
                CT.add_edge(i, j, weight = shiftT)
                w.append(shiftT)
                paths.append([i,j])

            if shiftT < 0:
                CT.add_edge(j, i, weight = -shiftT)
                w.append(-shiftT)
                paths.append([j,i])

    pos = nx.spring_layout(CT, k = 1)
    nx.draw(CT, 
            pos=pos, 
            with_labels = True, 
            node_size=500, 
            node_color = 'b',
            style=':',
            font_size = 18, 
            font_color = 'w')

    labels = nx.get_edge_attributes(CT,'weight')
    nx.draw_networkx_edge_labels(CT,pos,edge_labels=labels)
    plt.show()
    
    if not nx.is_forest(CT):
        print("There exists a cycle in the cluster graph.")
        print("Unable to consolidate cells times in this case.")
        return unresolved_cell_time
    
    paths = np.array(paths)
    w = np.array(w)
    pseudotime = np.array(unresolved_cell_time, dtype=object)
    flag = np.array([0]*n_nodes)
    w_cumm = np.array([0]*n_nodes)

    # for unconnected nodes
    for i in range(len(nodes)):
        node = nodes[i]
        if not node in paths.flatten():
            flag[node]=1

    # by Guangyu Wang
    node = nodes[0]
    idx = np.where(nodes == node)
    flag[idx] = 1
    while len(flag[flag==0])>0:
        for path in paths:
            if path[0]==node:
                #print("Forward: "+str(node))
                w_cumm[nodes==path[1]] = w_cumm[nodes==node]+w[np.all(paths==path, axis=1)]
                node = path[1]
                idx = np.where(nodes==node)
                flag[idx] = 1
            
            elif path[1]==node:
                #print("Backward: "+str(node))
                w_cumm[nodes==path[0]] = w_cumm[nodes==node]-w[np.all(paths==path, axis=1)]
                node=path[0]
                idx = np.where(nodes==node)
                flag[idx] = 1
            else:
                #print("Pass: "+str(node))
                pass

    # update pseudotime
    for node in range(len(w_cumm)):
        cells = pseudotime[node]
        for cell in cells:
            cells[cell] += w_cumm[node]
    return pseudotime


def plot_celltime_clusters(cell_time_per_cluster, rep_paths, embedding):
    longest_paths = [ipath[0] for ipath in rep_paths]
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(embedding[:,0],embedding[:,1], c='silver', alpha = 0.3)

    n_paths = len(longest_paths)
    cmap = ['viridis'] * n_paths

    for i in range(n_paths):
        colormap = cmap[i]
        cell_index = list(cell_time_per_cluster[i].keys())
        cell_time = list(cell_time_per_cluster[i].values())
        cells = embedding[cell_index]
        plt.scatter(cells[:,0], cells[:,1], c=cell_time, s=20, cmap = colormap)
        #plt.colorbar()
        #plt.text(pos[0], pos[1], "cluster"+str(i+1), fontsize=12)
        ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()


# combine cell time from clusters
def combine_clusters(cell_time_per_cluster):
    cell_time = dict()
    for d in cell_time_per_cluster:
        for k, v in d.items():
            cell_time[k] = v
    return cell_time


def interpolate_all_cell_time(cell_time, all_cell_embedding, sampling_ixs, step):
    x = all_cell_embedding[sampling_ixs,0]
    y = all_cell_embedding[sampling_ixs,1]

    xx = np.linspace(min(x), max(x), step[0]+1)
    yy = np.linspace(min(y), max(y), step[1]+1)
    xx, yy = np.meshgrid(xx, yy)
    
    points = np.transpose(np.vstack((x, y)))
    interp = interpolate.griddata(points, cell_time, (xx, yy), method='nearest')
    all_cell_time = list()
    for cell_coord in all_cell_embedding:
        gd = discretize(cell_coord, xmin=(min(x), min(y)), xmax=(max(x),max(y)), steps=step, capping = True)[0]
        all_cell_time.append(interp[gd[1], gd[0]])

    # drop the top 5 percentile
    all_cell_time = np.array(all_cell_time)
    all_cell_time[all_cell_time>np.quantile(all_cell_time, 0.95)]=np.quantile(all_cell_time, 0.95)
    
    # smoothing the data using nearest neighbours
    neigh = NearestNeighbors(n_neighbors=20, radius=1, n_jobs=mp.cpu_count()-1)
    neigh.fit(all_cell_embedding)
    A = neigh.kneighbors_graph(all_cell_embedding)
    B = A.toarray()

    all_cell_time_smooth = [np.mean(all_cell_time[B[i]!=0]) for i in
            range(len(all_cell_time))]
    all_cell_time_smooth -= np.min(all_cell_time_smooth)
    all_cell_time_smooth = all_cell_time_smooth/np.max(all_cell_time_smooth)

    print("\n\n\nPlotting estimated pseudotime for all cells ...")
    fig, ax = plt.subplots(figsize=(6,6))
    im = plt.scatter(all_cell_embedding[:,0], all_cell_embedding[:,1],
            c=all_cell_time_smooth, alpha = 1, s = 1)

    plt.xlim([min(x), max(x)])
    plt.ylim([min(y), max(y)])
    #plt.scatter(x, y, c=f, alpha = 0.2, s = 5, edgecolors='black')
    cax = plt.colorbar(im,fraction=0.03, pad=0., location='bottom')
    cax.set_label('normalized pseudotime')
    #plt.title('pseudotime: gastrulation')
    plt.axis('off')
    plt.show()
    
    return all_cell_time_smooth
    
    
def export_everything(load_cellDancer, cell_time, filename):
    gene_names = load_cellDancer['gene_name'].drop_duplicates().to_list()
    cell_number = load_cellDancer[load_cellDancer['gene_name']==gene_names[0]].shape[0]

    if len(load_cellDancer) == len(gene_names) * len(cell_time):
        load_cellDancer['time'] = np.tile(cell_time,len(gene_names))
        load_cellDancer.to_csv(filename, index=None)


def export_cell_time(cell_time, cell_fate, sampling_ixs, filename): 
    sample = np.array([True if i in sampling_ixs else False for i in
        range(len(cell_fate))], dtype=bool)
    data = np.vstack((range(len(cell_fate)), cell_fate, cell_time, sample)).T
    df = pd.DataFrame(data, columns = ['cellindex', 'traj_cluster', 'pseudotime',
    'downsampled'])
    df = df.astype({"cellindex": int, "traj_cluster": int, "pseudotime": float,
        "downsampled": bool})
    df.to_csv(filename, index=False)
        
def overlap_crit_intra_cluster(cell_embedding, cell_fate, quant):
    cutoff = list()
    for cluster_ID in np.unique(cell_fate):
        cell_cluster = cell_embedding[cell_fate == cluster_ID]
        temp1 = cell_cluster - cell_cluster[:,None]
        temp2 = np.linalg.norm(temp1, axis=-1)
        
        # drop the self distances
        temp3 = temp2[~np.eye(temp2.shape[0], dtype=bool)]
        cutoff.append((np.quantile(temp2, quant)))
    return max(cutoff)


def find_uniq_cell_pairs(pairs, distances):
    '''
    pairs: tuple (np.where format) 
    distances: 1d np.array <- pairwise distance 
    '''
    
    order = np.argsort(distances)
    ordered_pairs = np.array(pairs).T[order]

    fbd_cell_a = list()
    fbd_cell_b = list()
    uniq_pair = list()
    for pair in ordered_pairs:
        if pair[0] in fbd_cell_a or pair[1] in fbd_cell_b:
            continue
        else:
            uniq_pair.append(pair)
            fbd_cell_a.append(pair[0])
            fbd_cell_b.append(pair[1])
    return np.array(uniq_pair)


def overlap_intercluster(
        cell_embedding, 
        cell_fate, 
        cell_time_per_cluster, 
        cluster0_ID, 
        cluster1_ID, 
        cutoff, 
        BARELY_OVERLAP=5, 
        peak_mode='most_frequent_shift'):
    '''
    returns the indices of overlapping cells in pairs and dT.
    
    TODO: There could be cases that we cannot handle. 
    [e.g. multiple significant peaks]
    '''
    
    print("\n\nConsolidating time between clusters ", cluster0_ID, " and ",
            cluster1_ID, "\n\n")
    cluster0_cellID = np.where(cell_fate == cluster0_ID)[0]
    cluster1_cellID = np.where(cell_fate == cluster1_ID)[0]

    cell_cluster0 = cell_embedding[cluster0_cellID]
    cell_cluster1 = cell_embedding[cluster1_cellID]
    if cell_cluster0.size and cell_cluster1.size:
        temp1 = cell_cluster0 - cell_cluster1[:, None]
        temp2 = np.linalg.norm(temp1, axis=-1)
        closePairs = np.where(temp2 < cutoff)
        
        if len(closePairs[0]) == 0:
            print("No close cells between clusters\t", (cluster0_ID, cluster1_ID))
            return None, []
        #print("\ncells in cluster0: \n", cluster0_cellID[closePairs[1]])
        #print("\ncells in cluster1: \n", cluster1_cellID[closePairs[0]])
        
        # 1 cell can pair maximum 1 cell.
        idx = find_uniq_cell_pairs(closePairs, temp2[closePairs])
        # after swapping, now:
        # first column -> cluster0
        # second column -> cluster1
        idx[:,[1,0]] = idx[:,[0,1]]
    
        deltaT = dict()
        for pair in idx:
            pair_cellIDs=(cluster0_cellID[pair[0]], cluster1_cellID[pair[1]])
            deltaT[pair_cellIDs] = \
                    cell_time_per_cluster[cluster0_ID][pair_cellIDs[0]] \
                    -cell_time_per_cluster[cluster1_ID][pair_cellIDs[1]] 

        deltaT_values = np.array(list(deltaT.values()))
        
        # If there are too few overlaps, use the pair with smallest time
        # difference.
        if len(deltaT_values) < BARELY_OVERLAP:
            peak_mode = 'least_shift'

        if peak_mode in ['least_shift']:
            shiftT = deltaT_values[np.argmin(np.abs(deltaT_values))]
            closest_pair = list(deltaT.keys())[list(deltaT.values()).index(shiftT)]

        elif peak_mode in ['most_frequent_shift']:
            fig, axes = plt.subplots(nrows=1, ncols=2, 
                    gridspec_kw={'width_ratios':[1,1]}, figsize=(8,4))

            #print("Unique close pairs\n", idx)
            axes[0].title.set_text('overlapping cells between 2 clusters')
            axes[0].scatter(cell_embedding[:,0], cell_embedding[:,1], s=5, alpha=0.3)
            axes[0].scatter(cell_cluster0[idx[:,0]][:,0], cell_cluster0[idx[:,0]][:,1])
            axes[0].scatter(cell_cluster1[idx[:,1]][:,0], cell_cluster1[idx[:,1]][:,1])
            
            axes[1].title.set_text('histogram of overlapping time difference')
            sns.histplot(ax=axes[1], data=deltaT_values, kde=True, color='skyblue')
            kdeline = axes[1].lines[0]
            x = kdeline.get_xdata()
            y = kdeline.get_ydata()
            mode_idx = np.argmax(y)
            axes[1].vlines(x[mode_idx], 0, y[mode_idx], color='tomato', ls='--', lw=5)
            plt.tight_layout()
            plt.show()

            shiftT = x[mode_idx]
            # find the pair ~ shiftT
            shiftT = deltaT_values[np.argmin(np.abs(deltaT_values-shiftT))]
            closest_pair = list(deltaT.keys())[list(deltaT.values()).index(shiftT)]
        return shiftT, closest_pair


def assign_all_cell_fate(embedding, sampling_ixs, cell_fate):
    neigh = NearestNeighbors(n_neighbors=1, radius=20, n_jobs=mp.cpu_count()-1)
    neigh.fit(embedding[sampling_ixs])
    A = neigh.kneighbors_graph(embedding)
    B = A.toarray()
    all_cell_fate = np.array([(B[i][B[i]!=0]*cell_fate[B[i]!=0])[0]
                           for i in range(len(B))], dtype=int)
    return all_cell_fate


def compute_all_cell_time(embedding, cell_embedding, 
                          path_clusters, cell_fate,
                          vel_mesh, cell_grid_idx, grid_density,
                          sampling_ixs, step,
                          dt=0.001, t_total=10000, n_repeats = 10, 
                          n_jobs = mp.cpu_count()-1,
                          outfile = None):
    
    all_cell_fate = assign_all_cell_fate(embedding, sampling_ixs, cell_fate)
    clusters = np.unique(cell_fate)
    n_clusters = len(clusters)
    print("There are %d clusters." % (n_clusters))
    print("They are: ", clusters)
    
    plot_path_clusters(path_clusters, clusters, cell_embedding)

    cell_time_per_cluster = [cell_time_projection_cluster(cell_embedding, 
        path_clusters[i][0], i, cell_fate) for i in clusters]

    # intra-cluster time assignment
    for i in clusters:
        cell_time_per_cluster, refPaths = recursive_cell_time_assignment_intracluster(
            cell_time_per_cluster, 
            i, 
            [path_clusters[i][0]], 
            cell_fate, 
            cell_embedding, 
            vel_mesh, 
            cell_grid_idx, 
            grid_density,
            dt=dt, 
            t_total=t_total, 
            n_repeats=n_repeats, 
            n_jobs=n_jobs,
            MAX_ZERO_TIME_CELLS = 0.05)

    print("\nAll intra cluster cell time has been resolved.\n\n\n")
    #print("\nPlotting cluster internal time\n")
    #plot_celltime_clusters(cell_time_per_cluster, path_clusters, cell_embedding)
    
    # inter-cluster time alignment
    resolved_cell_time = cell_time_assignment_intercluster(cell_time_per_cluster, cell_fate, cell_embedding, tau = 0.05)
    cell_time = combine_clusters(resolved_cell_time)
    ordered_cell_time = np.array([cell_time[cell] for cell in sorted(cell_time.keys())])

    # interpolate to get the time for all cells.
    all_cell_time=interpolate_all_cell_time(ordered_cell_time, embedding, sampling_ixs, step)

    print("There are %d cells." % (len(cell_fate)))
    plot_cell_clusters(all_cell_fate, embedding)
    
    if outfile:
        print("\nExporting data to:\n ", outfile)
        export_cell_time(all_cell_time, all_cell_fate,  sampling_ixs, outfile)
    return all_cell_time, all_cell_fate
