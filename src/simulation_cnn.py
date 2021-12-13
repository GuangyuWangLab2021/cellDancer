#
# Simulation for V6. Model
#
# gene name, splice(u0), unsplice(s0), cell info (obs & var), structure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from torch.utils.data import *
import anndata
import random
import warnings
from sklearn.neighbors import NearestNeighbors

from scipy.integrate import solve_ivp
from scipy.integrate._ivp.radau import T

def _generate_points(u0_start, s0_start, alpha, beta, gamma, t1, t2, samples):

    def trans_dynamics(t, expr): 
        s = expr[0]
        u = expr[1]
        du_dt = alpha - beta*u
        ds_dt = beta*u - gamma*s
        return [ds_dt, du_dt]

    #print("t1 and t2:", t1, t2)
    t_space = np.linspace(t1, t2, samples)
    num_sol = solve_ivp(trans_dynamics, [0, t2], [s0_start, u0_start], method='RK45', dense_output=True)
    XY_num_sol = num_sol.sol(t_space)
    S, U = XY_num_sol[0], XY_num_sol[1]
    return U, S

def _jitter(U, S, scale_u, scale_s):
    S = S + np.random.normal(loc=0.0, scale=scale_s, size=np.size(S))
    U = U + np.random.normal(loc=0.0, scale=scale_u, size=np.size(U))
    S1 = S[(S>0)&(U>0)]
    U1 = U[(S>0)&(U>0)]
    return U1, S1

def _simulate(u0_start, s0_start, alpha, beta, gamma, t1, t2, samples, dt=0.001, scale=0.02):
    u0, s0 = _generate_points(u0_start, s0_start, alpha, beta, gamma, t1, t2, samples)
    u0_end, s0_end = u0[-1], s0[-1]
    u0, s0 = _jitter(u0, s0, scale*u0, scale*s0)
    u1 = u0 + (alpha - beta*u0)*dt
    s1 = s0 + (beta*u0 - gamma*s0)*dt

    expr = pd.DataFrame(u0, columns=['u0'])
    expr['s0'] = s0
    expr['u1'] = u1
    expr['s1'] = s1
    expr['alpha'] = alpha
    expr['beta'] = beta
    expr['gamma'] = gamma
    return expr, (u0_end, s0_end)

def _simulate_without_t( u0_start, s0_start, alpha, beta, gamma, percent_start_u, percent_end_u, samples, dt=0.001, scale=0.02):
    '''percentage_u: u_end/u_max'''

    def inversed_u(u, expr): 
        t = expr[0]
        dt_du = 1/(alpha - beta*u)
        return dt_du

    if alpha != 0:
        u_max = alpha/beta
        u_start = u0_start + (u_max-u0_start) * percent_start_u/100
        u_end = u0_start + (u_max-u0_start)  * percent_end_u/100
    else:
        u_max = u0_start
        u_start = u_max * (100-percent_start_u)/100
        u_end = u_max * (100-percent_end_u)/100

    t_sol = solve_ivp(inversed_u, [u0_start, u_end], [0], method='RK45', dense_output=True)
    t1 = t_sol.sol(u_start)[0]  
    t2 = t_sol.sol(u_end)[0]  
    return _simulate(u0_start, s0_start, alpha, beta, gamma, t1, t2, samples, dt, scale)

def forward(alpha, beta, gamma, percent_u1, percent_u2, samples, dt=0.001, scale=0.02):
    expr, end = _simulate_without_t(0, 0, alpha, beta, gamma, percent_u1, percent_u2, samples, dt, scale)
    return expr

def backward(alpha, beta, gamma, percent_u1, percent_u2, samples, dt=0.001, scale=0.02):
    u0_start = alpha/beta
    s0_start = alpha/gamma
    expr, end = _simulate_without_t(u0_start, s0_start, 0, beta, gamma, percent_u1, percent_u2, samples, dt, scale)
    return expr

def two_alpha(alpha1, alpha2, beta, gamma, percent_u1, percent_u2, samples1, samples2, dt=0.001, scale=0.02):
    expr1, (new_u0_start, new_s0_start) = _simulate_without_t(0, 0, alpha1, beta, gamma, 0, percent_u1, samples1, dt, scale)
    expr2, end2  = _simulate_without_t(new_u0_start, new_s0_start, alpha2, beta, gamma, 0, percent_u2, samples2, dt, scale)
    expr = expr1.append(expr2)
    expr.index = range(len(expr))
    return expr

def two_alpha2(alpha1, alpha2, beta, gamma, percent_u1, percent_u2, samples1, samples2, dt=0.001, scale=0.02):
    expr1, end1 = _simulate_without_t(0, 0, alpha1, beta, gamma, 0, percent_u1, samples1, dt, scale)
    expr2, end2  = _simulate_without_t(0, 0, alpha2, beta, gamma, 0, percent_u2, samples2, dt, scale)
    expr = expr1.append(expr2)
    expr.index = range(len(expr))
    return expr

def two_alpha3(alpha1, alpha2, beta, gamma, percent_u1, percent_u2, samples1, samples2, dt=0.001, scale=0.02):
    exprx, (new_u0_start, new_s0_start) = _simulate_without_t(0, 0, alpha2, beta, gamma, 0, 99.9, samples1, dt, scale)
    expr1, (new_u0_start2, new_s0_start2)  = _simulate_without_t(new_u0_start, new_s0_start, alpha1, beta, gamma, 0, percent_u1, samples2, dt, scale)
    expr2, end1  = _simulate_without_t(new_u0_start2, new_s0_start2, alpha2, beta, gamma, 0, percent_u2, samples2, dt, scale)
    expr = expr1.append(expr2)
    expr.index = range(len(expr))
    return expr

def circle(alpha, beta, gamma, percent_u1, percent_u2, samples1, samples2, dt=0.001, scale=0.02):
    return two_alpha(alpha, 0, beta, gamma, percent_u1, percent_u2, samples1, samples2, dt, scale)

def simulate_graph_subplot(ax, expr, title="", label=False):
    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i, alpha in enumerate(np.unique(expr['alpha'])):
        expr1 = expr[expr['alpha']==alpha]
        ax.quiver(expr1['s0'], expr1['u0'], expr1['s1']-expr1['s0'], expr1['u1']-expr1['u0'], angles='xy', color=color[i], label=r"$\alpha$="+str(round(alpha,2)))
    ax.legend(loc='lower right')
    if label == True:
        ax.set_xlabel('s')
        ax.set_ylabel('u')
    ax.set_title(title+r' $\beta$='+str(round(expr.iloc[0]['beta'],2))+', $\gamma$='+str(round(expr.iloc[0]['gamma'],2)))

def simulate_graph(expr, title="", label=False):
    fig, ax = plt.subplots()
    simulate_graph_subplot(ax, expr, title, label)
    plt.show()

def generate_simulation(filename, gene_num = 80, cell_num=2150, transform=None):
    import random
    subtype_num = gene_num // 8
    buff = 10
    u0s, s0s, u1s, s1s, alphas = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    gene_info = pd.DataFrame(columns = ['gene_name', 'type', 'alpha1', 'alpha2', 'beta', 'gamma', 'percent_u1', 'percent_u2', 'samples'])
    #normal
    for i in range(subtype_num):
        alpha = random.randrange(5,200)/10
        beta = random.randrange(5,80)/10
        gamma = random.randrange(10,30)/10
        end_u1, end_u2 = 99, 99.99
        samples1, samples2 = cell_num//2+buff, cell_num//2+buff
        expr = circle(alpha=alpha, beta=beta, gamma=gamma, percent_u1=end_u1, percent_u2=end_u2, samples1=samples1, samples2=samples2)
        expr = expr.head(cell_num)
        gene_name = "nm"+str(i).zfill(3)
        u0s[gene_name] = expr.u0
        s0s[gene_name] = expr.s0
        u1s[gene_name] = expr.u1
        s1s[gene_name] = expr.s1
        alphas[gene_name] = expr.alpha
        gene_info = gene_info.append({'gene_name':gene_name, 'type':"normal", 'alpha1':alpha, 'alpha2':0, 'beta':beta, 'gamma':gamma, 'percent_u1':end_u1, 'percent_u2':end_u2, 'samples':len(expr)}, ignore_index=True)
    #degradation unfinished
    for i in range(subtype_num):
        alpha = random.randrange(5,200)/10
        beta = random.randrange(5,80)/10
        gamma = random.randrange(10,30)/10
        end_u1, end_u2 = 99, random.randrange(90, 95)
        samples1, samples2 = cell_num*3//5+buff, cell_num*2//5+buff
        expr = circle(alpha=alpha, beta=beta, gamma=gamma, percent_u1=end_u1, percent_u2=end_u2, samples1=samples1, samples2=samples2)
        expr = expr.head(cell_num)
        gene_name = "du"+str(i).zfill(3)
        u0s[gene_name] = expr.u0
        s0s[gene_name] = expr.s0
        u1s[gene_name] = expr.u1
        s1s[gene_name] = expr.s1
        alphas[gene_name] = expr.alpha
        gene_info = gene_info.append({'gene_name':gene_name, 'type':"degradation_unfinished", 'alpha1':alpha, 'alpha2':0, 'beta':beta, 'gamma':gamma, 'percent_u1':end_u1, 'percent_u2':end_u2, 'samples':len(expr)}, ignore_index=True)
    #early switch
    for i in range(subtype_num):
        alpha = random.randrange(5,200)/10
        beta = random.randrange(5,80)/10
        gamma = random.randrange(10,30)/10
        end_u1, end_u2 = random.randrange(60, 80), 99.99
        samples1, samples2 = cell_num*2//5+buff, cell_num*3//5+buff
        expr = circle(alpha=alpha, beta=beta, gamma=gamma, percent_u1=end_u1, percent_u2=end_u2, samples1=samples1, samples2=samples2)
        expr = expr.head(cell_num)
        gene_name = "es"+str(i).zfill(3)
        u0s[gene_name] = expr.u0
        s0s[gene_name] = expr.s0
        u1s[gene_name] = expr.u1
        s1s[gene_name] = expr.s1
        alphas[gene_name] = expr.alpha
        gene_info = gene_info.append({'gene_name':gene_name, 'type':"early_switch", 'alpha1':alpha, 'alpha2':0, 'beta':beta, 'gamma':gamma, 'percent_u1':end_u1, 'percent_u2':end_u2, 'samples':len(expr)}, ignore_index=True)
    #accelerate
    for i in range(subtype_num):
        alpha1 = random.randrange(15,50)/10
        alpha2 = random.randrange(150,200)/10
        beta = random.randrange(5,80)/10
        gamma = random.randrange(10,30)/10
        end_u1, end_u2 = random.randrange(90, 99), random.randrange(30, 50)
        samples1, samples2 = cell_num//2+buff, cell_num//2+buff
        expr = two_alpha(alpha1=alpha1, alpha2=alpha2, beta=beta, gamma=gamma, percent_u1=end_u1, percent_u2=end_u2, samples1=samples1, samples2=samples2)
        expr = expr.head(cell_num)
        gene_name = "ac"+str(i).zfill(3)
        u0s[gene_name] = expr.u0
        s0s[gene_name] = expr.s0
        u1s[gene_name] = expr.u1
        s1s[gene_name] = expr.s1
        alphas[gene_name] = expr.alpha
        gene_info = gene_info.append({'gene_name':gene_name, 'type':"accelerate", 'alpha1':alpha1, 'alpha2':alpha2, 'beta':beta, 'gamma':gamma, 'percent_u1':end_u1, 'percent_u2':end_u2, 'samples':len(expr)}, ignore_index=True)
    #multi path
    for i in range(subtype_num):
        alpha1 = random.randrange(15,50)/10
        alpha2 = random.randrange(150,200)/10
        beta = random.randrange(5,80)/10
        gamma = random.randrange(10,30)/10
        end_u1, end_u2 = random.randrange(90, 99), random.randrange(30, 50)
        samples1, samples2 = cell_num//2+buff, cell_num//2+buff
        expr = two_alpha2(alpha1=alpha1, alpha2=alpha2, beta=beta, gamma=gamma, percent_u1=end_u1, percent_u2=end_u2, samples1=samples1, samples2=samples2)
        expr = expr.head(cell_num)
        gene_name = "mp"+str(i).zfill(3)
        u0s[gene_name] = expr.u0
        s0s[gene_name] = expr.s0
        u1s[gene_name] = expr.u1
        s1s[gene_name] = expr.s1
        alphas[gene_name] = expr.alpha
        gene_info = gene_info.append({'gene_name':gene_name, 'type':"multi_path", 'alpha1':alpha1, 'alpha2':alpha2, 'beta':beta, 'gamma':gamma, 'percent_u1':end_u1, 'percent_u2':end_u2, 'samples':len(expr)}, ignore_index=True)
    #decelerate
    for i in range(subtype_num):
        alpha1 = random.randrange(150,200)/10
        alpha2 = random.randrange(15,50)/10
        beta = random.randrange(5,80)/10
        gamma = random.randrange(10,30)/10
        end_u1, end_u2 = 99, 99.99
        samples1, samples2 = cell_num//2+buff, cell_num//2+buff
        expr = two_alpha3(alpha1=alpha1, alpha2=alpha2, beta=beta, gamma=gamma, percent_u1=end_u1, percent_u2=end_u2, samples1=samples1, samples2=samples2)
        expr = expr.head(cell_num)
        gene_name = "dc"+str(i).zfill(3)
        u0s[gene_name] = expr.u0
        s0s[gene_name] = expr.s0
        u1s[gene_name] = expr.u1
        s1s[gene_name] = expr.s1
        alphas[gene_name] = expr.alpha
        gene_info = gene_info.append({'gene_name':gene_name, 'type':"decelerate", 'alpha1':alpha1, 'alpha2':alpha2, 'beta':beta, 'gamma':gamma, 'percent_u1':end_u1, 'percent_u2':end_u2, 'samples':len(expr)}, ignore_index=True)

    #forwad
    for i in range(subtype_num):
        alpha = random.randrange(5,200)/10
        beta = random.randrange(5,80)/10
        gamma = random.randrange(10,30)/10
        begin_u, end_u = random.randrange(1, 10), random.randrange(90, 99)
        samples = cell_num+buff
        expr = forward(alpha=alpha, beta=beta, gamma=gamma, percent_u1=begin_u, percent_u2=end_u, samples=samples)
        expr = expr.head(cell_num)
        gene_name = "fw"+str(i).zfill(3)
        u0s[gene_name] = expr.u0
        s0s[gene_name] = expr.s0
        u1s[gene_name] = expr.u1
        s1s[gene_name] = expr.s1
        alphas[gene_name] = expr.alpha
        gene_info = gene_info.append({'gene_name':gene_name, 'type':"forward", 'alpha1':alpha, 'alpha2':-1, 'beta':beta, 'gamma':gamma, 'percent_u1':begin_u, 'percent_u2':end_u, 'samples':len(expr)}, ignore_index=True)
    #backward
    for i in range(subtype_num):
        alpha = random.randrange(5,200)/10
        beta = random.randrange(5,80)/10
        gamma = random.randrange(10,30)/10
        begin_u, end_u = random.randrange(1, 10), random.randrange(980, 999)/10
        samples = cell_num+buff
        expr = backward(alpha=alpha, beta=beta, gamma=gamma, percent_u1=begin_u, percent_u2=end_u, samples=samples)
        expr = expr.head(cell_num)
        gene_name = "bw"+str(i).zfill(3)
        u0s[gene_name] = expr.u0
        s0s[gene_name] = expr.s0
        u1s[gene_name] = expr.u1
        s1s[gene_name] = expr.s1
        alphas[gene_name] = expr.alpha
        gene_info = gene_info.append({'gene_name':gene_name, 'type':"backward", 'alpha1':alpha, 'alpha2':-1, 'beta':beta, 'gamma':gamma, 'percent_u1':begin_u, 'percent_u2':end_u, 'samples':len(expr)}, ignore_index=True)

    cell_info = pd.DataFrame()
    cell_info['barcode'] = s0s.index
    adata = anndata.AnnData(
        X=s0s.to_numpy(),
        obs = cell_info,
        var = gene_info,
        layers = {
            'u0s':u0s.to_numpy(),
            's0s': s0s.to_numpy(),
            'u1s':u1s.to_numpy(),
            's1s': s1s.to_numpy(),
            'alphas': alphas.to_numpy()}
    )
    adata.write_loom(filename=filename)
    #adata.write(filename=filename, compression='gzip')

def transform(adata, dimension=64, image_name="img64"):
    imgs = np.zeros((len(adata.var), 1, dimension, dimension))
    alphas = np.zeros((len(adata.var), 1, dimension, dimension))

    for i in range(len(adata.var)):
        u0max = np.max(adata.layers['u0s'][:, i])
        s0max = np.max(adata.layers['s0s'][:, i])
        u0 = adata.layers['u0s'][:,i]
        s0 = adata.layers['s0s'][:,i]
        alpha = adata.layers['alphas'][:,i]

        u0_trans = (u0*dimension/u0max).astype(int)
        s0_trans = (s0*dimension/s0max).astype(int)
        for j in range(len(adata.obs)):
            u = u0_trans[j] 
            s = s0_trans[j] 
            if u == dimension: u -= 1
            if s == dimension: s -= 1
            imgs[i, 0, u, s] += 1
            alphas[i, 0, u, s] += alpha[j]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alphas[i, 0] = alphas[i, 0] / imgs[i, 0]

    alphas[np.isnan(alphas)] = -1

    adata.varm[image_name] = imgs
    adata.varm['alpha.'+image_name] = alphas

def find_neighbors(adata, n_neighbors=10):
    for i in range(len(adata.var)):
        u0 = adata.layers['u0s'][:,i]
        s0 = adata.layers['s0s'][:,i]
        points = np.array([s0, u0]).transpose()
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)

        if i==0:
            #all_indices = np.array([indices])
            uns = np.array([u0[indices[:,1:]]])
            sns = np.array([s0[indices[:,1:]]])
        else:
            #all_indices = np.append(all_indices, [indices], axis=0)
            uns = np.append(uns, np.array([u0[indices[:,1:]]]), axis=0)
            sns = np.append(sns, np.array([s0[indices[:,1:]]]), axis=0)

    #adata.varm["neighbor_indices"] = all_indices
    adata.layers["uns"] = np.transpose(uns, (1,0,2))
    adata.layers["sns"] = np.transpose(sns, (1,0,2))

def load_simulation(path):
    adata = anndata.read_loom(path, sparse=False)
    return adata

def show_simus(adata, cols = 4):
    def trim_axs(axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]
    num = len(adata.var)
    rows = num//cols
    if num%cols != 0:
        rows = rows + 1
    if num <= 4:
        colsize = cols*4
        rowsize = rows*4
    else:
        colsize = cols*2
        rowsize = rows*2  
    axs = plt.figure(figsize=(colsize,rowsize), constrained_layout=True).subplots(rows, cols)
    if cols > 1:
        axs = trim_axs(axs, num)
    if rows + cols == 2:
        axs = [axs]

    gene_names = adata.var['gene_name']
    for ax, gene_name in zip(axs, gene_names):
        tempa = adata[:, adata.var['gene_name']==gene_name]
        u0 = pd.DataFrame({'u0':tempa.layers['u0s'][:,0]})
        s0 = pd.DataFrame({'s0':tempa.layers['s0s'][:,0]})
        u1 = pd.DataFrame({'u1':tempa.layers['u1s'][:,0]})
        s1 = pd.DataFrame({'s1':tempa.layers['s1s'][:,0]})
        alpha = pd.DataFrame({'alpha':tempa.layers['alphas'][:,0]})
        expr = u0.merge(s0, left_index=True, right_index=True).merge(u1, left_index=True, right_index=True).merge(s1, left_index=True, right_index=True).merge(alpha, left_index=True, right_index=True)
        #print(gene_name, i)
        #print(gene_info[gene_info['gene_name']==gene_name])
        #print(gene_info[gene_info['gene_name']==gene_name].beta[0])
        expr['beta'] = tempa.var['beta'][0]
        expr['gamma'] = tempa.var['gamma'][0]
        simulate_graph_subplot(ax, expr, gene_name+"\n")
    plt.show()

def show_simu_image(adata, img_name = 'img64', cols = 4):
    def trim_axs(axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]
    num = len(adata.var)
    rows = num//cols
    if num%cols != 0:
        rows = rows + 1
    if num <= 4:
        colsize = cols*4
        rowsize = rows*4
    else:
        colsize = cols*2
        rowsize = rows*2  
    axs = plt.figure(figsize=(colsize,rowsize), constrained_layout=True).subplots(rows, cols)
    if cols > 1:
        axs = trim_axs(axs, num)
    if rows + cols == 2:
        axs = [axs]

    gene_names = adata.var['gene_name']
    for ax, gene_name in zip(axs, gene_names):
        tempa = adata[:, adata.var['gene_name']==gene_name]
        image = tempa.varm[img_name][0][0]
        ax.imshow(image, origin='lower')
    plt.show()

class SimuDataset(Dataset):
    def __init__(self, data_path="./", type="all", index=None, point_number=None): #point_number=600 for training
        self.data_path = data_path
        self.type = type
        self.point_number = point_number
        self.adata = load_simulation(self.data_path)

        if self.type != "all":
            self.adata = self.adata[:, self.adata.var['type']==self.type]

        if index != None:
            self.adata = self.adata[:, index]

    def __len__(self):
        return len(self.adata.var)

    def __getitem__(self, idx):
        adata = self.adata[:, idx]
        u0max = np.max(adata.layers['u0s']).copy().astype(np.float32)
        s0max = np.max(adata.layers['s0s']).copy().astype(np.float32)

        if self.point_number != None:
            adata = adata[random.sample(range(len(adata.obs)), self.point_number), :]

        u0 = adata.layers['u0s'][:,0].copy().astype(np.float32)
        s0 = adata.layers['s0s'][:,0].copy().astype(np.float32)
        u1 = adata.layers['u1s'][:,0].copy().astype(np.float32)
        s1 = adata.layers['s1s'][:,0].copy().astype(np.float32)
        alpha = adata.layers['alphas'][:,0].copy().astype(np.float32)
        beta = adata.var['beta'][0].astype(np.float32)
        gamma = adata.var['gamma'][0].astype(np.float32)
        gene_name = adata.var['gene_name'][0]
        type = adata.var['type'][0]

        return u0, s0, u1, s1, alpha, beta, gamma, gene_name, type, u0max, s0max

class SimuDataset2(Dataset):
    def __init__(self, data_path="./", type="all", index=None, n_neighbors=10, dimension=64, image_name='img64'): #point_number=600 for training
        self.data_path = data_path
        self.type = type
        self.image_name = image_name
        self.adata = load_simulation(self.data_path)
        
        if self.type != "all":
            self.adata = self.adata[:, self.adata.var['type']==self.type]
        if index != None:
            self.adata = self.adata[:, index]

        find_neighbors(self.adata, n_neighbors=n_neighbors)
        transform(self.adata, dimension=dimension, image_name=image_name)

    def __len__(self):
        return len(self.adata.var)

    def __getitem__(self, idx):
        adata = self.adata[:, idx]
        u0max = np.float32(np.max(adata.layers['u0s']).copy())
        s0max = np.float32(np.max(adata.layers['s0s']).copy())

        image = adata.varm[self.image_name][0].copy()
        alpha = adata.varm['alpha.'+self.image_name][0].copy()
        beta = adata.var['beta'][0].astype(np.float32)
        gamma = adata.var['gamma'][0].astype(np.float32)
        gene_name = adata.var['gene_name'][0]
        type = adata.var['type'][0]

        u0 = adata.layers['u0s'][:,0].copy().astype(np.float32)
        s0 = adata.layers['s0s'][:,0].copy().astype(np.float32)
        u1 = adata.layers['u1s'][:,0].copy().astype(np.float32)
        s1 = adata.layers['s1s'][:,0].copy().astype(np.float32)
        un = adata.layers['uns'][:,0].copy().astype(np.float32)
        sn = adata.layers['sns'][:,0].copy().astype(np.float32)

        return image, u0, s0, u1, s1, un, sn, alpha, beta, gamma, gene_name, type, u0max, s0max

def test_without_t():
    expr, end = _simulate_without_t(1, 0.5, alpha=5.2, beta=2.0, gamma=1.0, percent_start_u=0, percent_end_u=60, samples=1000)
    simulate_graph(expr)
    expr = forward(alpha=5.2, beta=2.0, gamma=1.0, percent_u1=10, percent_u2=99, samples=1000)
    simulate_graph(expr)

    expr = backward(alpha=5.2, beta=2.0, gamma=1.0, percent_u1=10, percent_u2=99, samples=1000)
    simulate_graph(expr)

    expr = two_alpha(alpha1=5.2, alpha2=15, beta=2.0, gamma=1.0, percent_u1=90, percent_u2=60, samples1=500, samples2=500)
    simulate_graph(expr)

    expr = two_alpha3(alpha1=15, alpha2=5.2, beta=2.0, gamma=1.0, percent_u1=99, percent_u2=99.9, samples1=500, samples2=500)
    simulate_graph(expr)

    expr = two_alpha2(alpha1=5.2, alpha2=15, beta=2.0, gamma=1.0, percent_u1=90, percent_u2=60, samples1=500, samples2=500)
    simulate_graph(expr)

    expr = circle(alpha=5.2, beta=2.0, gamma=1.0, percent_u1=99, percent_u2=99.99, samples1=500, samples2=500)
    simulate_graph(expr)

    expr = circle(alpha=19.6, beta=1.0, gamma=1.9, percent_u1=99, percent_u2=99.99, samples1=500, samples2=500)
    simulate_graph(expr)

def test_ode(alpha, beta, gamma, t, samples):
    def trans_dynamics(t, expr): 
        s = expr[0]
        u = expr[1]
        du_dt = alpha - beta*u
        ds_dt = beta*u - gamma*s
        return [ds_dt, du_dt]

    t_space = np.linspace(0, t, samples)
    #available methods: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
    num_sol = solve_ivp(trans_dynamics, [0, t], [0, 0], method='RK45', dense_output=True)
    XY_num_sol = num_sol.sol(t_space)
    S, U = XY_num_sol[0], XY_num_sol[1]

    u0, s0, t0 = U, S, t_space
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(s0, u0, linewidth=1)
    axs[0].set_ylabel('u')
    axs[0].set_xlabel('s')
    axs[1].plot(t0, u0, linewidth=1, label='u')
    axs[1].plot(t0, s0, linewidth=1, label='s')
    axs[1].set_xlabel('t')
    axs[1].legend()
    plt.show()

def test_inversed_ode(alpha, beta, u, samples):
    #alpha, beta, u,  samples = 5, 2,  2,  1000
    def inversed_u(u, expr): 
        t = expr[0]
        dt_du = 1/(alpha - beta*u)
        return dt_du

    u_space = np.linspace(0, u, samples)
    num_sol = solve_ivp(inversed_u, [0, u], [0], method='RK45', dense_output=True)
    XY_num_sol = num_sol.sol(u_space)
    dt_du = XY_num_sol

    fig, axs = plt.subplots(1, 1)
    axs.plot(u_space, dt_du[0], linewidth=1)
    axs.set_ylabel('t')
    axs.set_xlabel('u')
    plt.show()

def test_find_neighbors(adata):
    def test(adata, x, y):
        u0 = adata.layers['u0s'][x,y]
        s0 = adata.layers['s0s'][x,y]
        un = adata.layers['uns'][x,y]
        sn = adata.layers['sns'][x,y]
        print("u0", u0)
        print("s0", s0)
        print("un", un)
        print("sn", sn)
        print("")

    test(adata, 5, 3)
    test(adata, 10, 3)
    test(adata, 1000, 3)

def test_simuDataset():
    dataset = SimuDataset(data_path="../../data/simulation/training.hdf5", type="decelerate",)
    dataset = SimuDataset(data_path="../../data/simulation/training.hdf5", type="decelerate",)
    dataset = SimuDataset(data_path="../../data/simulation/training.hdf5", type="decelerate", index=0)
    dataset.__len__()

    dataset = SimuDataset2('../../data/simulation/training.hdf5', type='normal')
    x = dataset.__getitem__(0)

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from utilities import set_rcParams
    set_rcParams()

    #test_without_t()
    #test_ode(5, 2, 1, 2, 1000)
    #test_inversed_ode(5, 2, 5/2*0.99, 1000)

    #generate_simulation("../../data/simulation/training.hdf5", gene_num = 80)
    adata = load_simulation('../../data/simulation/training.hdf5')
    show_simus(adata[:, adata.var['gene_name'].isin(['nm000', 'nm001'])], cols = 2)
    show_simus(adata[:, adata.var['type']=='normal'], cols = 4)

    find_neighbors(adata)

    #generate_simulation("../../data/simulation/small.hdf5", gene_num = 8)
    adata = load_simulation('../../data/simulation/small.hdf5')
    show_simus(adata)
    transform(adata)
    show_simu_image(adata)
    show_simu_image(adata, img_name='alpha.img64')