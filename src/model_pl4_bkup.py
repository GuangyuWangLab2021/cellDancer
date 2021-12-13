#
# V8.  fully connected layers, V6 intial value bug fixed. There is a new NA bug in __main__
#
import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from torch.utils.data import *
from sklearn.cluster import KMeans
import seaborn as sns
import sys
from joblib import Parallel, delayed

if __name__ == "__main__":# developer test
    sys.path.append('.')
    from simulation_cnn import *
    from realdata_bkup import *
else: # make to library
    from .simulation_cnn import * 
    from .realdata_bkup import *

class L2Module(nn.Module): #can change name #set the shape of the net
    '''
    network structure
    '''
    def __init__(self, h1, h2):#all init cannot change name
        super().__init__()
        self.l1 = nn.Linear(2, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, 3)

    def forward(self, u0, s0, alpha0, beta0, gamma0, dt):#better not change name
        input = torch.tensor(np.array([np.array(u0), np.array(s0)]).T)
        x = self.l1(input)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        x = self.l3(x)
        output = F.sigmoid(x)
        beta = torch.mean(output[:,0])
        gamma = torch.mean(output[:,1])
        alphas = output[:,2]

        alphas = alphas * alpha0
        beta =  beta * beta0
        gamma = gamma * gamma0

        u1 = u0 + (alphas - beta*u0)*dt
        s1 = s0 + (beta*u0 - gamma*s0)*dt
        return u1, s1, alphas, beta, gamma

    def save(self, model_path):
        torch.save({
            "l1": self.l1,
            "l2": self.l2,
            "l3": self.l3
        }, model_path)

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.l1 = checkpoint["l1"]
        self.l2 = checkpoint["l2"]
        self.l3 = checkpoint["l3"]

class DynamicModule(nn.Module):
    '''
    calculate loss function
    load network "L2Module"
    predict s1 u1
    '''
    def __init__(self, module, n_neighbors=30):
        super().__init__()
        self.module = module
        self.n_neighbors = n_neighbors

    def cost_fn(self, u0, s0, alpha0, beta0, gamma0, barcode = None, dt = 0.5):
        '''
        for real dataset
        calculate loss function
        predict u1 s1 from network 
        '''

        #generate neighbour indices and expr dataframe
        #print(u0, s0)
        points = np.array([s0.numpy(), u0.numpy()]).transpose()
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        expr = pd.merge(pd.DataFrame(s0, columns=['s0']), pd.DataFrame(u0, columns=['u0']), left_index=True, right_index=True)
        if barcode is not None:
            expr.index = barcode

        u0 = torch.tensor(expr['u0'])
        s0 = torch.tensor(expr['s0'])
        indices = torch.tensor(indices)

        u1, s1, alphas, beta, gamma = self.module(u0, s0, alpha0, beta0, gamma0, dt)

        def cosine(u0, s0, u1, s1, indices):
            """Cost function
            Return:
                list of cosine distance
            """
            # Velocity from (u0, s0) to (u1, s1)
            uv, sv = u1-u0, s1-s0 
            # Velocity from (u0, s0) to its neighbors
            unv, snv = u0[indices.T[1:]] - u0, s0[indices.T[1:]] - s0 

            den = torch.sqrt(unv**2 + snv**2) * torch.sqrt(uv**2+sv**2)
            den[den==0] = -1 # den==0 will cause nan in training 
            cosine = torch.where(den!=-1, (unv*uv + snv*sv) / den, torch.tensor(1.))
            cosine_max = torch.max(cosine, 0)[0]
            return 1 - cosine_max   
        cost = cosine(u0, s0, u1, s1, indices)
        return cost, u1, s1, alphas, beta, gamma

    #train with true u1, s1
    def cost_fn2(self, u0, s0, u1t, s1t, alpha0, beta0, gamma0, barcode = None, dt = 0.001):
        u0 = torch.tensor(u0)
        s0 = torch.tensor(s0)
        u1, s1, alphas, beta, gamma = self.module(u0, s0, alpha0, beta0, gamma0, dt)

        def cosine(u0, s0, u1, s1, u1t, s1t):
            """Cost function
            Return:
                list of cosine distance
            """
            # Velocity from (u0, s0) to (u1, s1)
            uv, sv = u1-u0, s1-s0 
            # Velocity from (u0, s0) to (u1t, s1t)
            utv, stv = u1t-u0, s1t-s0

            den = torch.sqrt(utv**2 + stv**2) * torch.sqrt(uv**2+sv**2)
            den[den==0] = -1 # den==0 will cause nan in training 
            cosine = torch.where(den!=-1, (utv*uv + stv*sv) / den, torch.tensor(1.))
            return 1 - cosine   
        cost = cosine(u0, s0, u1, s1, u1t, s1t)
        return cost, u1, s1, alphas, beta, gamma

    def summary_para(self, u0, s0, u1, s1, alphas, beta, gamma, cost, cost_mean, true_cost, true_cost_mean, figure=False): # before got detail; build df
        barcode = None
        detail = pd.merge(pd.DataFrame(s0, columns=['s0']), pd.DataFrame(u0, columns=['u0']), left_index=True, right_index=True) 
        detail['u1'] = u1
        detail['s1'] = s1
        detail['alpha'] = alphas
        detail['beta'] = beta
        detail['gamma'] = gamma
        detail['cost'] = cost
        detail['true_cost'] = true_cost
        if barcode is not None:
            detail.index = barcode

        cluster_number = 2
        kmeans = KMeans(n_clusters=cluster_number)
        if np.any(np.isposinf(alphas)) or np.any(np.isneginf(alphas)) or np.any(np.isnan(alphas)):
            alpha_label = 0
        else:
            alpha_label = kmeans.fit_predict(alphas.reshape(-1,1))
        detail['alpha_label'] = alpha_label

        alpha1 = np.median(detail.alpha[detail.alpha_label==0])
        alpha2 = np.median(detail.alpha[detail.alpha_label==1])
        if alpha1 < alpha2:
            alpha1, alpha2 = alpha2, alpha1
        beta = detail.beta[0]
        gamma = detail.gamma[0]
        brief = pd.DataFrame({
            'alpha1': alpha1,
            'alpha2': alpha2,
            'beta': beta,
            'gamma': gamma,
            'cost': cost_mean,
            'true_cost': true_cost_mean}, index=[0])

        if figure:
            self.summary(detail)
        return detail, brief

    def summary(self, detail): #display fig for detail (need?)
        cols, rows = 2, 1
        axs = plt.figure(figsize=(cols*4,rows*4), constrained_layout=True).subplots(rows, cols)
        pcm0 = sns.scatterplot(data=detail, x='s0', y='u0', hue='alpha', s=5, style='alpha_label', ax=axs[0]) 
        norm = plt.Normalize(detail['alpha'].min(), detail['alpha'].max())
        sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
        sm.set_array([])
        pcm0.get_legend().remove()
        pcm0.figure.colorbar(sm, ax=axs[0])
        pcm1 = axs[1].quiver(detail['s0'], detail['u0'], detail['s1']-detail['s0'], detail['u1']-detail['u0'], detail['cost']**(1/10), angles='xy', cmap=plt.cm.jet, clim=(0., 1.))
        plt.colorbar(pcm1, cmap=plt.cm.jet, ax=axs[1])
        plt.show()

class ltmodule(pl.LightningModule):
    '''
    taining network using loss function "DynamicModule"
    '''
    def __init__(self, backbone, pretrain=False, initial_zoom=1, initial_strech=1):
        super().__init__()
        self.backbone = backbone   # load network; caculate loss function; predict u1 s1 ("DynamicModule")
        self.pretrain = pretrain   # 
        self.validation_brief = pd.DataFrame() # cost score (every 10 times training)
        self.test_detail = None
        self.test_brief = None
        self.initial_zoom = initial_zoom
        self.initial_strech = initial_strech

    def save(self, model_path):
        self.backbone.module.save(model_path)    # save network

    def load(self, model_path):
        self.backbone.module.load(model_path)   # load network

    def configure_optimizers(self):      # name cannot be changed # define optimizer and paramter in optimizer need to test parameter !!!
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.8)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.8, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):# name cannot be changed 
        '''
        traning network
        batch: [] output returned from realDataset.__getitem__
        
        '''
        u0s, s0s, u1ts, s1ts, true_alphas, true_betas, true_gammas, gene_names, types, u0maxs, s0maxs = batch
        u0, s0, u1t, s1t, true_alpha, true_beta, true_gamma, gene_name, type, u0max, s0max = u0s[0], s0s[0], u1ts[0], s1ts[0], true_alphas[0], true_betas[0], true_gammas[0], gene_names[0], types[0], u0maxs[0], s0maxs[0]

        umax = np.max(np.array(u0))
        smax = np.max(np.array(s0))
        alpha0 = np.float32(umax*self.initial_zoom)
        beta0 = np.float32(1.0)
        gamma0 = np.float32(umax/smax*self.initial_strech)

        if self.pretrain:
            cost, u1, s1, alphas, beta, gamma = self.backbone.cost_fn2(u0, s0, u1t, s1t, alpha0, beta0, gamma0) # for simulation
        else:
            cost, u1, s1, alphas, beta, gamma = self.backbone.cost_fn(u0, s0, alpha0, beta0, gamma0) # for real dataset, u0: np.array(u0 for cells selected by __getitem__) to a tensor in pytorch, s0 the same as u0
        cost_mean = torch.mean(cost)    # cost: a list of cost of each cell for a given gene
        self.log("loss", cost_mean) # used for early stop. controled by log_every_n_steps(default 50) 

        return {
            "loss": cost_mean,
            "beta": beta,
            "gamma": gamma
        } 

    def training_epoch_end(self, outputs):# name cannot be changed 
        '''
        not used yet; steps after finished each epoch
        '''
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        beta = torch.stack([x["beta"] for x in outputs]).mean()
        gamma = torch.stack([x["gamma"] for x in outputs]).mean()

        #self.logger.experiment.add_scalar("loss", loss, self.current_epoch) #override loss in log
        #self.logger.experiment.add_scalar("beta", beta.data, self.current_epoch)
        #self.logger.experiment.add_scalar("gamma", gamma.data, self.current_epoch)
        #for name,params in self.backbone.module.named_parameters():
        #    self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def validation_step(self, batch, batch_idx):# name cannot be changed 
        '''
        predict u1, s1 on the training dataset 
        caculate every 10 times taining
        '''
        u0s, s0s, u1ts, s1ts, true_alphas, true_betas, true_gammas, gene_names, types, u0maxs, s0maxs = batch
        u0, s0, u1t, s1t, true_alpha, true_beta, true_gamma, gene_name, type, u0max, s0max = u0s[0], s0s[0], u1ts[0], s1ts[0], true_alphas[0], true_betas[0], true_gammas[0], gene_names[0], types[0], u0maxs[0], s0maxs[0]

        umax = np.max(np.array(u0))
        smax = np.max(np.array(s0))
        alpha0 = np.float32(umax*2)
        beta0 = np.float32(1.0)
        gamma0 = np.float32(umax/smax)

        if self.pretrain:
            cost, u1, s1, alphas, beta, gamma = self.backbone.cost_fn2(u0, s0, u1t, s1t, alpha0, beta0, gamma0)
        else:
            cost, u1, s1, alphas, beta, gamma = self.backbone.cost_fn(u0, s0, alpha0, beta0, gamma0)
            true_cost, t1, t2, t3, t4, t5 = self.backbone.cost_fn2(u0, s0, u1t, s1t, alpha0, beta0, gamma0)
        cost_mean = torch.mean(cost)
        true_cost_mean = torch.mean(true_cost)
        detail, brief = self.backbone.summary_para(
            u0, s0, u1.data.numpy(), s1.data.numpy(), 
            alphas.data.numpy(), beta.data.numpy(), gamma.data.numpy(), 
            cost.data.numpy(), cost_mean.data.numpy(),
            true_cost.data.numpy(), true_cost_mean.data.numpy())
        
        ## For single figure debug
        #print(self.current_epoch, "alpha0, beta0, gamma0")
        #print(alpha0, beta0, gamma0)
        #print(brief)
        #self.backbone.summary(detail)

        brief.insert(0, "gene_name", gene_name)
        brief.insert(1, "type", type)
        brief.insert(2, "epoch", self.current_epoch)

        if self.validation_brief.empty:
            self.validation_brief = brief
        else:
            self.validation_brief = self.validation_brief.append(brief)

    def test_step(self, batch, batch_idx):# name cannot be changed 
        '''
        define test_step
        '''

        u0s, s0s, u1ts, s1ts, true_alphas, true_betas, true_gammas, gene_names, types, u0maxs, s0maxs = batch
        u0, s0, u1t, s1t, true_alpha, true_beta, true_gamma, gene_name, type, u0max, s0max = u0s[0], s0s[0], u1ts[0], s1ts[0], true_alphas[0], true_betas[0], true_gammas[0], gene_names[0], types[0], u0maxs[0], s0maxs[0]

        umax = np.max(np.array(u0))
        smax = np.max(np.array(s0))
        alpha0 = np.float32(umax*2)
        beta0 = np.float32(1.0)
        gamma0 = np.float32(umax/smax)

        if self.pretrain:
            cost, u1, s1, alphas, beta, gamma = self.backbone.cost_fn2(u0, s0, u1t, s1t, alpha0, beta0, gamma0)
        else:
            cost, u1, s1, alphas, beta, gamma = self.backbone.cost_fn(u0, s0, alpha0, beta0, gamma0)
            true_cost, t1, t2, t3, t4, t5 = self.backbone.cost_fn2(u0, s0, u1t, s1t, alpha0, beta0, gamma0)
        cost_mean = torch.mean(cost)
        true_cost_mean = torch.mean(true_cost)
        self.test_detail, self.test_brief = self.backbone.summary_para(
            u0, s0, u1.data.numpy(), s1.data.numpy(), 
            alphas.data.numpy(), beta.data.numpy(), gamma.data.numpy(), 
            cost.data.numpy(), cost_mean.data.numpy(),
            true_cost.data.numpy(), true_cost_mean.data.numpy())
        
        self.test_detail.insert(0, "gene_name", gene_name)
        self.test_detail.insert(1, "type", type)
        self.test_brief.insert(0, "gene_name", gene_name)
        self.test_brief.insert(1, "type", type)

    def summary_test(self):
        return self.backbone.summary(self.test_result)

class dataMododule(pl.LightningDataModule): # data module for simulation data
    def __init__(self, data_path: str="./", type: str="all", index=None, point_number=600):
        super().__init__()
        self.training_dir = data_path
        self.training_dataset = SimuDataset(data_path=data_path, type=type, index=index, point_number=point_number)
        self.test_dataset = SimuDataset(data_path=data_path, type=type, index=index)
    
    def subset(self, indices):
        import copy
        temp = copy.copy(self)
        temp.training_dataset = Subset(self.training_dataset, indices)
        temp.test_dataset = Subset(self.test_dataset, indices)
        return temp

    def train_dataloader(self):# name cannot be changed 
        return DataLoader(self.training_dataset)

    def val_dataloader(self):# name cannot be changed 
        return DataLoader(self.test_dataset)

    def test_dataloader(self):# name cannot be changed 
        return DataLoader(self.test_dataset)

class realDataMododule(pl.LightningDataModule):
    '''
    load training and test data
    '''
    def __init__(self, adata=None, loom_file="", gene_list=None, smoothing=True, k=500, n_pca_dims=19, pooling=False, pooling_method='x^2+y^2=1', pooling_scale=2):
        super().__init__()
        self.training_dataset = realDataset(adata=adata, loom_file=loom_file, gene_list=gene_list, smoothing=smoothing, k=k, n_pca_dims=n_pca_dims, pooling=pooling, pooling_method=pooling_method, pooling_scale=pooling_scale)
        self.test_dataset = realDataset(adata=adata, loom_file=loom_file, gene_list=gene_list, smoothing=smoothing, k=k, n_pca_dims=n_pca_dims, pooling=False, pooling_method=pooling_method, pooling_scale=pooling_scale)

    def subset(self, indices):
        import copy
        temp = copy.copy(self)
        temp.training_dataset = Subset(self.training_dataset, indices)
        temp.test_dataset = Subset(self.test_dataset, indices)
        return temp

    def train_dataloader(self):# name cannot be changed 
        return DataLoader(self.training_dataset)
    def val_dataloader(self):# name cannot be changed 
        return DataLoader(self.training_dataset)
    def test_dataloader(self):# name cannot be changed 
        return DataLoader(self.test_dataset)

def _pretrain_thread(model_name, model_path, n_neighbors, data_path, data_type, data_index, max_epoches=500, check_n_epoch=10, initial_zoom=1, initial_strech=1, model_save_path=None, simulation=True):
    import random
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if simulation == True:
        selected_data = dataMododule(data_path=data_path, type=data_type, index = data_index)
    else:
        selected_data = realDataMododule(index = data_index)
    backbone = DynamicModule(L2Module(100, 100), n_neighbors) # 100，100 2层
    model = ltmodule(backbone=backbone, pretrain=False, initial_zoom=initial_zoom, initial_strech=initial_strech)
    if model_path != None:
        model_path = os.path.join(model_path, model_name)
        model.load(model_path)
    trainer = pl.Trainer(
        max_epochs=max_epoches, progress_bar_refresh_rate=0, reload_dataloaders_every_n_epochs=1, 
        logger = False,
        checkpoint_callback = False,
        check_val_every_n_epoch = check_n_epoch,
        weights_summary=None)
    trainer.fit(model, selected_data)
    trainer.test(model, selected_data)

    if(model_save_path != None):
        model.save(model_save_path)

    brief = model.validation_brief
    brief.insert(0, "model", model_name)
    detail = model.test_detail
    detail.insert(0, "model", model_name)
    return brief, detail

def _train_thread(datamodule, data_indices, model_name, model_path, n_neighbors=30, max_epoches=500, check_n_epoch=10, initial_zoom=1, initial_strech=1, model_save_path=None):
    '''
    real data
    '''
    
    import random
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    backbone = DynamicModule(L2Module(100, 100), n_neighbors)    # iniate network (L2Module) and loss function (DynamicModule)
    model = ltmodule(backbone=backbone, pretrain=False, initial_zoom=initial_zoom, initial_strech=initial_strech)
    if model_path != None:
        model_path = os.path.join(model_path, model_name)
        model.load(model_path)
    trainer = pl.Trainer(
        max_epochs=max_epoches, progress_bar_refresh_rate=0, reload_dataloaders_every_n_epochs=1, 
        logger = False,
        checkpoint_callback = False,
        check_val_every_n_epoch = check_n_epoch,
        weights_summary=None)   # iniate trainer

    print("indices", data_indices)
    selected_data = datamodule.subset(data_indices)  # IMPORTANT: 这个subset对应realdata.py里的每一个get_item块，
    #因为从前如果不用subset，就会训练出一个网络，对应不同gene的不同alpha，beta，gamma；
    #但是，如果使用subset，就分块训练每个基因不同网络，效果变好

    if max_epoches > 0:
        trainer.fit(model, selected_data)   # start and finish traning network
    trainer.test(model, selected_data)    # predict using model

    if(model_save_path != None):
        model.save(model_save_path)

    brief = model.validation_brief
    brief.insert(0, "model", model_name)
    detail = model.test_detail
    detail.insert(0, "model", model_name)
    return brief, detail

def pretrain(
    model_path = None, 
    model_number = 1, 
    type="all", 
    index = None,
    n_neighbors= 30, 
    initial_zoom=2, 
    initial_strech=1, 
    model_save_path = None,
    data_path = "../../data/simulation/training.hdf5", 
    result_path = None,
    max_epoches=1000, 
    simulation = True,
    n_jobs=8):
    '''when model_path is defined, model_number wont be used'''

    if not os.path.isfile(data_path):
        print("Error: No such file:", data_path)
        return

    brief = pd.DataFrame()
    detail = pd.DataFrame()
    if simulation:
        all_data = dataMododule(data_path=data_path, type=type, index=index)
    else:
        all_data = realDataMododule()
    data_len = all_data.test_dataset.__len__()

    if model_path != None:
        model_names = os.listdir(model_path)
        model_number = len(model_names)
    else:
        model_names = list(map(lambda x: "m"+str(x), range(model_number)))

    if index == None:
        data_indices = range(data_len)
    else:
        data_indices = [index]

    for model_index in range(model_number):
        model_name = model_names[model_index]
        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_pretrain_thread)(
                model_name=model_name,
                model_path=model_path, 
                n_neighbors = n_neighbors,
                data_path=data_path, 
                data_type=type,
                data_index=data_index, 
                max_epoches=max_epoches,
                initial_zoom=initial_zoom, initial_strech=initial_strech,
                model_save_path=model_save_path,
                simulation=simulation)
            for data_index in data_indices)

        for i in range(len(result)):
            temp_brief, temp_detail = result[i]
            brief = brief.append(temp_brief)
            detail = detail.append(temp_detail)

    save_path = result_path
    if save_path != None:
        brief.to_csv(os.path.join(save_path, "brief.csv"))
        detail.to_csv(os.path.join(save_path, "detail.csv"))
    return brief, detail

def train( # use train_thread
    datamodule,
    model_path = None, 
    model_number = 1, 
    initial_zoom=2, 
    initial_strech=1, 
    model_save_path = None,
    result_path = None,
    max_epoches=1000, 
    n_jobs=8):
    '''
    multple jobs
    when model_path is defined, model_number wont be used
    '''

    all_data = datamodule
    data_len = all_data.test_dataset.__len__()

    brief = pd.DataFrame()
    detail = pd.DataFrame()

    if model_path != None:
        model_names = os.listdir(model_path)
        model_number = len(model_names)
    else:
        model_names = list(map(lambda x: "m"+str(x), range(model_number)))

    for model_index in range(model_number):
        model_name = model_names[model_index]
        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_train_thread)(
                datamodule = datamodule,
                data_indices=[data_index], 
                model_name=model_name,
                model_path=model_path, 
                max_epoches=max_epoches,
                initial_zoom=initial_zoom, initial_strech=initial_strech,
                model_save_path=model_save_path)
            for data_index in range(data_len)) #for 循环里执行train_thread

        for i in range(len(result)):
            temp_brief, temp_detail = result[i]
            brief = brief.append(temp_brief)
            detail = detail.append(temp_detail)

    save_path = result_path
    if save_path != None:
        brief.to_csv(os.path.join(save_path, "brief.csv"))
        detail.to_csv(os.path.join(save_path, "detail.csv"))
    return brief, detail

def pretrain_pipeline():
    '''pretrain pipeline left unchanged. but train function should be used instead here in the next version'''
    brief, detail = pretrain(type="normal", max_epoches=200, index=2, n_jobs=1, model_save_path = '../../data/model2/normal.pt')

    show_details(detail, cols=4)
    show_details(detail, cols=4, true_cost=True)

    sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = False)
    sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)

def train_simudata_pipeline1():
    from utilities import set_rcParams
    set_rcParams()
    import warnings
    warnings.filterwarnings("ignore")
    from analysis import show_details

    data_path = "../../data/simulation/training.hdf5"
    type = "all"
    index = None
    all_data = dataMododule(data_path=data_path, type=type, index=index)

    model_path='../../data/model2'
    max_epoches=1000
    brief, detail = train(all_data, model_path=model_path, max_epoches=max_epoches, n_jobs=8)

    show_details(detail, cols=10)
    show_details(detail, cols=10, true_cost=True)
    sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = False)
    sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = False)

    brief.to_csv(os.path.join("../../result/result_pl4", "brief_p.csv"))
    detail.to_csv(os.path.join("../../result/result_pl4", "detail_p.csv"))

def train_simudata_pipeline2():
    from utilities import set_rcParams
    set_rcParams()
    import warnings
    warnings.filterwarnings("ignore")
    from analysis import show_details

    data_path = "../../data/simulation/small.hdf5"
    type = "all"
    all_data = dataMododule(data_path=data_path, type=type, index=None)

    max_epoches=500
    brief, detail = train(all_data, model_path=None, max_epoches=max_epoches, n_jobs=8)

    show_details(detail, cols=4)
    show_details(detail, cols=4, true_cost=True)
    sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = 'brief')
    sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="true_cost", legend = 'brief')

    brief.to_csv(os.path.join("../../result/result_pl4", "brief_np.csv"))
    detail.to_csv(os.path.join("../../result/result_pl4", "detail_np.csv"))


def training_realdata_figure1b():
    from utilities import set_rcParams
    #from utilities import *
    set_rcParams()
    import warnings
    warnings.filterwarnings("ignore")
    from analysis import show_details, show_details2, show_details_simplify

    #gene_list=["Abcc8", "Cdk1", "Nfib", "Rbfox3", "Sulf2", "Wfdc15b"]
    tgene_list=["Sulf2", "Gnao1", "Actn4", "Rbfox3", "Cdk1", "Abcc8", "Nfib",  "Wfdc15b"]
    tgene_list=['Abcc8', 'Actn4', 'Adk', 'Ank', 'Anxa4', 
                    'Btbd17', 'Cdk1', 'Cpe', 'Dcdc2a', 'Gnao1', 
                    'Gng12', 'Map1b', 'Mapre3', 'Nfib', 'Nnat', 
                    'Pak3', 'Pcsk2', 'Pim2', 'Ppp3ca', 'Rap1b', 
                    'Rbfox3', 'Smoc1', 'Sulf2', 'Tcp11', 'Tmem163', 
                    'Top2a', 'Tspan7', 'Wfdc15b'] #28
    model_path = '../../data/model2'
    adata = scv.datasets.pancreas()
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    find_neighbors(adata, n_pcs=30, n_neighbors=30) # calculate !!!!!!!!
    moments(adata)
    #datamodule = realDataMododule(adata = adata_scv_pancreas(), smoothing=False, k=100, gene_list=tgene_list, pooling=True, pooling_method='x^2+y^2=1')
    # data = smoothing2(adata)

    df = adata_to_pd(data)
    id = down_sampling(df)

    #   s
    datamodule = realDataMododule(adata = df[id,:], k=100, gene_list=tgene_list, pooling=True, pooling_method="y=1/x")

    datamodule = realDataMododule(adata = adata, smoothing=False, k=100, gene_list=tgene_list, pooling=True, pooling_method="y=1/x")

    brief_e0, detail_e0 = train(datamodule,model_path='../../data/model2', max_epoches=0, n_jobs=8)
    brief_e5, detail_e5 = train(datamodule,model_path='../../data/model2', max_epoches=5, n_jobs=8)
    brief_e10, detail_e10 = train(datamodule,model_path='../../data/model2', max_epoches=10, n_jobs=8)
    brief_e50, detail_e50 = train(datamodule,model_path='../../data/model2', max_epoches=50, n_jobs=8)
    brief_e100, detail_e100 = train(datamodule,model_path='../../data/model2', max_epoches=100, n_jobs=8)
    brief_e200, detail_e200 = train(datamodule,model_path='../../data/model2', max_epoches=200, n_jobs=8)
    
    brief_e210, detail_e210 = train(datamodule,model_path='../../data/model2', max_epoches=210, n_jobs=8)
    brief_e225, detail_e225 = train(datamodule,model_path='../../data/model2', max_epoches=225, n_jobs=8)
    brief_e250, detail_e250 = train(datamodule,model_path='../../data/model2', max_epoches=250, n_jobs=8)
    brief_e275, detail_e275 = train(datamodule,model_path='../../data/model2', max_epoches=275, n_jobs=8)

    brief_e300, detail_e300 = train(datamodule,model_path='../../data/model2', max_epoches=300, n_jobs=8)
    brief_e400, detail_e400 = train(datamodule,model_path='../../data/model2', max_epoches=400, n_jobs=8)
    brief_e500, detail_e500 = train(datamodule,model_path='../../data/model2', max_epoches=500, n_jobs=8)
    brief_e1000, detail_e1000 = train(datamodule,model_path='../../data/model2', max_epoches=1000, n_jobs=8)
    

    detail_e0.to_csv("output/detailcsv_lq/adj_e/detail_e0.csv")
    detail_e5.to_csv("output/detailcsv_lq/adj_e/detail_e5.csv")
    detail_e10.to_csv("output/detailcsv_lq/adj_e/detail_e10.csv")
    detail_e50.to_csv("output/detailcsv_lq/adj_e/detail_e50.csv")
    detail_e100.to_csv("output/detailcsv_lq/adj_e/detail_e100.csv")
    detail_e200.to_csv("output/detailcsv_lq/adj_e/detail_e200.csv")

    detail_e210.to_csv("output/detailcsv_lq/adj_e/detail_e210.csv")
    detail_e225.to_csv("output/detailcsv_lq/adj_e/detail_e225.csv")
    detail_e250.to_csv("output/detailcsv_lq/adj_e/detail_e250.csv")
    detail_e275.to_csv("output/detailcsv_lq/adj_e/detail_e275.csv")


    detail_e300.to_csv("output/detailcsv_lq/adj_e/detail_e300.csv")
    detail_e400.to_csv("output/detailcsv_lq/adj_e/detail_e400.csv")
    detail_e500.to_csv("output/detailcsv_lq/adj_e/detail_e500.csv")
    detail_e1000.to_csv("output/detailcsv_lq/adj_e/detail_e1000.csv")











    brief0, detail0 = train(datamodule, model_path=model_path, max_epoches=0, n_jobs=8)
    brief1, detail1 = train(datamodule, model_path=model_path, max_epoches=5, n_jobs=8)
    brief2, detail2 = train(datamodule, model_path=model_path, max_epoches=10, n_jobs=8)
    brief3, detail3 = train(datamodule, model_path=model_path, max_epoches=100, n_jobs=8)
    brief4, detail4 = train(datamodule, model_path=model_path, max_epoches=1000, n_jobs=8)

    show_details(detail4)

    gene_list=["Sulf2"]
    color = 'firebrick'
    seed = 0
    scale=0.003

    gene_list=["Gnao1"]
    color = 'firebrick'
    seed = 0
    scale=0.05

    gene_list=["Actn4"]
    color = 'firebrick'
    seed = 0
    scale=0.005

    gene_list=["Rbfox3"]
    color = 'firebrick'
    seed = 0
    scale=0.05

    gene_list=["Wfdc15b"]
    color = 'firebrick'
    seed = 0
    scale=0.005

    gene_list=["Abcc8"]
    color = 'firebrick'
    seed = 0
    scale=0.03

    show_details_simplify(detail0, gene_name =gene_list, scale=scale, color=color, seed = seed, title="epoch0")
    #show_details_simplify(detail1, gene_name =gene_list, scale=scale, color=color, seed = seed, title="epoch5")
    show_details_simplify(detail2, gene_name =gene_list, scale=scale, color=color, seed = seed, title="epoch10")
    show_details_simplify(detail3, gene_name =gene_list, scale=scale, color=color, seed = seed, title="epoch100")
    show_details_simplify(detail4, gene_name =gene_list, scale=scale, color=color, seed = seed, title="epoch1000")

if __name__ == "__main__":
    from utilities import set_rcParams
    set_rcParams()
    import warnings
    warnings.filterwarnings("ignore")
    from analysis import *
    # could be moved to the top

    #pretrain_v2()