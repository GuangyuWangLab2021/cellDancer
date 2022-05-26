import os
import sys
import glob
import shutil
import datetime
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from tqdm import tqdm
import pkg_resources
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
from .sampling import *

class DNN_Module(nn.Module):

    """Define network structure.
    """

    def __init__(self, h1, h2):
        super().__init__()
        self.l1 = nn.Linear(2, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, 3)

    def forward(self, u0, s0, alpha0, beta0, gamma0, dt):
        input = torch.tensor(np.array([np.array(u0), np.array(s0)]).T)
        x = self.l1(input)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        x = self.l3(x)
        output = torch.sigmoid(x)
        beta = output[:,0]
        gamma = output[:,1]
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

class deep_learning_module(nn.Module):
    '''
    calculate loss function
    load network "DNN_Module"
    predict s1 u1
    '''
    def __init__(self, module, n_neighbors=30):
        super().__init__()
        self.module = module
        self.n_neighbors = n_neighbors

    def velocity_calculate(self, 
                           u0, 
                           s0, 
                           alpha0, 
                           beta0, 
                           gamma0,
                           embedding1,
                           embedding2, 
                           epoch_num, 
                           barcode = None, 
                           dt = 0.5,
                           cost2_cutoff=None,
                           trace_cost_ratio=None,
                           corrcoef_cost_ratio=None):
        '''
        add embedding
        for real dataset
        calculate loss function
        predict u1 s1 from network 
        '''
        #generate neighbor indices and expr dataframe
        points = np.array([embedding1.numpy(), embedding2.numpy()]).transpose()

        self.n_neighbors=min((points.shape[0]-1), self.n_neighbors)
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(points)
        
        distances, indices = nbrs.kneighbors(points) # indices: row -> individe cell, col -> nearby cells, value -> index of cells, the fist col is the index of row
        expr = pd.merge(pd.DataFrame(s0, columns=['s0']), pd.DataFrame(u0, columns=['u0']), left_index=True, right_index=True)
        if barcode is not None:
            expr.index = barcode
        u0 = torch.tensor(expr['u0'])
        s0 = torch.tensor(expr['s0'])
        indices = torch.tensor(indices)
        u1, s1, alphas, beta, gamma = self.module(u0, s0, alpha0, beta0, gamma0, dt)

        def cosine_similarity(u0, s0, u1, s1, indices):
            """Cost function
            Return:
                list of cosine distance and a list of the index of the next cell
            """
            
            uv, sv = u1-u0, s1-s0 # Velocity from (u0, s0) to (u1, s1)
            unv, snv = u0[indices.T[1:]] - u0, s0[indices.T[1:]] - s0 # Velocity from (u0, s0) to its neighbors
            den = torch.sqrt(unv**2 + snv**2) * torch.sqrt(uv**2+sv**2)
            den[den==0] = -1
            cosine = torch.where(den!=-1, (unv*uv + snv*sv) / den, torch.tensor(1.)) # cosine: column -> individuel cell (cellI); row -> nearby cells of cell id ; value -> cosine between col and row cells
            
            cosine_max = torch.max(cosine, 0)[0]
            cosine_max_idx = torch.argmax(cosine, 0)
            cell_idx = torch.diag(indices[:, cosine_max_idx+1])
            return 1 - cosine_max, cell_idx
        
        def trace_cost(u0, s0, u1, s1, idx,version):
            '''
            '''
            uv, sv = u1-u0, s1-s0
            tan = torch.where(sv!=1000000, uv/sv, torch.tensor(0.00001))
            atan_theta = torch.atan(tan) + torch.pi/2
            atan_theta2=atan_theta[idx]
            atan_theta3 = atan_theta[idx[idx]]
            if version=="v1":
                cost = atan_theta2/atan_theta+atan_theta3/atan_theta2
            elif version=="v2":
                cost=torch.where(atan_theta<atan_theta2, 1, 0)+torch.where(atan_theta2<atan_theta3, 1, 0) 
                
            return(cost)

        def corrcoef_cost(alphas, u0, beta, s0):
            corrcoef1 = torch.corrcoef(torch.tensor([alphas.detach().numpy(),u0.detach().numpy()]))[1,0]
            corrcoef2 = torch.corrcoef(torch.tensor([beta.detach().numpy(), s0.detach().numpy()]))[1,0]
            corrcoef = corrcoef1 + corrcoef2
            cost=torch.where(corrcoef>=torch.tensor(0.0), torch.tensor(0.0), torch.tensor(-corrcoef))
            return(cost)
            
        
        if trace_cost_ratio==0 and corrcoef_cost_ratio==0:
            cost1 = cosine_similarity(u0, s0, u1, s1, indices)[0]
            cost_fin=torch.mean(cost1)

        else:
            # cosin cost
            cost1,idx = cosine_similarity(u0, s0, u1, s1, indices)
            cost1_normalize=(cost1-torch.min(cost1))/torch.max(cost1)
            cost1_mean = torch.mean(cost1_normalize)

            # trace cost
            if trace_cost_ratio>0:
                cost2 = trace_cost(u0, s0, u1, s1, idx,"v2")
                cost2_normalize=(cost2-torch.min(cost2))/torch.max(cost2)
                cost2_mean = torch.mean(cost2_normalize)
                cost2_relu=(max((cost2_mean-cost2_cutoff), 0))

            # corrcoef cost
            if corrcoef_cost_ratio>0:
                corrcoef_cost=corrcoef_cost(alphas, u0, beta, s0)

            # sum all cost
            cosin_cost_ratio=1-trace_cost_ratio-corrcoef_cost_ratio
            cost_fin = cosin_cost_ratio*cost1_mean + \
                       trace_cost_ratio*cost2_relu + \
                       corrcoef_cost_ratio*corrcoef_cost
            
        return cost_fin, u1, s1, alphas, beta, gamma


    def summary_para_validation(self, cost_mean): 
        loss_df = pd.DataFrame({'cost': cost_mean}, index=[0])
        return(loss_df)

    def summary_para(self, u0, s0, u1, s1, alphas, beta, gamma, cost): 
        cellDancer_df = pd.merge(pd.DataFrame(s0, columns=['s0']), pd.DataFrame(u0, columns=['u0']), left_index=True, right_index=True) 
        cellDancer_df['s1'] = s1
        cellDancer_df['u1'] = u1
        cellDancer_df['alpha'] = alphas
        cellDancer_df['beta'] = beta
        cellDancer_df['gamma'] = gamma
        cellDancer_df['cost'] = cost
        return cellDancer_df

class ltModule(pl.LightningModule):
    '''
    tainn network using loss function "deep_learning_module"
    '''
    def __init__(self, 
                backbone, 
                initial_zoom=2, 
                initial_strech=1,
                learning_rate=0.001,
                cost2_cutoff=0,
                optimizer='Adam',
                trace_cost_ratio=0,
                corrcoef_cost_ratio=0,
                cost_type='smooth',
                average_cost_window_size=10,
                smooth_weight=0.9):
        super().__init__()
        self.backbone = backbone
        self.validation_loss_df = pd.DataFrame()
        self.test_cellDancer_df = None
        self.test_loss_df = None
        self.initial_zoom = initial_zoom
        self.initial_strech = initial_strech
        self.learning_rate=learning_rate
        self.cost2_cutoff=cost2_cutoff
        self.optimizer=optimizer
        self.trace_cost_ratio=trace_cost_ratio
        self.corrcoef_cost_ratio=corrcoef_cost_ratio
        self.save_hyperparameters()
        self.get_loss=1000
        self.cost_type=cost_type
        self.average_cost_window_size=average_cost_window_size # will be used only when cost_tpye.isin(['average', 'median'])
        self.cost_window=[]
        self.smooth_weight=smooth_weight
        
    def save(self, model_path):
        self.backbone.module.save(model_path)    # save network

    def load(self, model_path):
        self.backbone.module.load(model_path)   # load network

    def configure_optimizers(self):     # define optimizer
        if self.optimizer=="Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=10**(-8), weight_decay=0.004, amsgrad=False)
        elif self.optimizer=="SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.8)
        return optimizer

    def training_step(self, batch, batch_idx):
        '''
        traning network
        batch: [] output returned from realDataset.__getitem__
        
        '''

        u0s, s0s, gene_names, u0maxs, s0maxs, embedding1s, embedding2s = batch
        u0, s0, u0max, s0max, embedding1, embedding2  = u0s[0], s0s[0], u0maxs[0], s0maxs[0], embedding1s[0], embedding2s[0]
        
        umax = u0max
        smax = s0max
        alpha0 = np.float32(umax*self.initial_zoom)
        beta0 = np.float32(1.0)
        gamma0 = np.float32(umax/smax*self.initial_strech)

        cost, u1, s1, alphas, beta, gamma = self.backbone.velocity_calculate(u0, s0, alpha0, beta0, gamma0,embedding1,embedding2,self.current_epoch,cost2_cutoff=self.cost2_cutoff,trace_cost_ratio=self.trace_cost_ratio,corrcoef_cost_ratio=self.corrcoef_cost_ratio)
        if self.cost_type=='average': # keep the window len <= check_val_every_n_epoch
            if len(self.cost_window)<self.average_cost_window_size:
                self.cost_window.append(cost)
            else:
                self.cost_window.pop(0)
                self.cost_window.append(cost)
            self.get_loss = torch.mean(torch.stack(self.cost_window))
            self.log("loss", self.get_loss)
            
        elif self.cost_type=='median': # keep the window len <= check_val_every_n_epoch
            if len(self.cost_window)<self.average_cost_window_size:
                self.cost_window.append(cost)
            else:
                self.cost_window.pop(0)
                self.cost_window.append(cost)
            self.get_loss = torch.median(torch.stack(self.cost_window))
            self.log("loss", self.get_loss)
            
        elif self.cost_type=='smooth':
            if self.get_loss==1000:
                self.get_loss=cost
            smoothed_val = cost * self.smooth_weight + (1 - self.smooth_weight) * self.get_loss  # calculate smoothed value
            self.get_loss = smoothed_val  
            self.log("loss", self.get_loss)
        else:
            self.get_loss = cost
            self.log("loss", self.get_loss) 
        
        return {
            "loss": cost,
            "beta": beta.detach(),
            "gamma": gamma.detach()
        }

    def training_epoch_end(self, outputs):
        '''
        steps after finished each epoch
        '''

        loss = torch.stack([x["loss"] for x in outputs]).mean()
        beta = torch.stack([x["beta"] for x in outputs]).mean()
        gamma = torch.stack([x["gamma"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):# name cannot be changed 
        '''
        predict u1, s1 on the training dataset 
        caculate every 10 times taining
        '''

        u0s, s0s, gene_names, u0maxs, s0maxs, embedding1s, embedding2s = batch
        u0, s0,gene_name, u0max, s0max, embedding1, embedding2  = u0s[0], s0s[0], gene_names[0], u0maxs[0], s0maxs[0], embedding1s[0], embedding2s[0]
        if self.current_epoch!=0:
            cost = self.get_loss.data.numpy()
            loss_df = self.backbone.summary_para_validation(cost)
            loss_df.insert(0, "gene_name", gene_name)
            loss_df.insert(1, "epoch", self.current_epoch)
            if self.validation_loss_df.empty:
                self.validation_loss_df = loss_df
            else:
                self.validation_loss_df = self.validation_loss_df.append(loss_df)

    def test_step(self, batch, batch_idx):
        '''
        define test_step
        '''
        u0s, s0s, gene_names, u0maxs, s0maxs, embedding1s, embedding2s = batch
        u0, s0, gene_name, u0max, s0max, embedding1, embedding2  = u0s[0], s0s[0], gene_names[0], u0maxs[0], s0maxs[0], embedding1s[0], embedding2s[0]
        umax = u0max
        smax = s0max
        alpha0 = np.float32(umax*2)
        beta0 = np.float32(1.0)
        gamma0 = np.float32(umax/smax)
        cost, u1, s1, alphas, beta, gamma = self.backbone.velocity_calculate(u0, s0, alpha0, beta0, gamma0,embedding1,embedding2,self.current_epoch,cost2_cutoff=self.cost2_cutoff,trace_cost_ratio=self.trace_cost_ratio,corrcoef_cost_ratio=self.corrcoef_cost_ratio)
        self.test_cellDancer_df= self.backbone.summary_para(
            u0, s0, u1.data.numpy(), s1.data.numpy(), 
            alphas.data.numpy(), beta.data.numpy(), gamma.data.numpy(), 
            cost.data.numpy())
        
        self.test_cellDancer_df.insert(0, "gene_name", gene_name)
        self.test_cellDancer_df.insert(0, "cellIndex", self.test_cellDancer_df.index)


class getItem(Dataset): # TO DO: Change to a suitable name
    def __init__(self, data_fit=None, data_predict=None,datastatus="predict_dataset", permutation_ratio=0.1,norm_u_s=True,norm_cell_distribution=False): 
        self.data_fit=data_fit
        self.data_predict=data_predict
        self.datastatus=datastatus
        self.permutation_ratio=permutation_ratio
        self.gene_name=list(data_fit.gene_name.drop_duplicates())
        self.norm_u_s=norm_u_s
        self.norm_max_u0=None
        self.norm_max_s0=None
        self.norm_cell_distribution=norm_cell_distribution

    def __len__(self):# name cannot be changed 
        return len(self.gene_name) # gene count

    def __getitem__(self, idx):# name cannot be changed
        gene_name = self.gene_name[idx]

        if self.datastatus=="fit_dataset":
            data_fitting=self.data_fit[self.data_fit.gene_name==gene_name] # u0 & s0 for cells for one gene
            if self.norm_cell_distribution==True:    # select cells to train using norm_cell_distribution methods
                u0 = data_fitting.u0
                s0 = data_fitting.s0
                u0max_fit = np.float32(max(u0))
                s0max_fit = np.float32(max(s0))
                u0 = np.round(u0/u0max_fit, 2)*u0max_fit
                s0 = np.round(s0/s0max_fit, 2)*s0max_fit
                upoints = np.unique(np.array([u0, s0]), axis=1)
                u0 = upoints[0]
                s0 = upoints[1]
                data_fitting = pd.DataFrame({'gene_name':gene_name,'u0':u0, 's0':s0,'embedding1':u0,'embedding2':s0})
        
            # random sampling in each epoch
            if self.permutation_ratio==1:
                data=data_fitting
            elif (self.permutation_ratio<1) & (self.permutation_ratio>0):
                data=data_fitting.sample(frac=self.permutation_ratio)  # select cells to train using random methods
            else:
                print('sampling ratio is wrong!')
        elif self.datastatus=="predict_dataset":
            data_pred=self.data_predict[self.data_predict.gene_name==gene_name] # u0 & s0 for cells for one gene
            data=data_pred
            
        data_pred=self.data_predict[self.data_predict.gene_name==gene_name] # u0 & s0 for cells for one gene

        u0max = np.float32(max(data_pred["u0"]))
        s0max = np.float32(max(data_pred["s0"]))
        u0 = np.array(data.u0.copy().astype(np.float32))
        s0 = np.array(data.s0.copy().astype(np.float32))
        if self.norm_u_s:
            u0=u0/u0max
            s0=s0/s0max

        # add embedding
        embedding1 = np.array(data.embedding1.copy().astype(np.float32))
        embedding2 = np.array(data.embedding2.copy().astype(np.float32))

        return u0, s0, gene_name, u0max, s0max, embedding1, embedding2



class feedData(pl.LightningDataModule):
    '''
    load training and test data
    '''
    def __init__(self, data_fit=None, data_predict=None,permutation_ratio=1,norm_u_s=True,norm_cell_distribution=False):
        super().__init__()

        self.fit_dataset = getItem(data_fit=data_fit, data_predict=data_predict,datastatus="fit_dataset", permutation_ratio=permutation_ratio,norm_u_s=norm_u_s,norm_cell_distribution=norm_cell_distribution)
        
        self.predict_dataset = getItem(data_fit=data_fit, data_predict=data_predict,datastatus="predict_dataset", permutation_ratio=permutation_ratio,norm_u_s=norm_u_s)

    def subset(self, indices):
        import copy
        temp = copy.copy(self)
        temp.fit_dataset = Subset(self.fit_dataset, indices)
        temp.predict_dataset = Subset(self.predict_dataset, indices)
        return temp

    def train_dataloader(self):
        return DataLoader(self.fit_dataset,num_workers=0)
    def val_dataloader(self):
        return DataLoader(self.fit_dataset,num_workers=0)
    def test_dataloader(self):
        return DataLoader(self.predict_dataset,num_workers=0,)

def _train_thread(datamodule, 
                  data_indices,
                  save_path=None,
                  max_epoches=None,
                  check_val_every_n_epoch=None,
                  norm_u_s=None,
                  patience=None,
                  ini_model=None,
                  model_save_path=None):

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    backbone = deep_learning_module(DNN_Module(100, 100))    # iniate network (DNN_Module) and loss function (DynamicModule)
    model = ltModule(backbone=backbone)

    selected_data = datamodule.subset(data_indices)

    u0, s0, this_gene_name, u0max, s0max, embedding1, embedding2=selected_data.fit_dataset.__getitem__(0)

    data_df=pd.DataFrame({'u0':u0,'s0':s0,'embedding1':embedding1,'embedding2':embedding2})
    data_df['gene_name']=this_gene_name

    _, sampling_ixs_select_model, _ = downsampling_embedding(data_df, # for select model
                        para='neighbors',
                        step=(20,20),
                        n_neighbors=30,
                        target_amount=None,
                        projection_neighbor_choice='embedding')
    gene_downsampling=downsampling(data_df=data_df, gene_list=[this_gene_name], downsampling_ixs=sampling_ixs_select_model)
    if ini_model=='circle':
        model_path=model_path=pkg_resources.resource_stream(__name__,os.path.join('model', 'branch.pt')).name
    if ini_model=='branch':
        model_path=model_path=pkg_resources.resource_stream(__name__,os.path.join('model', 'branch.pt')).name
    else:
        model_path=select_initial_net(this_gene_name, gene_downsampling, data_df)
    model.load(model_path)

    early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.0, patience=patience,mode='min')

    if check_val_every_n_epoch is None:
        # not use early stop
        trainer = pl.Trainer(
            max_epochs=max_epoches, progress_bar_refresh_rate=0, reload_dataloaders_every_n_epochs=1, 
            logger = False,
            enable_checkpointing = False,
            enable_model_summary=False,
            )
    else:
        # use early stop
        trainer = pl.Trainer(
            max_epochs=max_epoches, progress_bar_refresh_rate=0, reload_dataloaders_every_n_epochs=1, 
            logger = False,
            enable_checkpointing = False,
            check_val_every_n_epoch = check_val_every_n_epoch,
            enable_model_summary=False,
            callbacks=[early_stop_callback]
            )

    if max_epoches > 0:
        trainer.fit(model, selected_data)   # train network

    trainer.test(model, selected_data,verbose=False)    # predict
    
    if(model_save_path != None):
        model.save(model_save_path)

    loss_df = model.validation_loss_df
    cellDancer_df = model.test_cellDancer_df

    if norm_u_s:
        cellDancer_df.s0=cellDancer_df.s0*s0max
        cellDancer_df.u0=cellDancer_df.u0*u0max
        cellDancer_df.s1=cellDancer_df.s1*s0max
        cellDancer_df.u1=cellDancer_df.u1*u0max
        cellDancer_df.beta=cellDancer_df.beta*u0max
        cellDancer_df.gamma=cellDancer_df.gamma*s0max

    if(model_save_path != None):
        model.save(model_save_path)
    
    header_loss_df=['gene_name','epoch','loss']
    header_cellDancer_df=['cellIndex','gene_name','s0','u0','s1','u1','alpha','beta','gamma','loss']
    
    loss_df.to_csv(os.path.join(save_path,'TEMP', ('loss'+'_'+this_gene_name+'.csv')),header=header_loss_df,index=False)
    cellDancer_df.to_csv(os.path.join(save_path,'TEMP', ('celldancer_estimation_'+this_gene_name+'.csv')),header=header_cellDancer_df,index=False)
    return None


def build_datamodule(cell_type_u_s,
                   speed_up,
                   norm_u_s,
                   permutation_ratio, 
                   norm_cell_distribution=False, 
                   gene_list=None,
                   downsample_method='neighbors',
                   n_neighbors_downsample=30,
                   step=(200,200),
                   downsample_target_amount=None):
    
    '''
    set fitting data, data to be predicted, and sampling ratio in fitting data
    '''
    step_i=step[0]
    step_j=step[1]
    
    if gene_list is None:
        data_df=cell_type_u_s[['gene_name', 'u0','s0','embedding1','embedding2','cellID']]
    else:
        data_df=cell_type_u_s[['gene_name', 'u0','s0','embedding1','embedding2','cellID']][cell_type_u_s.gene_name.isin(gene_list)]

    if speed_up:
        _, sampling_ixs, _ = downsampling_embedding(data_df,
                            para=downsample_method,
                            target_amount=downsample_target_amount,
                            step=(step_i,step_j),
                            n_neighbors=n_neighbors_downsample,
                            projection_neighbor_choice='embedding')
        data_df_one_gene=cell_type_u_s[cell_type_u_s['gene_name']==list(gene_list)[0]]
        downsample_cellid=data_df_one_gene.cellID.iloc[sampling_ixs]
        gene_downsampling=data_df[data_df.cellID.isin(downsample_cellid)]

        feed_data = feedData(data_fit = gene_downsampling, data_predict=data_df, permutation_ratio=permutation_ratio,norm_u_s=norm_u_s,norm_cell_distribution=norm_cell_distribution) # default 
    else:
        feed_data = feedData(data_fit = data_df, data_predict=data_df, permutation_ratio=permutation_ratio,norm_u_s=norm_u_s,norm_cell_distribution=norm_cell_distribution) # default 

    return(feed_data)


def velocity( # use train_thread # change name to velocity estiminate
    cell_type_u_s,
    gene_list=None,
    max_epoches=200, 
    check_val_every_n_epoch=10,
    patience=3,
    permutation_ratio=0.125,
    speed_up=True,
    norm_u_s=True,
    norm_cell_distribution=True,
    n_jobs=-1,
    save_path=None,
):

    """Velocity estimation for each cell.
        
    Arguments
    ---------
    cell_type_u_s: `pandas.DataFrame`
        Data frame of raw data. Columns=['gene_name', 'u0' ,'s0' ,'cellID' ,'clusters' ,'embedding1' ,'embedding2']
    gene_list: `list` (default: None)
        Gene set selected to train. None if to estimate the velocity of all genes.
    max_epoches: `int` (default: 200)
        Stop training once this number of epochs is reached.
    check_val_every_n_epoch: `int` (default: 10)
        Check loss every n train epochs. 
    patience: `int` (default: 3)
        Number of checks with no improvement after which training will be stopped. Under the default configuration, 3 check happens after every training epoch. 
    permutation_ratio: `float` (default: 0.125)
        Sampling ratio of cells in each epoch when training each gene.
    speed_up: `bool` (default: True)
        True if speed up by downsampling cells. False if to use all cells to train the model.
    norm_u_s: `bool` (default: True)
        True to normalize u0 and s0 if u0 or s0 of genes in this dataset that too high.
    norm_cell_distribution: `bool` (default: True)
        True if the bias of cell distribution is to be romoved on embedding space (many cell share same embedding position).
    n_jobs: `int` (default: -1)
        The maximum number of concurrently running jobs.
    save_path: `str` (default: 200)
        The directory to save the result of velocity estimation.

    Returns
    -------
    `loss_df` (pandas.DataFrame), `cellDancer_df` (pandas.DataFrame)
    """

    # set output dir
    datestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S");
    folder_name='cell_dancer_velocity_'+datestring

    if save_path is None:
        save_path=os.getcwd()

    try:shutil.rmtree(os.path.join(save_path,folder_name))
    except:os.mkdir(os.path.join(save_path,folder_name))
    save_path=os.path.join(save_path,folder_name)
    print('Using '+save_path+' as the output path.')

    try:shutil.rmtree(os.path.join(save_path,'TEMP'))
    except:os.mkdir(os.path.join(save_path,'TEMP'))
    
    # set gene_list if not given
    if gene_list is None:
        gene_list=list(cell_type_u_s.gene_name.drop_duplicates())

    # buring
    gene_list_buring=[list(cell_type_u_s.gene_name.drop_duplicates())[0]]
    datamodule=build_datamodule(cell_type_u_s,speed_up,norm_u_s,permutation_ratio,norm_cell_distribution,gene_list=gene_list_buring)

    result = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_train_thread)(
            datamodule = datamodule,
            data_indices=[data_index], 
            max_epoches=max_epoches,
            check_val_every_n_epoch=check_val_every_n_epoch,
            patience=patience,
            save_path=save_path,
            norm_u_s=norm_u_s)
        for data_index in range(0,len(gene_list_buring)))

    # clean directory
    shutil.rmtree(os.path.join(save_path,'TEMP'))
    os.mkdir(os.path.join(save_path,'TEMP'))
    
    data_len = len(gene_list)
    
    id_ranges=list()
    interval=40
    for i in range(0,data_len,interval):
        idx_start=i
        if data_len<i+interval:
            idx_end=data_len
        else:
            idx_end=i+interval
        id_ranges.append((idx_start,idx_end))
    for id_range in tqdm(id_ranges,desc="Velocity Estimation", total=len(id_ranges)):
        gene_list_batch=gene_list[id_range[0]:id_range[1]]
        datamodule=build_datamodule(cell_type_u_s,speed_up,norm_u_s,permutation_ratio,norm_cell_distribution,gene_list=gene_list_batch)

        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_train_thread)(
            datamodule = datamodule,
            data_indices=[data_index], 
            max_epoches=max_epoches,
            check_val_every_n_epoch=check_val_every_n_epoch,
            patience=patience,
            save_path=save_path,
            norm_u_s=norm_u_s)
            for data_index in range(0,len(gene_list_batch)))

    cellDancer_df = os.path.join(save_path,'TEMP', "celldancer_estimation*.csv")
    cellDancer_df_files = glob.glob(cellDancer_df)
    loss_df = os.path.join(save_path, 'TEMP',"loss*.csv")
    loss_df_files = glob.glob(loss_df)

    def combine_csv(save_path,files):
        with open(save_path,"wb") as fout:
            # first file:
            with open(files[0], "rb") as f:
                fout.write(f.read())
            # the rest:    
            for filepath in files[1:]:
                with open(filepath, "rb") as f:
                    next(f)
                    fout.write(f.read())
        return(pd.read_csv(save_path))

    cellDancer_df=combine_csv(os.path.join(save_path,"celldancer_estimation.csv"),cellDancer_df_files)
    loss_df=combine_csv(os.path.join(save_path,"celldancer_estimation.csv"),loss_df_files)

    shutil.rmtree(os.path.join(save_path,'TEMP'))

    cellDancer_df.sort_values(by = ['gene_name', 'cellIndex'], ascending = [True, True])
    onegene=cell_type_u_s[cell_type_u_s.gene_name==cell_type_u_s.gene_name[0]]
    embedding_info=onegene[['cellID','clusters','embedding1','embedding2']]
    gene_amt=len(cellDancer_df.gene_name.drop_duplicates())
    embedding_col=pd.concat([embedding_info]*gene_amt)
    embedding_col.index=cellDancer_df.index
    cellDancer_df=pd.concat([cellDancer_df,embedding_col],axis=1)
    cellDancer_df.to_csv(os.path.join(save_path, ('celldancer_estimation.csv')),index=False)

    loss_df.to_csv(os.path.join(save_path, ('loss.csv')),index=False)

    return loss_df, cellDancer_df

    
def select_initial_net(gene, gene_downsampling, data_df):
    '''
    check if right top conner has cells
    circle.pt is the model for single kinetic
    branch.pt is multiple kinetic
    '''
    gene_u_s = gene_downsampling[gene_downsampling.gene_name==gene]
    gene_u_s_full = data_df[data_df.gene_name==gene]
    
    s_max=np.max(gene_u_s.s0)
    u_max = np.max(gene_u_s.u0)
    s_max_90per = 0.9*s_max
    u_max_90per = 0.9*u_max
    
    gene_u_s_full['position'] = 'position_cells'
    gene_u_s_full.loc[(gene_u_s_full.s0>s_max_90per) & (gene_u_s_full.u0>u_max_90per), 'position'] = 'cells_corner'

    if gene_u_s_full.loc[gene_u_s_full['position']=='cells_corner'].shape[0]>0.001*gene_u_s_full.shape[0]:
        # model in circle shape
        model_path=pkg_resources.resource_stream(__name__,os.path.join('model', 'circle.pt')).name
    else:
        # model in seperated branch shape
        model_path=pkg_resources.resource_stream(__name__,os.path.join('model', 'branch.pt')).name
    return(model_path)