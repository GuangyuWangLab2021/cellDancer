#
# V8.  fully connected layers, V6 intial value bug fixed. There is a new NA bug in __main__
#
import pytorch_lightning as pl
import os
from scipy.integrate._ivp.radau import P
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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sampling import *
from load_data import *

# if __name__ == "__main__":# developer test
#     sys.path.append('.')
#     from simulation_cnn import *
#     from load_data import *
# else: # make to library
#     from .simulation_cnn import * 
#     from .load_data import *

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
        # beta = torch.mean(output[:,0])   # mean of beta Guangyu
        # gamma = torch.mean(output[:,1])   # mean of gama Guangyu
        beta = output[:,0]   # mean of beta Guangyu
        gamma = output[:,1]   # mean of gama Guangyu
        alphas = output[:,2]

        alphas = alphas * alpha0
        beta =  beta * beta0
        gamma = gamma * gamma0

        def corrcoef_cost(alphas, u0, beta, s0):
            # print('epoch'+str(epoch_num))
            corrcoef1 = torch.corrcoef(torch.tensor([alphas.detach().numpy(),u0.detach().numpy()]))[1,0]
            #print("corrcoef1: "+str(corrcoef1))
            corrcoef2 = torch.corrcoef(torch.tensor([beta.detach().numpy(), s0.detach().numpy()]))[1,0]
            #print("corrcoef2: "+str(corrcoef2))
            corrcoef = corrcoef1 + corrcoef2
            #print("corrcoef: "+str(corrcoef))
            cost=torch.where(corrcoef>=torch.tensor(0.0), torch.tensor(0.0), torch.tensor(-corrcoef))
            # cost_corrcoef1=torch.where(corrcoef1>=torch.tensor(0.0), torch.tensor(0.0), torch.tensor(-corrcoef1))
            # cost_corrcoef2=torch.where(corrcoef2>=torch.tensor(0.0), torch.tensor(0.0), torch.tensor(-corrcoef2))
            # cost=(cost_corrcoef1+cost_corrcoef2)/2
            # if epoch_num>100: 
            #print("cost: "+str(cost))
            return(cost)

        u1 = u0 + (alphas - beta*u0)*dt
        s1 = s0 + (beta*u0 - gamma*s0)*dt

        cost = corrcoef_cost(alphas, u0, beta, s0)
        return u1, s1, alphas, beta, gamma, cost

    def save(self, model_path):
        torch.save({
            "l1": self.l1,
            "l2": self.l2,
            "l3": self.l3
        }, model_path)

    def load(self, model_path):
        print("model_path: ",model_path)
        checkpoint = torch.load(model_path)
        self.l1 = checkpoint["l1"]
        self.l2 = checkpoint["l2"]
        self.l3 = checkpoint["l3"]

class stochasticModule(nn.Module): # deep learning module
    '''
    calculate loss function
    load network "L2Module"
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
                           cost_version=1,
                           cost2_cutoff=0.3,
                           cost1_ratio=0.8,
                           with_trace_cost=True,
                           with_corrcoef_cost=True):
        '''
        add embedding (Guangyu)
        for real dataset
        calculate loss function
        predict u1 s1 from network 
        '''
        # print('epoch'+str(epoch_num))

        #generate neighbour indices and expr dataframe
        #print(u0, s0)
        points = np.array([embedding1.numpy(), embedding2.numpy()]).transpose()
        
        # 用downsampling以后的cell，计算neighbors，作为输入
        # 加入neighbor信息
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points) # indices: raw is individe cell, col is nearby cells, value is the index of cells, the fist col is the index of row
        expr = pd.merge(pd.DataFrame(s0, columns=['s0']), pd.DataFrame(u0, columns=['u0']), left_index=True, right_index=True)
        if barcode is not None:
            expr.index = barcode
        u0 = torch.tensor(expr['u0'])
        s0 = torch.tensor(expr['s0'])
        indices = torch.tensor(indices)
        u1, s1, alphas, beta, gamma, cost3 = self.module(u0, s0, alpha0, beta0, gamma0, dt)
        # print(beta)
        def cosine_similarity(u0, s0, u1, s1, indices):
            """Cost function
            Return:
                list of cosine distance and a list of the index of the next cell
            """
            # Velocity from (u0, s0) to (u1, s1)
            uv, sv = u1-u0, s1-s0 
            # Velocity from (u0, s0) to its neighbors
            unv, snv = u0[indices.T[1:]] - u0, s0[indices.T[1:]] - s0 
            den = torch.sqrt(unv**2 + snv**2) * torch.sqrt(uv**2+sv**2)
            den[den==0] = -1 # den==0 will cause nan in training 
            cosine = torch.where(den!=-1, (unv*uv + snv*sv) / den, torch.tensor(1.)) # cosine: col is individuel cell (cellI), row is nearby cells of cellI, value is the cosine between col and row cells
            cosine_max = torch.max(cosine, 0)[0]
            cosine_max_idx = torch.argmax(cosine, 0)
            cell_idx = torch.diag(indices[:, cosine_max_idx+1])
            return 1 - cosine_max, cell_idx
        
        def trace_cost(u0, s0, u1, s1, idx,version):
            '''
            Guangyu
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
            # print('epoch'+str(epoch_num))
            corrcoef1 = torch.corrcoef(torch.tensor([alphas.detach().numpy(),u0.detach().numpy()]))[1,0]
            # if epoch_num>100: 
            #print(corrcoef1)
            corrcoef2 = torch.corrcoef(torch.tensor([beta.detach().numpy(), s0.detach().numpy()]))[1,0]
            # if epoch_num>100
            #print(corrcoef2)
            corrcoef = corrcoef1 + corrcoef2
            # if epoch_num>100
            #print(corrcoef)
            cost=torch.where(corrcoef>=torch.tensor(0.0), torch.tensor(0.0), torch.tensor(-corrcoef))
            # if epoch_num>100: 
            #print(cost)
            return(cost)

        # if with_trace_cost:cost_version=2

        # with_trace_cost=self.with_trace_cost,
        # with_corrcoef_cost=self.with_corrcoef_cost



        if cost_version==1:
            cost1 = cosine_similarity(u0, s0, u1, s1, indices)[0]
            cost_fin=torch.mean(cost1)
        elif cost_version==2:
            cost1,idx = cosine_similarity(u0, s0, u1, s1, indices)
            cost2 = trace_cost(u0, s0, u1, s1, idx,"v2")

            cost1_normalize=(cost1-torch.min(cost1))/torch.max(cost1)
            cost2_normalize=(cost2-torch.min(cost2))/torch.max(cost2)

            cost1_mean = torch.mean(cost1_normalize)
            cost2_mean = torch.mean(cost2_normalize)
            if cost2_mean<cost2_cutoff:           
                cost_mean_v2 = cost1_mean
            else:
                cost_mean_v2 = cost1_ratio*cost1_mean + (1-cost1_ratio)*(cost2_mean-cost2_cutoff) # relu activate
            cost_fin=cost_mean_v2
        elif cost_version==3:
            cost1,idx = cosine_similarity(u0, s0, u1, s1, indices)
            cost2 = trace_cost(u0, s0, u1, s1, idx,"v2")
            
            
            cost1_normalize=(cost1-torch.min(cost1))/torch.max(cost1)
            cost2_normalize=(cost2-torch.min(cost2))/torch.max(cost2)
            # print(cost1_normalize[0:10])
            # print(cost2_normalize[0:10])
            # print(cost3)

            cost1_mean = torch.mean(cost1_normalize)
            cost2_mean = torch.mean(cost2_normalize)

            ratio2 = 0.3
            cost_mean_v2 = cost1_ratio*cost1_mean + (1-cost1_ratio-ratio2)*(max((cost2_mean-cost2_cutoff), 0)) + ratio2*cost3
            
            cost_fin=cost_mean_v2
            #print('cost1_mean: '+str(cost1_mean))
            # print('cost1_ratio: '+str(cost1_ratio))
            #print('cost1_ratio*cost1_mean: '+str(cost1_ratio*cost1_mean))

            #print('cost2_mean: '+str(cost2_mean))
            # print('cost2_ratio: '+str(1-cost1_ratio-ratio2))
            #print('cost2_ratio*cost2_mean: '+str((1-cost1_ratio-ratio2)*(max((cost2_mean-cost2_cutoff), 0))))

            #print('cost3: '+str(cost3))
            # print('cost3_ratio: '+str(ratio2))
            #print('cost_fin: '+str(cost_fin))

        return cost_fin, u1, s1, alphas, beta, gamma # to do

    def velocity_calculate_test(self, u0, s0, alpha0, beta0, gamma0,embedding1,embedding2, barcode = None, dt = 0.5):
        '''
        add embedding (Guangyu)
        for real dataset
        calculate loss function
        predict u1 s1 from network 
        '''

        #generate neighbour indices and expr dataframe
        #print(u0, s0)
        points = np.array([embedding1.numpy(), embedding2.numpy()]).transpose()
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points) # indices: raw is individe cell, col is nearby cells, value is the index of cells, the fist col is the index of row

        expr = pd.merge(pd.DataFrame(s0, columns=['s0']), pd.DataFrame(u0, columns=['u0']), left_index=True, right_index=True)
        if barcode is not None:
            expr.index = barcode
            
        u0 = torch.tensor(expr['u0'])
        s0 = torch.tensor(expr['s0'])
        # print('--------indices0--------')
        # print(indices)
        #np.savetxt("output/indices.csv", indices, delimiter=",")

        indices = torch.tensor(indices)
        # print('--------indices1--------')
        # print(indices)

        u1, s1, alphas, beta, gamma = self.module(u0, s0, alpha0, beta0, gamma0, dt)

        def cosine_similarity(u0, s0, u1, s1, indices):
            """Cost function
            Return:
                list of cosine distance and a list of the index of the next cell
            """
            # Velocity from (u0, s0) to (u1, s1)
            uv, sv = u1-u0, s1-s0 
            # Velocity from (u0, s0) to its neighbors
            unv, snv = u0[indices.T[1:]] - u0, s0[indices.T[1:]] - s0 

            den = torch.sqrt(unv**2 + snv**2) * torch.sqrt(uv**2+sv**2)
            den[den==0] = -1 # den==0 will cause nan in training 
            cosine = torch.where(den!=-1, (unv*uv + snv*sv) / den, torch.tensor(1.)) # cosine: col is individuel cell (cellI), row is nearby cells of cellI, value is the cosine between col and row cells
            cosine_max = torch.max(cosine, 0)[0]
            cosine_max_idx = torch.argmax(cosine, 0)
            cell_idx = torch.diag(indices[:, cosine_max_idx+1])
            # print("------cosine------")
            # print(u0.shape)
            # print(s0.shape)
            # print(u1.shape)
            # print(s1.shape)
            # print(cosine.shape)
            # print(cosine)
            # print(cosine_max.shape)
            # print(cosine_max)
            # print(cosine_max_idx)
            # print(indices)
            # print(cell_idx)
            return 1 - cosine_max, cell_idx
        cost1,idx = cosine_similarity(u0, s0, u1, s1, indices)

        cost=cost1+0
        print("cost for velocity_calculate_test:"+str(cost))

        return cost, u1, s1, alphas, beta, gamma

    #train with true u1, s1
    def velocity_calculate2(self, u0, s0, u1t, s1t, alpha0, beta0, gamma0, barcode = None, dt = 0.001):
        u0 = torch.tensor(u0)
        s0 = torch.tensor(s0)
        u1, s1, alphas, beta, gamma, _ = self.module(u0, s0, alpha0, beta0, gamma0, dt)

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
        #print("cost for velocity_calculate2:"+str(cost))

        return cost, u1, s1, alphas, beta, gamma

    def summary_para(self, u0, s0, u1, s1, alphas, beta, gamma, cost, cost_mean, backgroud_true_cost, backgroud_true_cost_mean, figure=False): # before got detail; build df
        # print("----------summary_para-----------")
        barcode = None
        detail = pd.merge(pd.DataFrame(s0, columns=['s0']), pd.DataFrame(u0, columns=['u0']), left_index=True, right_index=True) 
        detail['u1'] = u1
        detail['s1'] = s1
        detail['alpha'] = alphas
        detail['beta'] = beta
        detail['gamma'] = gamma
        detail['cost'] = cost
        detail['backgroud_true_cost'] = backgroud_true_cost
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
            'backgroud_true_cost': backgroud_true_cost_mean}, index=[0])
        print
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

class ltModule(pl.LightningModule):
    '''
    taining network using loss function "stochasticModule"
    '''
    def __init__(self, 
                backbone, 
                pretrain=False, 
                initial_zoom=1, 
                initial_strech=1,
                learning_rate=0.01,
                cost_version=1,
                cost2_cutoff=0.3,
                cost1_ratio=0.8,
                optimizer="SGD",
                with_trace_cost=True,
                with_corrcoef_cost=True):
        super().__init__()
        self.backbone = backbone   # load network; caculate loss function; predict u1 s1 ("DynamicModule")
        self.pretrain = pretrain   # 
        self.validation_brief = pd.DataFrame() # cost score (every 10 times training)
        self.test_detail = None
        self.test_brief = None
        self.initial_zoom = initial_zoom
        self.initial_strech = initial_strech
        self.learning_rate=learning_rate
        self.cost_version=cost_version
        self.cost2_cutoff=cost2_cutoff
        self.cost1_ratio=cost1_ratio
        self.optimizer=optimizer
        self.with_trace_cost=with_trace_cost
        self.with_corrcoef_cost=with_corrcoef_cost
        self.save_hyperparameters()

    def save(self, model_path):
        self.backbone.module.save(model_path)    # save network

    def load(self, model_path):
        self.backbone.module.load(model_path)   # load network

    def configure_optimizers(self):      # name cannot be changed # define optimizer and paramter in optimizer need to test parameter !!!
        if self.optimizer=="SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.8)
        elif self.optimizer=="Adam":
            # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999),eps=10**(-8),weight_decay=0.004,amsgrad=False)

        # https://blog.csdn.net/BVL10101111/article/details/72615621
        return optimizer

    def training_step(self, batch, batch_idx):# name cannot be changed 
        '''
        traning network
        batch: [] output returned from realDataset.__getitem__
        
        '''
        ###############################################
        #########       add embedding         #########
        ###############################################
        #print('-----------training_step------------')
        u0s, s0s, u1ts, s1ts, true_alphas, true_betas, true_gammas, gene_names, types, u0maxs, s0maxs, embedding1s, embedding2s = batch
        u0, s0, u1t, s1t, _, _, _, _, _, u0max, s0max, embedding1, embedding2  = u0s[0], s0s[0], u1ts[0], s1ts[0], true_alphas[0], true_betas[0], true_gammas[0], gene_names[0], types[0], u0maxs[0], s0maxs[0], embedding1s[0], embedding2s[0]
        
        umax = u0max
        smax = s0max
        alpha0 = np.float32(umax*self.initial_zoom)
        beta0 = np.float32(1.0)
        gamma0 = np.float32(umax/smax*self.initial_strech)

        if self.pretrain:
            #cost, u1, s1, alphas, beta, gamma = self.backbone.velocity_calculate2(u0, s0, u1t, s1t, alpha0, beta0, gamma0) # for simulation
            cost, u1, s1, alphas, beta, gamma = self.backbone.velocity_calculate(u0, s0, alpha0, beta0, gamma0,embedding1,embedding2,self.current_epoch,cost_version=self.cost_version,cost2_cutoff=self.cost2_cutoff,cost1_ratio=self.cost1_ratio) 
        else:
            cost, u1, s1, alphas, beta, gamma = self.backbone.velocity_calculate(u0, s0, alpha0, beta0, gamma0,embedding1,embedding2,self.current_epoch,cost_version=self.cost_version,cost2_cutoff=self.cost2_cutoff,cost1_ratio=self.cost1_ratio,with_trace_cost=self.with_trace_cost,with_corrcoef_cost=self.with_corrcoef_cost) # for real dataset, u0: np.array(u0 for cells selected by __getitem__) to a tensor in pytorch, s0 the same as u0
        # print("cost for training_step: "+str(cost))
        cost_mean=cost
        # cost_mean = torch.mean(cost)    # cost: a list of cost of each cell for a given gene
        self.log("loss", cost_mean) # used for early stop. controled by log_every_n_steps(default 50) 

        return {
            "loss": cost_mean,
            "beta": beta,
            "gamma": gamma
        } 

    def training_epoch_end(self, outputs):# name cannot be changed 
        '''
        steps after finished each epoch
        '''
        # print("training_epoch_end")
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
        ###############################################
        #########       add embedding         #########
        ###############################################
        # print('-----------validation_step------------')
        u0s, s0s, u1ts, s1ts, true_alphas, true_betas, true_gammas, gene_names, types, u0maxs, s0maxs, embedding1s, embedding2s = batch
        u0, s0, u1t, s1t, _, _, _, gene_name, type, u0max, s0max, embedding1, embedding2  = u0s[0], s0s[0], u1ts[0], s1ts[0], true_alphas[0], true_betas[0], true_gammas[0], gene_names[0], types[0], u0maxs[0], s0maxs[0], embedding1s[0], embedding2s[0]
        
        umax = u0max
        smax = s0max
        alpha0 = np.float32(umax)
        beta0 = np.float32(1.0)
        gamma0 = np.float32(umax/smax)

        if self.pretrain:
            cost, u1, s1, alphas, beta, gamma = self.backbone.velocity_calculate2(u0, s0, u1t, s1t, alpha0, beta0, gamma0)
        else:
            cost, u1, s1, alphas, beta, gamma = self.backbone.velocity_calculate(u0, s0, alpha0, beta0, gamma0,embedding1,embedding2,self.current_epoch,cost_version=self.cost_version,cost2_cutoff=self.cost2_cutoff,cost1_ratio=self.cost1_ratio)
            # print("cost for validation_step: "+str(cost))
            #backgroud_true_cost, _, _, _, _, _ = self.backbone.velocity_calculate2(u0, s0, u1t, s1t, alpha0, beta0, gamma0)
            backgroud_true_cost = self.backbone.velocity_calculate2(u0, s0, u1t, s1t, alpha0, beta0, gamma0)[0]
        cost_mean = torch.mean(cost)
        # print("cost_mean: "+str(cost_mean))
        backgroud_true_cost_mean = torch.mean(backgroud_true_cost)
        detail, brief = self.backbone.summary_para(
            u0, s0, u1.data.numpy(), s1.data.numpy(), 
            alphas.data.numpy(), beta.data.numpy(), gamma.data.numpy(), 
            cost.data.numpy(), cost_mean.data.numpy(),
            backgroud_true_cost.data.numpy(), backgroud_true_cost_mean.data.numpy())
        
        ## For single figure debug
        # print(self.current_epoch, "alpha0, beta0, gamma0")
        
        #print(alpha0, beta0, gamma0)
        #print(brief)
        #self.backbone.summary(detail)

        brief.insert(0, "gene_name", gene_name)
        brief.insert(1, "type", type)
        brief.insert(2, "epoch", self.current_epoch)

        #brief.to_csv(os.path.join(save_path, "brief.csv"),mode='a',header=False)
        

        if self.validation_brief.empty:
            self.validation_brief = brief
        else:
            self.validation_brief = self.validation_brief.append(brief)

    def test_step(self, batch, batch_idx):# name cannot be changed 
        '''
        define test_step
        '''
        ###############################################
        #########       add embedding         #########
        ###############################################
        # print('-----------test_step------------')
        u0s, s0s, u1ts, s1ts, true_alphas, true_betas, true_gammas, gene_names, types, u0maxs, s0maxs, embedding1s, embedding2s = batch
        u0, s0, u1t, s1t, _, _, _, gene_name, type, u0max, s0max, embedding1, embedding2  = u0s[0], s0s[0], u1ts[0], s1ts[0], true_alphas[0], true_betas[0], true_gammas[0], gene_names[0], types[0], u0maxs[0], s0maxs[0], embedding1s[0], embedding2s[0]
        
        umax = u0max
        smax = s0max
        alpha0 = np.float32(umax*2)
        beta0 = np.float32(1.0)
        gamma0 = np.float32(umax/smax)

        if self.pretrain:
            cost, u1, s1, alphas, beta, gamma = self.backbone.velocity_calculate2(u0, s0, u1t, s1t, alpha0, beta0, gamma0)
        else:
            cost, u1, s1, alphas, beta, gamma = self.backbone.velocity_calculate(u0, s0, alpha0, beta0, gamma0,embedding1,embedding2,self.current_epoch, cost_version=self.cost_version,cost2_cutoff=self.cost2_cutoff,cost1_ratio=self.cost1_ratio)
            backgroud_true_cost, _, _, _, _, _ = self.backbone.velocity_calculate2(u0, s0, u1t, s1t, alpha0, beta0, gamma0)
        cost_mean = torch.mean(cost)
        backgroud_true_cost_mean = torch.mean(backgroud_true_cost)
        self.test_detail, self.test_brief = self.backbone.summary_para(
            u0, s0, u1.data.numpy(), s1.data.numpy(), 
            alphas.data.numpy(), beta.data.numpy(), gamma.data.numpy(), 
            cost.data.numpy(), cost_mean.data.numpy(),
            backgroud_true_cost.data.numpy(), backgroud_true_cost_mean.data.numpy())
        
        self.test_detail.insert(0, "gene_name", gene_name)
        self.test_detail.insert(1, "type", type)
        self.test_brief.insert(0, "gene_name", gene_name)
        self.test_brief.insert(1, "type", type)

    def summary_test(self):
        return self.backbone.summary(self.test_result)

class dataModule(pl.LightningDataModule): # data module for simulation data
    def __init__(self, data_path: str="./", type: str="all", index=None, point_number=600):
        super().__init__()
        self.training_dir = data_path
        # self.training_dataset = SimuDataset(data_path=data_path, type=type, index=index, point_number=point_number)
        # self.test_dataset = SimuDataset(data_path=data_path, type=type, index=index)
    
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

class feedData(pl.LightningDataModule): #change name to feedData
    '''
    load training and test data
    '''
    def __init__(self, data_fit=None, data_predict=None,sampling_ratio=1):
        super().__init__()

        #change name to fit
        self.training_dataset = realDataset(data_fit=data_fit, data_predict=data_predict,datastatus="fit_dataset", sampling_ratio=sampling_ratio)
        
        #change name to predict
        self.test_dataset = realDataset(data_fit=data_fit, data_predict=data_predict,datastatus="predict_dataset", sampling_ratio=sampling_ratio)

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
        selected_data = dataModule(data_path=data_path, type=data_type, index = data_index)
    else:
        selected_data = feedData(index = data_index)
    backbone = stochasticModule(L2Module(100, 100), n_neighbors) # 100，100 2层
    model = ltModule(backbone=backbone, pretrain=False, initial_zoom=initial_zoom, initial_strech=initial_strech)
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

def _train_thread(datamodule, 
                    data_indices, 
                    model_name, 
                    model_path, 
                    result_path=None,
                    n_neighbors=30, 
                    max_epoches=500, 
                    check_n_epoch=10, 
                    initial_zoom=1, 
                    initial_strech=1, 
                    model_save_path=None,
                    learning_rate=0.01,
                    cost_version=1,
                    cost2_cutoff=0.3,
                    cost1_ratio=0.8,
                    optimizer="SGD",
                    filepath_brief=None,
                    filepath_detail=None,
                    gene_shape_classify_dict=None,
                    with_trace_cost=True,
                    with_corrcoef_cost=True):
    '''
    real data
    '''
    
    import random
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    backbone = stochasticModule(L2Module(100, 100), n_neighbors)    # iniate network (L2Module) and loss function (DynamicModule)
    model = ltModule(backbone=backbone, 
                    pretrain=False, 
                    initial_zoom=initial_zoom, 
                    initial_strech=initial_strech,
                    learning_rate=learning_rate,
                    cost_version=cost_version,
                    cost2_cutoff=cost2_cutoff,
                    cost1_ratio=cost1_ratio,
                    optimizer=optimizer,
                    with_trace_cost=with_trace_cost,
                    with_corrcoef_cost=with_corrcoef_cost)


    print("indices", data_indices)
    selected_data = datamodule.subset(data_indices)  # IMPORTANT: 这个subset对应realdata.py里的每一个get_item块，
    #因为从前如果不用subset，就会训练出一个网络，对应不同gene的不同alpha，beta，gamma；
    #但是，如果使用subset，就分块训练每个基因不同网络，效果变好

    #print("---selected_data---")
    #print(selected_data )
    #print("---selected_data.training_dataset---")
    u0, s0, u1, s1, alpha, beta, gamma, this_gene_name, type, u0max, s0max, embedding1, embedding2=selected_data.training_dataset.__getitem__(0)
    #print("this_gene_name: "+this_gene_name)


    if model_path != None:
        this_gene_model=gene_shape_classify_dict[gene_shape_classify_dict.gene_name==this_gene_name]['model_type_dir'].reset_index(drop=True)[0]

        model_path=this_gene_model
        print('model_path')
        print(model_path)
        model.load(model_path)

        #model.load('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Sulf2/Sulf2.pt')
        model_name=gene_shape_classify_dict[gene_shape_classify_dict.gene_name==this_gene_name]['model_type'].reset_index(drop=True)[0]
        # model_path = os.path.join(model_path, model_name)
        # model.load(model_path)
    early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.0, patience=3,mode='min')
    checkpoint_callback = ModelCheckpoint(monitor="loss",
                                          dirpath='output/callback_checkpoint/',
                                          save_top_k=1,
                                          mode='min',
                                          auto_insert_metric_name=True
                                          )

    trainer = pl.Trainer(
        max_epochs=max_epoches, progress_bar_refresh_rate=0, reload_dataloaders_every_n_epochs=1, 
        logger = False,
        checkpoint_callback = False,
        check_val_every_n_epoch = check_n_epoch,
        weights_summary=None,
        callbacks=[early_stop_callback]
        #callbacks=[early_stop_callback,checkpoint_callback]
        )
    '''   by Lingqun
    trainer = pl.Trainer(
    max_epochs=500, progress_bar_refresh_rate=0, reload_dataloaders_every_n_epochs=1,
    logger=logger,
    logger = False,
    checkpoint_callback = False,
    check_val_every_n_epoch = 10,
    log_every_n_steps=50,
    weights_summary=None)#,
    callbacks=[EarlyStopping(monitor="loss", min_delta=0.0, patience=200)])
    '''





    if max_epoches > 0:
        trainer.fit(model, selected_data)   # start and finish traning network
    

    # print('checkpoint_callback.best_model_path')
    # print(checkpoint_callback.best_model_path)
    # model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model, selected_data)    # predict using model
    
    if(model_save_path != None):
        model.save(model_save_path)

    brief = model.validation_brief
    brief.insert(0, "model", model_name)
    detail = model.test_detail
    detail.insert(0, "model", model_name)
    # print("brief: ")
    # print(brief)
    # print("detail: ")
    # print(detail)



    if (os.path.exists(filepath_brief)) :header_brief=False
    else:header_brief=['model','gene_name','type','epoch','alpha1','alpha2','beta','gamma','cost','backgroud_true_cost']

    if (os.path.exists(filepath_detail)) :header_detail=False
    else:header_detail=['model','gene_name','type','s0','u0','u1','s1','alpha','beta','gamma','cost','backgroud_true_cost','alpha_label']
    # print("header")
    # print(header_brief)
    # print(header_detail)
    brief.to_csv(os.path.join(result_path, ('brief_e'+str(max_epoches)+'.csv')),mode='a',header=header_brief)
    detail.to_csv(os.path.join(result_path, ('detail_e'+str(max_epoches)+'.csv')),mode='a',header=header_detail)

    return None

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
        all_data = dataModule(data_path=data_path, type=type, index=index)
    else:
        all_data = feedData()
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
        result = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
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

def train( # use train_thread # change name to velocity estiminate
    datamodule,
    model_path = None, 
    model_number = 1, 
    initial_zoom=2, 
    initial_strech=1, 
    model_save_path = None,
    result_path = None,
    max_epoches=1000, 
    n_jobs=8,
    learning_rate=0.01,
    cost_version=1,
    cost2_cutoff=0.3,
    n_neighbors=30,
    cost1_ratio=0.8,
    optimizer="SGD",
    gene_shape_classify_dict=None,
    with_trace_cost=True,
    with_corrcoef_cost=True):
    '''
    multple jobs
    when model_path is defined, model_number wont be used
    '''

    all_data = datamodule
    data_len = all_data.test_dataset.__len__()

    brief = pd.DataFrame()
    detail = pd.DataFrame()
    filepath_brief= os.path.join(result_path, ('brief_e'+str(max_epoches)+'.csv'))
    filepath_detail=os.path.join(result_path, ('detail_e'+str(max_epoches)+'.csv'))
    if (os.path.exists(filepath_brief)) :os.remove(filepath_brief)
    if (os.path.exists(filepath_detail)) :os.remove(filepath_detail)

    # if model_path != None:
    #     model_names = os.listdir(model_path)
    #     model_number = len(model_names)
    # else:
    #     model_names = list(map(lambda x: "m"+str(x), range(model_number)))

    if model_path == None:
        model_names = list(map(lambda x: "m"+str(x), range(model_number)))

    for model_index in range(model_number):
        #model_name = model_names[model_index]
        model_name = None
        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_train_thread)(
                datamodule = datamodule,
                data_indices=[data_index], 
                model_name=model_name,
                model_path=model_path, 
                result_path=result_path,
                max_epoches=max_epoches,
                initial_zoom=initial_zoom, initial_strech=initial_strech,
                model_save_path=model_save_path,
                learning_rate=learning_rate,
                cost_version=cost_version,
                cost2_cutoff=cost2_cutoff,
                n_neighbors=n_neighbors,
                cost1_ratio=cost1_ratio,
                optimizer=optimizer,
                filepath_brief=filepath_brief,
                filepath_detail=filepath_detail,
                gene_shape_classify_dict=gene_shape_classify_dict,
                with_trace_cost=True,
                with_corrcoef_cost=True)
            for data_index in range(data_len)) #for 循环里执行train_thread
        

        # for i in range(len(result)):
        #     temp_brief, temp_detail = result[i]
        #     #brief = brief.append(temp_brief)
        #     #detail = detail.append(temp_detail)
        #     #print("result_path: "+result_path)
        #     temp_brief.to_csv(os.path.join(result_path, "brief.csv"),mode='a',header=False)
        #     temp_detail.to_csv(os.path.join(result_path, "detail.csv"),mode='a',header=False)
    
    # if save_path != None:
    #     brief.to_csv(os.path.join(save_path, "brief.csv"))
    #     detail.to_csv(os.path.join(save_path, "detail.csv"))


    # brief=pd.read_csv(os.path.join(result_path, ('brief_e'+str(max_epoches)+'.csv')),names=['model','gene_name','type','epoch','alpha1','alpha2','beta','gamma','cost','backgroud_true_cost'])
    # detail=pd.read_csv(os.path.join(result_path, ('detail_e'+str(max_epoches)+'.csv')),names=['model','gene_name','type','s0','u0','u1','s1','alpha','beta','gamma','cost','backgroud_true_cost','alpha_label'])
    brief=pd.read_csv(os.path.join(result_path, ('brief_e'+str(max_epoches)+'.csv')))
    detail=pd.read_csv(os.path.join(result_path, ('detail_e'+str(max_epoches)+'.csv')))

    return brief, detail

# def pretrain_pipeline():
#     '''pretrain pipeline left unchanged. but train function should be used instead here in the next version'''
#     brief, detail = pretrain(type="normal", max_epoches=200, index=2, n_jobs=1, model_save_path = '../../data/model2/normal.pt')

#     show_details(detail, cols=4)
#     show_details(detail, cols=4, backgroud_true_cost=True)

#     sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = False)
#     sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="backgroud_true_cost", legend = False)

def train_simudata_pipeline1():
    from utilities import set_rcParams
    set_rcParams()
    import warnings
    warnings.filterwarnings("ignore")
    from analysis import show_details

    data_path = "../../data/simulation/training.hdf5"
    type = "all"
    index = None
    all_data = dataModule(data_path=data_path, type=type, index=index)

    model_path='../../data/model2'
    max_epoches=1000
    brief, detail = train(all_data, model_path=model_path, max_epoches=max_epoches, n_jobs=8)

    show_details(detail, cols=10)
    show_details(detail, cols=10, backgroud_true_cost=True)
    sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = False)
    sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="backgroud_true_cost", legend = False)

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
    all_data = dataModule(data_path=data_path, type=type, index=None)

    max_epoches=500
    brief, detail = train(all_data, model_path=None, max_epoches=max_epoches, n_jobs=8)

    show_details(detail, cols=4)
    show_details(detail, cols=4, backgroud_true_cost=True)
    sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="cost", legend = 'brief')
    sns.relplot(data=brief, kind="line", col="type", col_wrap=4, hue="gene_name", x="epoch", y="backgroud_true_cost", legend = 'brief')

    brief.to_csv(os.path.join("../../result/result_pl4", "brief_np.csv"))
    detail.to_csv(os.path.join("../../result/result_pl4", "detail_np.csv"))


# old version
# def downsampling(data_df,gene_choice,para,target_amount,step_i,step_j):
#     data_df_downsampled=pd.DataFrame()
#     for gene in gene_choice:
#         data_df_one_gene=data_df[data_df['gene_list']==gene]
#         idx = sampling_adata(data_df_one_gene, 
#                                 para=para,
#                                 target_amount=target_amount,
#                                 step_i=step_i,
#                                 step_j=step_j)
#         data_df_one_gene_downsampled = data_df_one_gene[data_df_one_gene.index.isin(idx)]
#         data_df_downsampled=data_df_downsampled.append(data_df_one_gene_downsampled)
#     return(data_df_downsampled)


def downsampling_embedding(data_df,para,target_amount,step_i,step_j, n_neighbors):
    '''
    Guangyu
    sampling cells by embedding
    return: sampled embedding, the indexs of sampled cells, and the neighbors of sampled cells
    '''
    print("===data_df===")
    print(data_df)
    gene = data_df['gene_list'].drop_duplicates().iloc[0]
    embedding = data_df.loc[data_df['gene_list']==gene][['embedding1','embedding2']]
    idx_downSampling_embedding = sampling_embedding(embedding,
                para=para,
                target_amount=target_amount,
                step_i=step_i,
                step_j=step_j
                )
    embedding_downsampling = embedding.iloc[idx_downSampling_embedding][['embedding1','embedding2']]
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(embedding_downsampling)  # NOTE should support knn in high dimensions
    embedding_knn = nn.kneighbors_graph(mode="connectivity")
    neighbor_ixs = embedding_knn.indices.reshape((-1, n_neighbors + 1))
    return(embedding_downsampling, idx_downSampling_embedding, neighbor_ixs)

def downsampling(data_df, gene_choice, downsampling_ixs):
    '''
    Guangyu
    '''
    data_df_downsampled=pd.DataFrame()
    for gene in gene_choice:
        data_df_one_gene=data_df[data_df['gene_list']==gene]
        data_df_one_gene_downsampled = data_df_one_gene.iloc[downsampling_ixs]
        data_df_downsampled=data_df_downsampled.append(data_df_one_gene_downsampled)

        # plt.scatter(data_df_one_gene['embedding1'], data_df_one_gene['embedding2'])
        # plt.scatter(data_df_one_gene.iloc[downsampling_ixs]['embedding1'], data_df_one_gene.iloc[downsampling_ixs]['embedding2'])
        # plt.scatter(embedding_downsampling.iloc[neighbor_ixs[0,:]]['embedding1'], embedding_downsampling.iloc[neighbor_ixs[0,:]]['embedding2'])
        # plt.scatter(embedding_downsampling.iloc[0]['embedding1'], embedding_downsampling.iloc[0]['embedding2'])
        # plt.show()
    return(data_df_downsampled)

def vaildation_plot(gene,validation_result,save_path_validation):
    plt.figure()
    plt.scatter(validation_result.epoch, validation_result.cost)
    plt.title(gene)
    plt.savefig(save_path_validation)

def select_initial_net(gene, gene_downsampling, data_df):
    '''
    Guangyu
    check if right top conner has cells
    model1 is the model for single kinetic
    model2 is multiple kinetic
    '''
    # gene = 'Rbfox3'
    gene_u_s = gene_downsampling[gene_downsampling.gene_list==gene]
    gene_u_s_full = data_df[data_df.gene_list==gene]
    
    s_max=np.max(gene_u_s.s0)
    u_max = np.max(gene_u_s.u0)
    s_max_90per = 0.9*s_max
    u_max_90per = 0.9*u_max
    
    gene_u_s_full['color'] = 'blue'
    gene_u_s_full.loc[(gene_u_s_full.s0>s_max_90per) & (gene_u_s_full.u0>u_max_90per), 'color'] = 'red'

    # plt.scatter(gene_u_s_full.s0, gene_u_s_full.u0, c = gene_u_s_full['color'])
    # plt.scatter(gene_u_s.s0, gene_u_s.u0)
    # plt.title(gene)
    # plt.show()

    if gene_u_s_full.loc[gene_u_s_full['color']=='red'].shape[0]>0.001*gene_u_s_full.shape[0]:
        model = 'Sulf2'
    else:
        model = 'Ntrk2_e500'
    return(model)


# filter gene
def identify_in_grid(u, s, onegene_u0_s0):
    select_cell =onegene_u0_s0[(onegene_u0_s0[:,0]>u[0]) & (onegene_u0_s0[:,0]<u[1]) & (onegene_u0_s0[:,1]>s[0]) & (onegene_u0_s0[:,1]<s[1]), :]
    if select_cell.shape[0]==0:
        return False
    else:
        return True


def build_grid_list(u_fragment,s_fragment,onegene_u0_s0):
    min_u0 = min(onegene_u0_s0[:,0])
    max_u0 = max(onegene_u0_s0[:,0])
    min_s0 = min(onegene_u0_s0[:,1])
    max_s0 = max(onegene_u0_s0[:,1])
    u0_coordinate=np.linspace(start=min_u0, stop=max_u0, num=u_fragment+1).tolist()
    s0_coordinate=np.linspace(start=min_s0, stop=max_s0, num=s_fragment+1).tolist()
    u0_array = np.array([u0_coordinate[0:(len(u0_coordinate)-1)], u0_coordinate[1:(len(u0_coordinate))]]).T
    s0_array = np.array([s0_coordinate[0:(len(s0_coordinate)-1)], s0_coordinate[1:(len(s0_coordinate))]]).T
    return u0_array, s0_array

def calculate_occupy_ratio(gene_choice,data, u_fragment, s_fragment):
    # data = raw_data2
    ratio = np.empty([len(gene_choice), 1])
    for idx, gene in enumerate(gene_choice):
        print(idx)
        onegene_u0_s0=data[data.gene_list==gene][['u0','s0']].to_numpy()
        u_grid, s_grid=build_grid_list(u_fragment,s_fragment,onegene_u0_s0)
        # occupy = np.empty([1, u_grid.shape[0]*s_grid.shape[0]])
        occupy = 0
        for i, s in enumerate(s_grid):
            for j,u in enumerate(u_grid):
                #print(one_grid)
                if identify_in_grid(u, s,onegene_u0_s0):
                    # print(1)
                    occupy = occupy + 1
        occupy_ratio=occupy/(u_grid.shape[0]*s_grid.shape[0])
        # print('occupy_ratio for '+gene+"="+str(occupy_ratio))
        ratio[idx,0] = occupy_ratio
    ratio2 = pd.DataFrame({'gene_choice': gene_choice, 'ratio': ratio[:,0]})
    return(ratio2)



# if __name__ == "__main__":
#     from utilities import set_rcParams
#     #from utilities import *
#     set_rcParams()
#     import warnings
#     import os
#     import sys
#     import argparse
#     warnings.filterwarnings("ignore")
#     from analysis import show_details, show_details2, show_details_simplify
#     from celldancer_plots import *
#     from sampling import *

#     time_start=time.time()

#     print('\nvelocity_estimate.py version 1.0.0')
#     print('python velocity_estimate.py')
#     print('time_start'+str(time_start))
#     print('')

#     use_all_gene=False
#     plot_trigger=True
#     platform = 'local'
#     if platform == "local":
#         #model_dir='model/model2'
#         #model_dir='model/Ntrk2_e500'
#         model_dir = {"Sulf2": '/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Sulf2/Sulf2.pt', 
#                     "Ntrk2_e500": "/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Ntrk2_e500/Ntrk2_e500.pt"}
#         config = pd.read_csv('/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/config/config_test.txt', sep=';',header=None)
#         data_source = config.iloc[0][0]
#         platform = config.iloc[0][1]
#         epoches=[int(config.iloc[0][2])]
#         num_jobs=int(config.iloc[0][3])
#         learning_rate=float(config.iloc[0][4])
#         cost_version=int(config.iloc[0][5])
#         cost1_ratio=float(config.iloc[0][6])
#         cost2_cutoff=float(config.iloc[0][7])
#         downsample_method=config.iloc[0][8]
#         downsample_target_amount=int(config.iloc[0][9])
#         step_i=int(config.iloc[0][10])
#         step_j=int(config.iloc[0][11])
#         sampling_ratio=float(config.iloc[0][12])
#         n_neighbors=int(config.iloc[0][13])
#         optimizer=config.iloc[0][14] #["SGD","Adam"]
#     elif platform == 'hpc':
#         model_dir="/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/model2"
#         #model_dir="/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/Ntrk2_e500"
#         #model_dir = {'Sulf2': '/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/Sulf2', 
#         #            'Ntrk2_e500': '/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/Ntrk2_e500'}
#         print("---Parameters---")
#         for i in sys.argv:
#             print(i)
#         print("----------------")
#         data_source=sys.argv[1]
#         platform=sys.argv[2]
#         epoches=[int(sys.argv[3])]
#         num_jobs=int(sys.argv[4])
#         learning_rate=float(sys.argv[5])
#         cost_version=int(sys.argv[6])
#         cost1_ratio=float(sys.argv[7])
#         cost2_cutoff=float(sys.argv[8])
#         downsample_method=sys.argv[9]
#         downsample_target_amount=int(sys.argv[10])
#         step_i=int(sys.argv[11])
#         step_j=int(sys.argv[12])
#         sampling_ratio=float(sys.argv[13])
#         n_neighbors=int(sys.argv[14])
#         optimizer=sys.argv[15] #["SGD","Adam"] default->adam
#         full_start=int(sys.argv[16])
#         full_end =int(sys.argv[17])

#     # set data_source
#     if data_source=="scv":
#         if platform=="local":
#             raw_data_path="/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/data/scv_data_full.csv" #["velocyto/data/denGyr.csv","data/scv_data.csv"]
#         elif platform=="hpc":
#             raw_data_path_hpc='/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/scv_data_full.csv'        #["/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/velocyto/data/denGyr.csv","/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/data/scv_data.csv"]
#             raw_data_path=raw_data_path_hpc
#         gene_choice=["Ank","Abcc8","Tcp11","Nfib","Ppp3ca",
#                 "Rbfox3","Cdk1","Gng12","Map1b","Cpe",
#                 "Gnao1","Pcsk2","Tmem163","Pak3","Wfdc15b",
#                 "Nnat","Anxa4","Actn4","Btbd17","Dcdc2a",
#                 "Adk","Smoc1","Mapre3","Pim2","Tspan7",
#                 "Top2a","Rap1b","Sulf2"]
#         #gene_choice=["Sulf2","Top2a","Abcc8"]

#     elif data_source=="denGyr":
#         if platform=="local":
#             raw_data_path="/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/veloNN-main_github/src/velonn/data/denGyr_full.csv" #["data/denGyr.csv","data/scv_data.csv"]
#         elif platform=="hpc":
#             raw_data_path_hpc="/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/denGyr_full.csv"         #["/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/velocyto/data/denGyr.csv","/condo/wanglab/tmhsxl98/Velocity/cell_dancer/data/data/scv_data.csv"]
#             raw_data_path=raw_data_path_hpc
#         # gene_choice=['Ank','Btbd17','Cdk1','Cpe','Gnao1',
#         #             'Gng12','Map1b','Mapre3','Nnat','Ntrk2',
#         #             'Pak3','Pcsk2','Ppp3ca','Rap1b','Rbfox3',
#         #             'Smoc1','Sulf2','Tmem163','Top2a','Tspan7',
#         #             "Pdgfra","Igfbpl1"]
#         # gene_choice=['Ank','Btbd17','Cdk1','Cpe','Gnao1',
#         #             'Gng12','Map1b','Mapre3','Nnat','Ntrk2',
#         #             'Pak3','Pcsk2','Ppp3ca','Rap1b','Rbfox3',
#         #             'Smoc1','Sulf2','Tmem163','Top2a','Tspan7',
#         #             "Pdgfra","Igfbpl1",#
#         #             #Added GENE from page 11 of https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0414-6/MediaObjects/41586_2018_414_MOESM3_ESM.pdf
#         #             "Syngr1","Fam210b","Meg3","Fam19a2","Kcnc3","Dscam"]#"Hagh"] time spent:  46.042665135860446  min
#         # gene_choice=['Adam23','Arid5b','Blcap','Coch','Dcx',
#         #             'Elavl2','Elavl3','Elavl4','Eomes','Eps15',
#         #             'Fam210b','Foxk2','Gpc6','Icam5','Kcnd2',
#         #             'Pfkp','Psd3','Sult2b1','Thy1','Car2','Clip3','Ntrk2','Nnat'] #21 genes
#         gene_choice=['Rimbp2','Dctn3','Psd3','Dcx','Elavl4','Ntrk2']
#         # gene_choice=['Nnat','Ntrk2','Gnao1','Cpe','Ank']
#         # gene_choice=["Ank"]
#         #gene_choice=["Gnao1"]
#         #gene_choice=["Dcx",'Elavl4']
#         #gene_choice=["Ntrk2"]
#         #gene_choice=["Nnat"]
#         #gene_choice=["Kcnc3","Dscam"]
#         #gene_choice=['Elavl4','Eomes','Dcx','Psd3','Sult2b1','Thy1','Car2']
#         #gene_choice=['Elavl4']

#     #### mkdir for output_path with parameters(naming)
#     folder_name=(data_source+
#         "Lr"+str(learning_rate)+
#         "Costv"+str(cost_version)+
#         "C1r"+str(cost1_ratio)+
#         "C2cf"+str(cost2_cutoff)+
#         "Down"+downsample_method+str(downsample_target_amount)+"_"+str(step_i)+"_"+str(step_j)+
#         "Ratio"+str(sampling_ratio)+
#         "N"+str(n_neighbors)+
#         "O"+optimizer)
#     if platform=="local":
#         output_path=("/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/output/detailcsv/"+folder_name+"/")
#         if os.path.isdir(output_path):pass
#         else:os.mkdir(output_path)
#     elif platform=="hpc":
#         output_path=("/condo/wanglab/tmhsxl98/Velocity/cell_dancer/output/detailcsv/adj_e/"+folder_name+"/")
#         if os.path.isdir(output_path):pass
#         else:os.mkdir(output_path)

#     ######################################################
#     ############             Guangyu          ############
#     ######################################################
#     #raw_data_path='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/data/simulation_data/one_gene.csv'

#     load_raw_data=pd.read_csv(raw_data_path,names=['gene_list', 'u0','s0',"clusters",'cellID','embedding1','embedding2'])
#     if use_all_gene: 
#         gene_choice=list(set(load_raw_data.gene_list))
#         gene_choice.sort()
#         gene_choice=gene_choice[full_start: full_end]
#         print('---gene_choice---')
#         print(gene_choice)
    
#     data_df=load_raw_data[['gene_list', 'u0','s0','embedding1','embedding2']][load_raw_data.gene_list.isin(gene_choice)]

#     #!!!!!!!!!!! data_df=load_raw_data[['gene_list', 'u0','s0','cellID','embedding1','embedding2']][load_raw_data.gene_list.isin(gene_choice)]
#     embedding_downsampling, sampling_ixs, neighbor_ixs = downsampling_embedding(data_df,
#                         para=downsample_method,
#                         target_amount=downsample_target_amount,
#                         step_i=step_i,
#                         step_j=step_j,
#                         n_neighbors=n_neighbors)
#     gene_downsampling = downsampling(data_df=data_df, gene_choice=gene_choice, downsampling_ixs=sampling_ixs)

#     _, sampling_ixs_select_model, _ = downsampling_embedding(data_df,
#                         para=downsample_method,
#                         target_amount=downsample_target_amount,
#                         step_i=20,
#                         step_j=20,
#                         n_neighbors=n_neighbors)
#     gene_downsampling_select_model = downsampling(data_df=data_df, gene_choice=gene_choice, downsampling_ixs=sampling_ixs_select_model)


#     gene_shape_classify_dict=pd.DataFrame({'gene_name':gene_choice})
#     gene_shape_classify_dict['model_type']=gene_shape_classify_dict.apply (lambda row: select_initial_net(row.gene_name,gene_downsampling_select_model, data_df), axis=1)
#     if platform=="local":
#         gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Sulf2','model_type_dir']='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Sulf2/Sulf2.pt'
#         gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Ntrk2_e500','model_type_dir']='/Users/shengyuli/OneDrive - Houston Methodist/work/Velocity/veloNN/cellDancer-development/src/model/Ntrk2_e500/Ntrk2_e500.pt'
#     elif platform=="hpc":
#         gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Sulf2','model_type_dir']='/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/Sulf2/Sulf2.pt'
#         gene_shape_classify_dict.loc[gene_shape_classify_dict.model_type=='Ntrk2_e500','model_type_dir']='/condo/wanglab/tmhsxl98/Velocity/cell_dancer/model/Ntrk2_e500/Ntrk2_e500.pt'
    
#     # set fitting data, data to be predicted, and sampling ratio in fitting data
#     feed_data = feedData(data_fit = gene_downsampling, data_predict=data_df, sampling_ratio=sampling_ratio) # default sampling_ratio=0.5
#     #epoches = [5,10,100,300,500]
#     #epoches = [5]
#     #model_save_path="model_Ntrk2/Ntrk2_500.pt"
#     model_save_path=None
#     #model_dir=None
#     for epoch in epoches: # delete
#         #############################################
#         ###########  Fitting and Predict ############
#         #############################################
#         print('-------epoch----------------')
#         print(epoch)
#         #epoch = 1
#         cost_version=1
#         brief, detail = train(feed_data,
#                                 model_path=model_dir, 
#                                 max_epoches=epoch, 
#                                 model_save_path=model_save_path,
#                                 result_path=output_path,
#                                 n_jobs=num_jobs,
#                                 learning_rate=learning_rate,
#                                 cost_version=cost_version,
#                                 cost2_cutoff=cost2_cutoff,
#                                 n_neighbors=n_neighbors,
#                                 cost1_ratio=cost1_ratio,
#                                 optimizer=optimizer,
#                                 gene_shape_classify_dict=gene_shape_classify_dict,
#                                 with_trace_cost=True,
#                                 with_corrcoef_cost=True)

#         #detail.to_csv(output_path+"detail_e"+str(epoch)+".csv")
#         #brief.to_csv(output_path+"brief_e"+str(epoch)+".csv")
#         detail["alpha_new"]=detail["alpha"]/detail["beta"]
#         detail["beta_new"]=detail["beta"]/detail["beta"]
#         detail["gamma_new"]=detail["gamma"]/detail["beta"]
#         detailfinfo="e"+str(epoch)
#         ##########################################
#         ###########       Plot        ############
#         ##########################################
#         if plot_trigger:
#             pointsize=120
#             pointsize=50
#             color_scatter="#95D9EF" #blue
#             alpha_inside=0.3
#             #color_scatter="#DAC9E7" #light purple
#             color_scatter="#8D71B3" #deep purple
#             alpha_inside=0.2
#             color_map="coolwarm"
#             alpha_inside=0.3
#             alpha_inside=1
#             vmin=0
#             vmax=5
#             step_i=20
#             step_j=20

#             for i in gene_choice:
#                 save_path=output_path+i+"_"+"e"+str(epoch)+".pdf"# notice: changed
#                 velocity_plot(detail, [i],detailfinfo,color_scatter,pointsize,alpha_inside,color_map,vmin,vmax,save_path,step_i=step_i,step_j=step_j) # from cell dancer
#                 save_path_validation=output_path+i+"_validation_"+"e"+str(epoch)+".pdf"
#                 if epoch>0:vaildation_plot(gene=i,validation_result=brief[brief["gene_name"]==i],save_path_validation=save_path_validation)
    
#     time_end=time.time()
#     print('time spent: ',(time_end-time_start)/60,' min')   

#     gene_choice=list(set(load_raw_data.gene_list))
#     raw_data2 = load_raw_data[load_raw_data.gene_list.isin(gene_choice)][['gene_list', 'u0','s0']]
#     ratio = calculate_occupy_ratio(gene_choice, raw_data2, 30, 30)
#     ratio.sort_values(by = ['ratio'])


############################
#### old version
############################





# df=pd.DataFrame()
# for gene in gene_choice:
#     cost=brief[brief.gene_name==gene].iloc[-1]['cost']
#     df=df.append(pd.DataFrame({'gene_choice': gene, 'cost': float(cost)}, index=[0]))
# df.sort_values(by = ['cost'])

# df2 = pd.merge(df, ratio, on ='gene_choice')
# df3 = df2.sort_values(by = ['cost'])
# plt.scatter(df3.loc[:(df3.shape[0]-1),'cost'], df3.loc[:(df3.shape[0]-1),'ratio'])


# gene_choice=['Ntrk2']
# gene_choice=['Adam23','Arid5b','Blcap','Coch','Dcx',
#             'Elavl2','Elavl3','Elavl4','Eomes','Eps15',
#             'Fam210b','Foxk2','Gpc6','Icam5','Kcnd2',
#             'Pfkp','Psd3','Sult2b1','Thy1','Car2','Clip3']






# gene_choice=list(set(load_raw_data.gene_list))

# e301=gene_choice[0:200]
# e302=gene_choice[201:400]
# e303=gene_choice[401:600]
# e304=gene_choice[601:800]
# e305=gene_choice[801:1000]
# e306=gene_choice[1001:1200]
# e307=gene_choice[1201:1400]
# e308=gene_choice[1401:1600]
# e309=gene_choice[1601:1800]
# e310_1=gene_choice[1801:2000]
# e310_2=gene_choice[2001:2158]


# filename=['e301','e302','e303','e304','e305','e306','e307','e308','e309','e310_1','e310_2']
# gene_range_list=[e301,e302,e303,e304,e305,e306,e307,e308,e309,e310_1,e310_2]

# df=pd.Dataframe()
# for f,g in zip(filename,gene_range_list):
# 	df_tem = pd.DataFrame({'gene':g})
# 	df_tem['file']=f
# 	df.append(df_tem)

