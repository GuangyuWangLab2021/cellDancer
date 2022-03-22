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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
if __name__ == "__main__":
    sys.path.append('.')
    from sampling import *
else:
    try:
        from .sampling import *
    except ImportError:
        from sampling import *
        
class L2Module(nn.Module): #can change name #set the shape of the net
    '''
    network structure
    '''
    def __init__(self, h1, h2, ratio):#all init cannot change name
        super().__init__()
        self.l1 = nn.Linear(2, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, 3)
        self.ratio = ratio

    def forward(self, u0, s0, alpha0, beta0, gamma0, dt):#better not change name
        input = torch.tensor(np.array([np.array(u0), np.array(s0)]).T)
        x = self.l1(input)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        x = self.l3(x)
        output = torch.sigmoid(x)
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
            # print('----------cost 3 is running')
            return(cost)

        u1 = u0 + (alphas - beta*u0)*dt
        s1 = s0 + (beta*u0 - gamma*s0)*dt
        if self.ratio ==0:
            cost = 0
        else:
            cost = corrcoef_cost(alphas, u0, beta, s0)
        return u1, s1, alphas, beta, gamma, cost

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
                           cost2_cutoff=None,
                           trace_cost_ratio=None,
                           corrcoef_cost_ratio=None):
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

        # cosin_cost_ratio=0.5
        # trace_cost_ratio=0.2
        # corrcoef_cost_ratio=0.3
        
        if trace_cost_ratio==0 and corrcoef_cost_ratio==0:
            # print('------running - just cosin cost, trace_cost_ratio==0 and corrcoef_cost_ratio==0')
            cost1 = cosine_similarity(u0, s0, u1, s1, indices)[0]
            cost_fin=torch.mean(cost1)
        else:
            # print('trace_cost_ratio!=0 or corrcoef_cost_ratio!=0')

            # cosin cost
            cost1,idx = cosine_similarity(u0, s0, u1, s1, indices)
            cost1_normalize=(cost1-torch.min(cost1))/torch.max(cost1)
            cost1_mean = torch.mean(cost1_normalize)

            # trace cost
            if trace_cost_ratio>0:
                # print('------running - trace cost')
                cost2 = trace_cost(u0, s0, u1, s1, idx,"v2")
                cost2_normalize=(cost2-torch.min(cost2))/torch.max(cost2)
                cost2_mean = torch.mean(cost2_normalize)
                cost2_relu=(max((cost2_mean-cost2_cutoff), 0))

            # corrcoef cost
            if corrcoef_cost_ratio>0:
                # print('------running - corrcoef_cost')
                cost3=cost3

            # sum cost
            cosin_cost_ratio=1-trace_cost_ratio-corrcoef_cost_ratio
            cost_fin = cosin_cost_ratio*cost1_mean + \
                       trace_cost_ratio*cost2_relu + \
                       corrcoef_cost_ratio*cost3
            

        # if cost_version==1:
        #     cost1 = cosine_similarity(u0, s0, u1, s1, indices)[0]
        #     cost_fin=torch.mean(cost1)

        # elif cost_version==2:
        #     cost1,idx = cosine_similarity(u0, s0, u1, s1, indices)
        #     cost2 = trace_cost(u0, s0, u1, s1, idx,"v2")

        #     cost1_normalize=(cost1-torch.min(cost1))/torch.max(cost1)
        #     cost2_normalize=(cost2-torch.min(cost2))/torch.max(cost2)

        #     cost1_mean = torch.mean(cost1_normalize)
        #     cost2_mean = torch.mean(cost2_normalize)
        #     if cost2_mean<cost2_cutoff:           
        #         cost_mean_v2 = cost1_mean
        #     else:
        #         cost_mean_v2 = cost1_ratio*cost1_mean + (1-cost1_ratio)*(cost2_mean-cost2_cutoff) # relu activate
        #     cost_fin=cost_mean_v2
        # elif cost_version==3:
        #     cost1,idx = cosine_similarity(u0, s0, u1, s1, indices)
        #     cost2 = trace_cost(u0, s0, u1, s1, idx,"v2")
            
            
        #     cost1_normalize=(cost1-torch.min(cost1))/torch.max(cost1)
        #     cost2_normalize=(cost2-torch.min(cost2))/torch.max(cost2)
        #     # print(cost1_normalize[0:10])
        #     # print(cost2_normalize[0:10])
        #     # print(cost3)

        #     cost1_mean = torch.mean(cost1_normalize)
        #     cost2_mean = torch.mean(cost2_normalize)

        #     ratio2 = 0.3
        #     cost_mean_v2 = cost1_ratio*cost1_mean + (1-cost1_ratio-ratio2)*(max((cost2_mean-cost2_cutoff), 0)) + ratio2*cost3
            
        #     cost_fin=cost_mean_v2
        return cost_fin, u1, s1, alphas, beta, gamma # to do


    def summary_para_validation(self, cost_mean): # before got detail; build df
        brief = pd.DataFrame({'cost': cost_mean}, index=[0])
        return(brief)



    def summary_para(self, u0, s0, u1, s1, alphas, beta, gamma, cost): # before got detail; build df
        detail = pd.merge(pd.DataFrame(s0, columns=['s0']), pd.DataFrame(u0, columns=['u0']), left_index=True, right_index=True) 
        detail['s1'] = s1
        detail['u1'] = u1
        detail['alpha'] = alphas
        detail['beta'] = beta
        detail['gamma'] = gamma
        detail['cost'] = cost
        return detail


class ltModule(pl.LightningModule):
    '''
    taining network using loss function "stochasticModule"
    '''
    def __init__(self, 
                backbone, 
                pretrain=False, 
                initial_zoom=None, 
                initial_strech=None,
                learning_rate=None,
                cost2_cutoff=None,
                optimizer=None,
                trace_cost_ratio=None,
                corrcoef_cost_ratio=None,
                cost_type=None,
                average_cost_window_size=None,
                smooth_weight=None):
        super().__init__()
        self.backbone = backbone   # load network; caculate loss function; predict u1 s1 ("DynamicModule")
        self.pretrain = pretrain   # 
        self.validation_brief = pd.DataFrame() # cost score (every 10 times training)
        self.test_detail = None
        self.test_brief = None
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
        self.average_cost_window_size=average_cost_window_size # will be used only cost_tpye=='average'
        self.cost_window=[]
        self.smooth_weight=smooth_weight
        
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
        # print('-----------training_step------------')
        u0s, s0s, u1ts, s1ts, true_alphas, true_betas, true_gammas, gene_names, types, u0maxs, s0maxs, embedding1s, embedding2s = batch
        u0, s0, u1t, s1t, _, _, _, _, _, u0max, s0max, embedding1, embedding2  = u0s[0], s0s[0], u1ts[0], s1ts[0], true_alphas[0], true_betas[0], true_gammas[0], gene_names[0], types[0], u0maxs[0], s0maxs[0], embedding1s[0], embedding2s[0]
        
        umax = u0max
        smax = s0max
        alpha0 = np.float32(umax*self.initial_zoom)
        beta0 = np.float32(1.0)
        gamma0 = np.float32(umax/smax*self.initial_strech)


        cost, u1, s1, alphas, beta, gamma = self.backbone.velocity_calculate(u0, s0, alpha0, beta0, gamma0,embedding1,embedding2,self.current_epoch,cost2_cutoff=self.cost2_cutoff,trace_cost_ratio=self.trace_cost_ratio,corrcoef_cost_ratio=self.corrcoef_cost_ratio) # for real dataset, u0: np.array(u0 for cells selected by __getitem__) to a tensor in pytorch, s0 the same as u0
        # print("cost for training_step: "+str(cost))
        # cost_mean = torch.mean(cost)    # cost: a list of cost of each cell for a given gene
        if self.cost_type=='average':
            print('-----STOP: using average cost-----')
            # keep the window len <= check_n_epoch
            if len(self.cost_window)<self.average_cost_window_size:
                self.cost_window.append(cost)
            else:
                self.cost_window.pop(0)
                self.cost_window.append(cost)
            self.get_loss = torch.mean(torch.stack(self.cost_window))
            self.log("loss", self.get_loss) # used for early stop. controled by log_every_n_steps
            
        elif self.cost_type=='median':
            print('-----STOP: using average cost-----')
            # keep the window len <= check_n_epoch
            if len(self.cost_window)<self.average_cost_window_size:
                self.cost_window.append(cost)
            else:
                self.cost_window.pop(0)
                self.cost_window.append(cost)
            self.get_loss = torch.median(torch.stack(self.cost_window))
            self.log("loss", self.get_loss) # used for early stop. controled by log_every_n_steps
            
        elif self.cost_type=='smooth':
            print('-----STOP: using smooth cost-----')
            if self.get_loss==1000:
                self.get_loss=cost
            smoothed_val = cost * self.smooth_weight + (1 - self.smooth_weight) * self.get_loss  # Calculate smoothed value
            self.get_loss = smoothed_val  

            self.log("loss", self.get_loss)
        else:
            print('-----STOP: not using cost-----')
            self.get_loss = cost
            self.log("loss", self.get_loss) # used for early stop. controled by log_every_n_steps
        
        
        return {
            "loss": cost,
            "beta": beta.detach(),
            "gamma": gamma.detach()
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
        #self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def validation_step(self, batch, batch_idx):# name cannot be changed 
        '''
        predict u1, s1 on the training dataset 
        caculate every 10 times taining
        '''
        
        # print(self.get_loss)
        ###############################################
        #########       add embedding         #########
        ###############################################
        # print('-----------validation_step------------')
        u0s, s0s, u1ts, s1ts, true_alphas, true_betas, true_gammas, gene_names, types, u0maxs, s0maxs, embedding1s, embedding2s = batch
        u0, s0, u1t, s1t, _, _, _, gene_name, type, u0max, s0max, embedding1, embedding2  = u0s[0], s0s[0], u1ts[0], s1ts[0], true_alphas[0], true_betas[0], true_gammas[0], gene_names[0], types[0], u0maxs[0], s0maxs[0], embedding1s[0], embedding2s[0]
        if self.current_epoch!=0:
            cost = self.get_loss.data.numpy()
            # print("cost_mean: "+str(cost_mean))
            # brief = self.backbone.summary_para_validation(cost) # training cost
            brief = self.backbone.summary_para_validation(cost) # average cost
            brief.insert(0, "gene_name", gene_name)
            brief.insert(1, "epoch", self.current_epoch)
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
        cost, u1, s1, alphas, beta, gamma = self.backbone.velocity_calculate(u0, s0, alpha0, beta0, gamma0,embedding1,embedding2,self.current_epoch,cost2_cutoff=self.cost2_cutoff,trace_cost_ratio=self.trace_cost_ratio,corrcoef_cost_ratio=self.corrcoef_cost_ratio)
        self.test_detail= self.backbone.summary_para(
            u0, s0, u1.data.numpy(), s1.data.numpy(), 
            alphas.data.numpy(), beta.data.numpy(), gamma.data.numpy(), 
            cost.data.numpy())
        
        self.test_detail.insert(0, "gene_name", gene_name)


class getItem(Dataset): # TO DO: Change to a suitable name
    def __init__(self, data_fit=None, data_predict=None,datastatus="predict_dataset", sampling_ratio=1,auto_norm_u_s=True,binning=False): #point_number=600 for training
        self.data_fit=data_fit
        self.data_predict=data_predict
        self.datastatus=datastatus
        self.sampling_ratio=sampling_ratio
        self.gene_list=list(set(data_fit.gene_list))
        self.auto_norm_u_s=auto_norm_u_s
        self.norm_max_u0=None
        self.norm_max_s0=None
        self.binning=binning

    def __len__(self):# name cannot be changed 
        return len(self.gene_list) # gene count

    def __getitem__(self, idx):# name cannot be changed
        gene_name = self.gene_list[idx]


        if self.datastatus=="fit_dataset":
            data_fitting=self.data_fit[self.data_fit.gene_list==gene_name] # u0 & s0 for cells for one gene
            if self.binning==True:    # select cells to train using binning methods
                u0 = data_fitting.u0
                s0 = data_fitting.s0
                u0max_fit = np.float32(max(u0))
                s0max_fit = np.float32(max(s0))
                u0 = np.round(u0/u0max_fit, 2)*u0max_fit
                s0 = np.round(s0/s0max_fit, 2)*s0max_fit
                upoints = np.unique(np.array([u0, s0]), axis=1)
                u0 = upoints[0]
                s0 = upoints[1]
                data_fitting = pd.DataFrame({'gene_list':gene_name,'u0':u0, 's0':s0,'embedding1':u0,'embedding2':s0})
                # print(data_fitting.shape) #guangyu
        
        # TODO: OPtimize #############
            # random sampling ratio selection
            if self.sampling_ratio==1:
                data=data_fitting
            elif (self.sampling_ratio<1) & (self.sampling_ratio>0):
                data=data_fitting.sample(frac=self.sampling_ratio)  # select cells to train using random methods
                # print('sampling_ratio:'+str(self.sampling_ratio))
                # print(data.shape)
            else:
                print('sampling ratio is wrong!')
        elif self.datastatus=="predict_dataset":
            data_pred=self.data_predict[self.data_predict.gene_list==gene_name] # u0 & s0 for cells for one gene
            data=data_pred
        # #############
            
        data_pred=self.data_predict[self.data_predict.gene_list==gene_name] # u0 & s0 for cells for one gene
        #print('gene_name: '+gene_name)
        #print(data_pred)
        # 未来可能存在的问题：训练cell，和predict cell的u0和s0重大，不match，若不match？（当前predict cell 里是包含训练cell的，所以暂定用predict的u0max和s0max，如果不包含怎么办？还是在外面算好再传参？）
        u0max = np.float32(max(data_pred["u0"]))
        s0max = np.float32(max(data_pred["s0"]))
        # set u0 array and s0 array
        u0 = np.array(data.u0.copy().astype(np.float32))
        s0 = np.array(data.s0.copy().astype(np.float32))
        # plt.figure()
        # plt.title('before norm')
        # plt.scatter(s0,u0)
        if self.auto_norm_u_s:
            u0=u0/u0max
            s0=s0/s0max
        # plt.figure()
        # plt.title('after norm')
        # plt.scatter(s0,u0)
        # below will be deprecated later since this is for the use of realdata
        u1 = np.float32(0)
        s1 = np.float32(0)
        alpha = np.float32(0)
        beta = np.float32(0)
        gamma = np.float32(0)
        type = "real"

        # add embedding (Guangyu)
        embedding1 = np.array(data.embedding1.copy().astype(np.float32))
        embedding2 = np.array(data.embedding2.copy().astype(np.float32))
        # print(embedding1)

        return u0, s0, u1, s1, alpha, beta, gamma, gene_name, type, u0max, s0max, embedding1, embedding2



class feedData(pl.LightningDataModule): #change name to feedData
    '''
    load training and test data
    '''
    def __init__(self, data_fit=None, data_predict=None,sampling_ratio=1,auto_norm_u_s=True,binning=False):
        super().__init__()

        #change name to fit
        self.training_dataset = getItem(data_fit=data_fit, data_predict=data_predict,datastatus="fit_dataset", sampling_ratio=sampling_ratio,auto_norm_u_s=auto_norm_u_s,binning=binning)
        
        #change name to predict
        self.test_dataset = getItem(data_fit=data_fit, data_predict=data_predict,datastatus="predict_dataset", sampling_ratio=sampling_ratio,auto_norm_u_s=auto_norm_u_s)

    def subset(self, indices):
        import copy
        temp = copy.copy(self)
        temp.training_dataset = Subset(self.training_dataset, indices)
        temp.test_dataset = Subset(self.test_dataset, indices)
        return temp

    def train_dataloader(self):# name cannot be changed 
        return DataLoader(self.training_dataset,num_workers=0)
    def val_dataloader(self):# name cannot be changed 
        return DataLoader(self.training_dataset,num_workers=0)
    def test_dataloader(self):# name cannot be changed 
        return DataLoader(self.test_dataset,num_workers=0)

def _train_thread(datamodule, 
                    data_indices, 
                    result_path=None,
                    n_neighbors=None, 
                    max_epoches=None, 
                    check_n_epoch=None, 
                    initial_zoom=None, 
                    initial_strech=None, 
                    model_save_path=None,
                    learning_rate=None,
                    cost2_cutoff=None,
                    optimizer=None,
                    filepath_brief=None,
                    filepath_detail=None,
                    trace_cost_ratio=None,
                    corrcoef_cost_ratio=None,
                 auto_norm_u_s=None,
                  cost_type=None,
                  average_cost_window_size=None,
                 patience=None,
                 smooth_weight=None,
                 ini_model='normal'):
    # print("train thread---------")
    import random
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    backbone = stochasticModule(L2Module(100, 100, corrcoef_cost_ratio), n_neighbors)    # iniate network (L2Module) and loss function (DynamicModule)
    model = ltModule(backbone=backbone, 
                    pretrain=False, 
                    initial_zoom=initial_zoom, 
                    initial_strech=initial_strech,
                    learning_rate=learning_rate,
                    cost2_cutoff=cost2_cutoff,
                    optimizer=optimizer,
                    trace_cost_ratio=trace_cost_ratio,
                    corrcoef_cost_ratio=corrcoef_cost_ratio,
                    cost_type=cost_type,
                    average_cost_window_size=average_cost_window_size,
                    smooth_weight=smooth_weight,)


    # print("indices", data_indices)
    selected_data = datamodule.subset(data_indices)  # IMPORTANT: 这个subset对应realdata.py里的每一个get_item块，
    #因为从前如果不用subset，就会训练出一个网络，对应不同gene的不同alpha，beta，gamma；
    #但是，如果使用subset，就分块训练每个基因不同网络，效果变好

    u0, s0, u1, s1, alpha, beta, gamma, this_gene_name, type, u0max, s0max, embedding1, embedding2=selected_data.training_dataset.__getitem__(0)
    
    # data_df=load_raw_data[['gene_list', 'u0','s0','cellID','embedding1','embedding2']][load_raw_data.gene_list.isin(gene_choice)]
    data_df=pd.DataFrame({'u0':u0,'s0':s0,'embedding1':embedding1,'embedding2':embedding2})
    data_df['gene_list']=this_gene_name
    print(this_gene_name)
    _, sampling_ixs_select_model, _ = downsampling_embedding(data_df,
                        para='neighbors',
                        target_amount=0,
                        step_i=20,
                        step_j=20,
                        n_neighbors=n_neighbors)
    gene_downsampling=downsampling(data_df=data_df, gene_choice=[this_gene_name], downsampling_ixs=sampling_ixs_select_model)
    if ini_model=='normal':
        model_path='/Users/wanglab/Documents/ShengyuLi/Velocity/bin/cellDancer-development_20220128/src/model/normal/normal.pt'
    if ini_model=='sulf':
        model_path='/Users/wanglab/Documents/ShengyuLi/Velocity/bin/cellDancer-development_20220128/src/model/Sulf2/Sulf2.pt'
    else:
        model_path=select_initial_net(this_gene_name, gene_downsampling, data_df)
    model.load(model_path)

    early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.0, patience=patience,mode='min')
    checkpoint_callback = ModelCheckpoint(monitor="loss",
                                          dirpath='output/callback_checkpoint/',
                                          save_top_k=1,
                                          mode='min',
                                          auto_insert_metric_name=True
                                          )

    if check_n_epoch is None:
        print('not using early stop')
        trainer = pl.Trainer(
            max_epochs=max_epoches, progress_bar_refresh_rate=0, reload_dataloaders_every_n_epochs=1, 
            logger = False,
            checkpoint_callback = False,
            weights_summary=None,
            )
    else:
        print('using early stop')
        trainer = pl.Trainer(
            max_epochs=max_epoches, progress_bar_refresh_rate=0, reload_dataloaders_every_n_epochs=1, 
            logger = False,
            checkpoint_callback = False,
            check_val_every_n_epoch = int(check_n_epoch),
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

    trainer.test(model, selected_data)    # predict using model
    
    if(model_save_path != None):
        model.save(model_save_path)

    brief = model.validation_brief
    detail = model.test_detail

    if (os.path.exists(filepath_brief)) :header_brief=False
    # else:header_brief=['model','gene_name','type','epoch','alpha1','alpha2','beta','gamma','cost','backgroud_true_cost']
    else:header_brief=['gene_name','epoch','cost']

    if auto_norm_u_s:
        detail.s0=detail.s0*s0max
        detail.u0=detail.u0*u0max
        detail.s1=detail.s1*s0max
        detail.u1=detail.u1*u0max
        detail.beta=detail.beta*u0max
        detail.gamma=detail.gamma*s0max
    
    if (os.path.exists(filepath_detail)) :header_detail=False
    else:header_detail=['gene_name','s0','u0','s1','u1','alpha','beta','gamma','cost']
    brief.to_csv(os.path.join(result_path, ('brief_e'+str(max_epoches)+'.csv')),mode='a',header=header_brief)
    detail.to_csv(os.path.join(result_path, ('detail_e'+str(max_epoches)+'.csv')),mode='a',header=header_detail)

    return None

def downsample_raw(load_raw_data,downsample_method,n_neighbors_downsample,downsample_target_amount,auto_downsample,auto_norm_u_s,sampling_ratio,step_i,step_j,gene_choice=None,binning=False):
    
    if gene_choice is None:
        data_df=load_raw_data[['gene_list', 'u0','s0','embedding1','embedding2']]
    else:
        data_df=load_raw_data[['gene_list', 'u0','s0','embedding1','embedding2']][load_raw_data.gene_list.isin(gene_choice)]

    # data_df.s0=data_df.s0/max(data_df.s0)
    # data_df=load_raw_data[['gene_list', 'u0','s0','cellID','embedding1','embedding2']][load_raw_data.gene_list.isin(gene_choice)]
    
    if auto_downsample:
        _, sampling_ixs, _ = downsampling_embedding(data_df,
                            para=downsample_method,
                            target_amount=downsample_target_amount,
                            step_i=step_i,
                            step_j=step_j,
                            n_neighbors=n_neighbors_downsample)
        gene_downsampling = downsampling(data_df=data_df, gene_choice=gene_choice, downsampling_ixs=sampling_ixs)
        

        feed_data = feedData(data_fit = gene_downsampling, data_predict=data_df, sampling_ratio=sampling_ratio,auto_norm_u_s=auto_norm_u_s,binning=binning) # default 
    else:
        feed_data = feedData(data_fit = data_df, data_predict=data_df, sampling_ratio=sampling_ratio,auto_norm_u_s=auto_norm_u_s,binning=binning) # default 

    # set fitting data, data to be predicted, and sampling ratio in fitting data
    

    return(feed_data)


def train( # use train_thread # change name to velocity estiminate
    load_raw_data,
    gene_choice=None,
    downsample_method='neighbors',
    n_neighbors_downsample=30,
    downsample_target_amount=0,
    auto_downsample=True,
    auto_norm_u_s=True,
    sampling_ratio=0.125,
    step_i=200,
    step_j=200,
    initial_zoom=2, 
    initial_strech=1, 
    model_save_path = None,
    result_path = None,
    check_n_epoch=5,
    max_epoches=200, 
    n_jobs=os.cpu_count(),
    learning_rate=0.001,
    cost2_cutoff=0.3,
    n_neighbors=30,
    optimizer="Adam",
    trace_cost_ratio=0,
    corrcoef_cost_ratio=0,
    cost_type='average',
    average_cost_window_size=10, 
    patience=3,
    smooth_weight=0.9,
    binning=False):
    '''
    multple jobs
    when model_path is defined, model_number wont be used
    '''
    
    datamodule=downsample_raw(load_raw_data,downsample_method,n_neighbors_downsample,downsample_target_amount,auto_downsample,auto_norm_u_s,sampling_ratio,step_i,step_j,gene_choice=gene_choice,binning=binning)
    
    if check_n_epoch=='None':check_n_epoch=None
    all_data = datamodule
    data_len = all_data.test_dataset.__len__()
    brief = pd.DataFrame()
    detail = pd.DataFrame()
    filepath_brief= os.path.join(result_path, ('brief_e'+str(max_epoches)+'.csv'))
    filepath_detail=os.path.join(result_path, ('detail_e'+str(max_epoches)+'.csv'))
    if (os.path.exists(filepath_brief)) :os.remove(filepath_brief)
    if (os.path.exists(filepath_detail)) :os.remove(filepath_detail)

    result = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_train_thread)(
            datamodule = datamodule,
            data_indices=[data_index], 
            result_path=result_path,
            max_epoches=max_epoches,
            check_n_epoch=check_n_epoch,
            initial_zoom=initial_zoom, initial_strech=initial_strech,
            model_save_path=model_save_path,
            learning_rate=learning_rate,
            cost2_cutoff=cost2_cutoff,
            n_neighbors=n_neighbors,
            optimizer=optimizer,
            filepath_brief=filepath_brief,
            filepath_detail=filepath_detail,
            trace_cost_ratio=trace_cost_ratio,
            corrcoef_cost_ratio=corrcoef_cost_ratio,
            auto_norm_u_s=auto_norm_u_s,
            cost_type=cost_type,
            average_cost_window_size=average_cost_window_size,
            patience=patience,
            smooth_weight=smooth_weight)
        for data_index in range(data_len)) #for 循环里执行train_thread
        
    brief=pd.read_csv(os.path.join(result_path, ('brief_e'+str(max_epoches)+'.csv')))
    detail=pd.read_csv(os.path.join(result_path, ('detail_e'+str(max_epoches)+'.csv')))

    return brief, detail


def downsampling_embedding(data_df,para,target_amount,step_i,step_j, n_neighbors):
    '''
    Guangyu
    sampling cells by embedding
    return: sampled embedding, the indexs of sampled cells, and the neighbors of sampled cells
    '''
    gene = data_df['gene_list'].drop_duplicates().iloc[0]
    embedding = data_df.loc[data_df['gene_list']==gene][['embedding1','embedding2']]
    idx_downSampling_embedding = sampling_embedding(embedding,
                para=para,
                target_amount=target_amount,
                step_i=step_i,
                step_j=step_j
                )
    embedding_downsampling = embedding.iloc[idx_downSampling_embedding][['embedding1','embedding2']]
    n_neighbors = min((embedding_downsampling.shape[0]-2), n_neighbors)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    # print(embedding_downsampling)
    print(embedding_downsampling.shape)
    print(n_neighbors)
    nn.fit(embedding_downsampling)  # NOTE should support knn in high dimensions
    embedding_knn = nn.kneighbors_graph(mode="connectivity")
    neighbor_ixs = embedding_knn.indices.reshape((-1, n_neighbors))
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
    
    
    import pathlib
    model_path=pathlib.Path(__file__).parent.resolve()
    if gene_u_s_full.loc[gene_u_s_full['color']=='red'].shape[0]>0.001*gene_u_s_full.shape[0]:
        model = 'Sulf2'
        
        model_path=os.path.join(model_path,'model','Sulf2','Sulf2.pt')
        # model_path='/Users/wanglab/Documents/ShengyuLi/Velocity/bin/cellDancer-development_20220128/src/model/Sulf2/Sulf2.pt'
    else:
        model = 'Ntrk2_e500'
        model_path=os.path.join(model_path,'model','Ntrk2_e500','Ntrk2_e500.pt')
        # model_path='/Users/wanglab/Documents/ShengyuLi/Velocity/bin/cellDancer-development_20220128/src/model/Ntrk2_e500/Ntrk2_e500.pt'
    return(model_path)

