import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
import seaborn as sns

def get_splice_predict_unsplice_predict(data, dt=0.001, scale=1):
    unsplice = data.unsplice
    splice = data.splice
    alpha = data.alpha
    beta = data.beta
    gamma = data.gamma

    unsplice_predict = unsplice + (alpha - beta*unsplice)*dt
    splice_predict = splice + (beta*unsplice - gamma*splice)*dt
    data['unsplice_predict_true'] = unsplice_predict
    data['splice_predict_true'] = splice_predict
    return data

def get_splice_predict_unsplice_predict_scVelo(data, dt=0.001, scale=1):
    unsplice = data.dynamic_unsplice
    splice = data.dynamic_splice
    alpha = data.alpha
    beta = data.beta
    gamma = data.gamma

    unsplice_predict = unsplice + (alpha - beta*unsplice)*dt
    splice_predict = splice + (beta*unsplice - gamma*splice)*dt
    data['unsplice_predict_true'] = unsplice_predict
    data['splice_predict_true'] = splice_predict
    return data

def subtract(list1, list2):
    difference = []
    zip_object = zip(list1, list2)
    for list1_i, list2_i in zip_object:
        difference.append(list1_i-list2_i)
    return np.array(difference)

def cosine_similarity(unsplice, splice, unsplice_predict_t, splice_predict_t, unsplice_predict, splice_predict):
    """Cost function
    Return:
        list of cosine distance and a list of the index of the next cell
    """
    # Velocity from (unsplice, splice) to (unsplice_predict, splice_predict)
    # unsplice, splice, unsplice_predict_t, splice_predict_t, unsplice_predict, splice_predict =multiple_path['unsplice'], multiple_path['splice'], multiple_path['unsplice_predict'], multiple_path['splice_predict'], multiple_path_back_ground_true2['unsplice_predict_true'], multiple_path_back_ground_true2['splice_predict_true']

        
    uv = subtract(unsplice_predict.to_list(), unsplice.to_list())
    sv = subtract(splice_predict.to_list(), splice.to_list())
    uv_t = subtract(unsplice_predict_t.to_list(), unsplice.to_list())
    sv_t = subtract(splice_predict_t.to_list(), splice.to_list())
    cos = []
    for i in range(len(unsplice)):
        cos.append(1 - spatial.distance.cosine([uv[i], sv[i]], [uv_t[i], sv_t[i]]))
    return cos


def cosine_similarity_scVelo(unsplice, splice, unsplice_predict_t, splice_predict_t, unsplice_predict, splice_predict):
    """Cost function
    Return:
        list of cosine distance and a list of the index of the next cell
    """
    # Velocity from (unsplice, splice) to (unsplice_predict, splice_predict)
    # unsplice, splice, unsplice_predict_t, splice_predict_t, unsplice_predict, splice_predict =predict.dynamic_unsplice, predict.dynamic_splice, observe.unsplice_predict_true, observe.splice_predict_true, predict.dynamic_unsplice_predict, predict.dynamic_splice_predict

        
    uv = unsplice_predict.to_list()
    sv = splice_predict.to_list()
    uv_t = subtract(unsplice_predict_t.to_list(), unsplice.to_list())
    sv_t = subtract(splice_predict_t.to_list(), splice.to_list())
    cos = []
    for i in range(len(unsplice)):
        cos.append(1 - spatial.distance.cosine([uv[i], sv[i]], [uv_t[i], sv_t[i]]))
    return cos

def caluculate_cosin(predic, observe):
    observe = get_splice_predict_unsplice_predict(observe)
    cos = cosine_similarity(predic.unsplice, predic.splice, predic.unsplice_predict, predic.splice_predict, observe.unsplice_predict_true, observe.splice_predict_true)
    return cos

def caluculate_cosin_scVelo(predict, observe):
    # predict, observe=simulation100_scVelo, simulation100_simulate
    observe = get_splice_predict_unsplice_predict(observe)
    cos_dynamic = cosine_similarity_scVelo(predict.dynamic_unsplice, predict.dynamic_splice, observe.unsplice_predict_true, observe.splice_predict_true, predict.dynamic_unsplice_predict, predict.dynamic_splice_predict)
    cos_static = cosine_similarity_scVelo(predict.static_unsplice, predict.static_splice, observe.unsplice_predict_true, observe.splice_predict_true, predict.static_unsplice_predict, predict.static_splice_predict)
    return cos_dynamic, cos_static


def get_esitimate(one_test_true, one_test):
    one_test_true_possitive = one_test_true[one_test_true['alpha']==one_test_true.loc[0,'alpha']]
    idx = one_test_true_possitive.index
    alpha_possitive = one_test_true.loc[0,'alpha']
    beta_possitive = one_test_true.loc[0,'beta']
    gamma_possitive = one_test_true.loc[0,'gamma']
    test_possitive = one_test.iloc[idx]
    alpha_possitive_predict = test_possitive['alpha'].mean()
    beta_possitive_predict = test_possitive['beta'].mean()
    gamma_possitive_predict = test_possitive['gamma'].mean()
    possitive = [alpha_possitive/beta_possitive, beta_possitive/beta_possitive, gamma_possitive/beta_possitive]
    possitive_predict = [alpha_possitive_predict/beta_possitive_predict, beta_possitive_predict/beta_possitive_predict, gamma_possitive_predict/beta_possitive_predict]

    one_test_true_zero = one_test_true[one_test_true['alpha']==one_test_true.loc[1100,'alpha']]
    idx = one_test_true_zero.index
    alpha_zero = one_test_true.loc[1100,'alpha']
    beta_zero = one_test_true.loc[1100,'beta']
    gamma_zero = one_test_true.loc[1100,'gamma']
    test_zero = one_test.iloc[idx]
    alpha_zero_predict = test_zero['alpha'].mean()
    beta_zero_predict = test_zero['beta'].mean()
    gamma_zero_predict = test_zero['gamma'].mean()
    zero = [alpha_zero/beta_zero, beta_zero/beta_zero, gamma_zero/beta_zero]
    zero_predict = [alpha_zero_predict/beta_zero_predict, beta_zero_predict/beta_zero_predict, gamma_zero_predict/beta_zero_predict]
    return(possitive, possitive_predict, zero, zero_predict)


def get_similarity_cellDancer(ratio,detail_input_path,raw_input_path,type,foldername_para,path):
    """
    esitimate cellDancer
    """
    multiple_path = pd.read_csv(detail_input_path+'ratio'+str(ratio)+foldername_para+'/celldancer_estimation.csv')
    
    multiple_path_back_ground_true = pd.read_csv(raw_input_path+type+'_path_'+path+'_1000__R'+str(ratio)+'.csv')
    
    cos_all = pd.DataFrame()
    for i in range(1000):
        if i%10==0: print(i)
        # simulation100_cellDancer = multiple_path[multiple_path['gene_name']=='simulation'+str(i).zfill(3)]
        simulation100_cellDancer = multiple_path[multiple_path['gene_name']=='simulation'+str(i)]

        #simulation100_simulate = multiple_path_back_ground_true[multiple_path_back_ground_true['gene_name']=='simulation'+str(i).zfill(3)]
        simulation100_simulate = multiple_path_back_ground_true[multiple_path_back_ground_true['gene_name']=='simulation'+str(i)]
        cos = caluculate_cosin(simulation100_cellDancer, simulation100_simulate)
        cellDancer = pd.DataFrame({'similarity':cos, 'method': 'cellDancer', 'ratio':ratio, 'gene': i})
        cos_all = pd.concat([cos_all, cellDancer])
        # print(cos_all)
        # plt.hist(d)
        # plt.show()
        # plt.hist(s)
        # plt.show()
    cos_all.loc[:,'cellID']=list(cos_all.index)
    return(cos_all)


def get_similarity_scVelo(ratio,scv_result_input_path,raw_input_path,type,path):
    """
    esitimate scVelo
    """
    multiple_path_scVelo = pd.read_csv(scv_result_input_path+'scvelo_result_'+type+'_path__splice_unsplice_splice_predict_unsplice_predict_dynamic_and_steady_df_'+str(ratio)+'.csv')

    multiple_path_back_ground_true = pd.read_csv(raw_input_path+type+'_path_'+path+'_1000__R'+str(ratio)+'.csv')
    scVelo1 = pd.DataFrame()
    scVelo2 = pd.DataFrame()

    # for i in range(2):
    for i in range(1000):
        if i%10==0:print(i)
        # simulation100_scVelo = multiple_path_scVelo[multiple_path_scVelo['gene_name']=='simulation'+str(i).zfill(3)]
        # simulation100_simulate = multiple_path_back_ground_true[multiple_path_back_ground_true['gene_name']=='simulation'+str(i).zfill(3)]

        simulation100_scVelo = multiple_path_scVelo[multiple_path_scVelo['gene_name']=='simulation'+str(i)]
        simulation100_simulate = multiple_path_back_ground_true[multiple_path_back_ground_true['gene_name']=='simulation'+str(i)]

        d, s = caluculate_cosin_scVelo(simulation100_scVelo, simulation100_simulate)
        scVelo1_tmp = pd.DataFrame({'similarity':d, 'method': 'dynamic', 'ratio':ratio, 'gene': i})
        scVelo1 = pd.concat([scVelo1, scVelo1_tmp])
        scVelo2_tmp = pd.DataFrame({'similarity':s, 'method': 'static', 'ratio':ratio, 'gene': i})
        scVelo2 = pd.concat([scVelo2, scVelo2_tmp])
    # scVelo1 = pd.DataFrame({'similarity':dynamic, 'method': 'dynamic', 'ratio':ratio})
    # scVelo2 = pd.DataFrame({'similarity':static, 'method': 'static', 'ratio': ratio})
    scVelo = pd.concat([scVelo1, scVelo2])
    scVelo.loc[:,'cellID']=list(scVelo.index)
    return(scVelo)


def plot_box_error(combined_error_df,sim_cutoff):
    plt.figure()

    plt.rcParams["figure.figsize"] = (3.5,3)

    plt.title('sim_cutoff '+str(sim_cutoff))
    sns.color_palette("flare", as_cmap=True)
    ax = sns.boxplot(x="ratio", y="error_rate", hue='method', data=combined_error_df,linewidth=0.3,showfliers = False)
    ax = sns.stripplot(x="ratio", y="error_rate", hue='method', data=combined_error_df, jitter=True, split=True, linewidth=0.5,size=0.3,alpha=.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig('/Users/wanglab/Documents/ShengyuLi/Velocity/data/simulation/data/wing_path/wing_path_20220218/analysis_result/'+'error_sim'+str(sim_cutoff)+'.pdf')
