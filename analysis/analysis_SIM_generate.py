from analysis_SIM_generate_fun import *

######################################
########## SIM -Demos ################
######################################
def SIM_demos():
    import sys

    sys.path.append('.')

    # from utilities import set_rcParams
    # set_rcParams()

    #forward
    data = generate_forward(gene_num=6, alpha=15, beta=20, gamma=18, sample=1000, noise_level=0.5)
    data_g1 = data[data['gene_list']=="simulation000"]
    plt.scatter(data_g1['s0'], data_g1['u0'], c=data_g1['alpha'], s=1)
    plt.show()

    #backward
    data = generate_backward(gene_num=6, alpha=15, beta=10, gamma=12, sample=1000, noise_level=0.2)
    data_g1 = data[data['gene_list']=="simulation000"]
    plt.scatter(data_g1['s0'], data_g1['u0'], c=data_g1['alpha'], s=1)
    plt.show()

    #two stage alpha
    data = generate_onepath(gene_num=6, alpha1=15, alpha2=30, beta1=10, beta2=10, gamma1=12, gamma2=12, path1_pct=99, path2_pct=99, path1_sample=1000, path2_sample=1000, noise_level=0.2)
    data_g1 = data[data['gene_list']=="simulation000"]
    plt.scatter(data_g1['s0'], data_g1['u0'], c=data_g1['alpha'], s=1)
    plt.show()

    #multiple path
    data = generate_multipath(gene_num=6, alpha1=18, alpha2=30, beta1=10, beta2=10, gamma1=12, gamma2=12, path1_pct=99, path2_pct=90, path1_sample=1000, path2_sample=1000, noise_level=0.2)
    data_g1 = data[data['gene_list']=="simulation000"]
    plt.scatter(data_g1['s0'], data_g1['u0'], c=data_g1['alpha'], s=1)
    plt.show()

    #circle: including normal, early switch, and early stop
    data = generate_onepath(gene_num=6, alpha1=15, alpha2=0, beta1=10, beta2=3, gamma1=12, gamma2=4, path1_pct=99, path2_pct=99, path1_sample=1000, path2_sample=1000, noise_level=0.2)
    data_g1 = data[data['gene_list']=="simulation000"]
    plt.scatter(data_g1['s0'], data_g1['u0'], c=data_g1['alpha'], s=1)
    plt.show()

    #float circle (not started from 0,0)
    data = generate_float_circle(gene_num=6, alpha1=15, alpha2=6, beta1=10, beta2=10, gamma1=12, gamma2=12, path1_pct=99, path2_pct=99, path1_sample=1000, path2_sample=1000, noise_level=0.2)
    data_g1 = data[data['gene_list']=="simulation000"]
    plt.scatter(data_g1['s0'], data_g1['u0'], c=data_g1['alpha'], s=1)
    plt.show()


    #generate by alpha, beta, gamma of each cell
    data = generate_onepath(gene_num=6, alpha1=15, alpha2=0, beta1=10, beta2=3, gamma1=12, gamma2=4, path1_pct=99, path2_pct=99, path1_sample=1000, path2_sample=1000, noise_level=0.2)
    data_g1 = data[data['gene_list']=="simulation000"]
    data_g1 = data_g1[['alpha', 'beta', 'gamma']]
    data_cell = generate_by_each_cell(data_g1, t=3, noise_level=0.2)

    plt.scatter(data_cell.t, data_cell.alpha, s=1)
    plt.xlabel("t")
    plt.ylabel("alpha")
    plt.show()
    plt.scatter(data_cell.t, data_cell.beta, s=1)
    plt.xlabel("t")
    plt.ylabel("beta")
    plt.show()
    plt.scatter(data_cell.t, data_cell.gamma, s=1)
    plt.xlabel("t")
    plt.ylabel("gamma")
    plt.show()
    plt.scatter(data_cell['s0'], data_cell['u0'], c=data_cell['alpha'], s=1)
    plt.xlabel("s")
    plt.ylabel("u")
    plt.show()
    
    generate_2circle()
    generate_2backward()
    generate_2backward2()
######################################
########## END - SIM -Demos ##########
######################################

################################
########## SIM - Multipath #####
################################
def SIM_multi_path():
    ############## finding range
    # info: beta Longitudinal narrowing
    # info: gamma Horizontally flatten
    # alpha: size
    beta1_list=[10]
    beta2_list=[30,40,50]
    gamma1_list=[30,40,50]
    gamma2_list=[5,10,20,25]
    path1_pct_list=[95]
    path2_pct_list=[95]
    alpha1_list=[20,25,30,35]
    alpha2_list=[45,50,55]

    for beta1 in beta1_list:
        for beta2 in beta2_list:
            for gamma1 in gamma1_list:
                for gamma2 in gamma2_list:
                    for path1_pct in path1_pct_list:
                        for path2_pct in path2_pct_list:
                            for alpha1 in alpha1_list:
                                for alpha2 in alpha2_list:
                                    data = generate_multipath(gene_num=6, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, gamma1=gamma1, gamma2=gamma2, path1_pct=path1_pct, path2_pct=path2_pct, path1_sample=1000, path2_sample=20, noise_level=0.2)
                                    data_g1 = data[data['gene_list']=="simulation000"]
                                    plt.scatter(data_g1['s0'], data_g1['u0'], c=data_g1['alpha'], s=1)
                                    plt.title('alpha('+str(alpha1)+','+str(alpha2)+')_beta('+str(beta1)+','+str(beta2)+')_gamma('+str(gamma1)+','+str(gamma2)+')_pathpct('+str(path1_pct)+','+str(path2_pct)+')')
                                    plt.axis('scaled')
                                    plt.show()
    ############## End - finding range



    ############## generating multi_path_sim
    def gen_multi_path_sim(path1_sample,ratio):
        genn_amt=1000

        beta1_list=np.random.uniform(low=9, high=11, size=(genn_amt,)) #upper side
        beta2_list=np.random.uniform(low=30, high=50, size=(genn_amt,)) #downside
        gamma1_list=np.random.uniform(low=30, high=50, size=(genn_amt,))
        gamma2_list=np.random.uniform(low=5, high=25, size=(genn_amt,))
        path1_pct=95
        path2_pct=95
        alpha1_list=np.random.uniform(low=20, high=35, size=(genn_amt,))
        alpha2_list=np.random.uniform(low=45, high=55, size=(genn_amt,))

        i=0
        data_df=pd.DataFrame()
        for alpha1,alpha2,beta1,beta2,gamma1,gamma2 in zip(alpha1_list,alpha2_list,beta1_list,beta2_list,gamma1_list,gamma2_list):
            data = generate_multipath(gene_num=1, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, gamma1=gamma1, gamma2=gamma2, path1_pct=path1_pct, path2_pct=path2_pct, path1_sample=path1_sample, path2_sample=int(path1_sample/ratio), noise_level=0.2)
            data.gene_list = 'simulation'+str(i)
            data_df=data_df.append(data)
            # plt.scatter(data_g1['s0'], data_g1['u0'], c=data_g1['alpha'], s=1)
            # plt.title('alpha('+str(alpha1)+','+str(alpha2)+')_beta('+str(beta1)+','+str(beta2)+')_gamma('+str(gamma1)+','+str(gamma2)+')_pathpct('+str(path1_pct)+','+str(path2_pct)+')')
            # plt.axis('scaled')
            # plt.show()
            i=i+1
            print(i)
        data_df.to_csv(('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/simulation/data/multi_path/multi_path/multi_path_Path1Upper_('+str(path1_sample)+')__R'+str(ratio)+'.csv'),index=False)

    ratio_list=[0.2,0.4,0.6,0.8]
    path1_sample=1000 #upper
    for ratio in ratio_list:
        print(ratio)
        gen_multi_path_sim(path1_sample,ratio)
    ############## End generating multi_path_sim

######################################
########## END - SIM - Multipath #####
######################################

##################################
########## - SIM - Wings #########
##################################
def SIM_wing_path():
    s_list=[70,80]
    u_list=[10,15]
    alpha_list=[]
    gamma_list=[]
    for s_value in s_list:
        for u_value in u_list:
            alpha=u_value
            gamma=u_value/s_value
            alpha_list.append(alpha)
            gamma_list.append(gamma)

    s_list2=[100,110]
    u_list2=[30,40]
    alpha2_list=[]
    gamma2_list=[]
    for s_value in s_list2:
        for u_value in u_list2:
            alpha=u_value
            gamma=u_value/s_value
            alpha2_list.append(alpha)
            gamma2_list.append(gamma)

        ############## finding range

        # alpha1_list=[5,10]
        # alpha2_list=[30,40]
        # # alpha1_list=[40]
        # # alpha2_list=[60]
        # beta1_list=[20,40,60]
        # beta2_list=[20,40,60]
        # # beta1_list=[20]
        # # beta2_list=[20]
        # gamma1_list=[40]
        # gamma2_list=[60]
        # # gamma1_list=[40]
        # # gamma2_list=[40]
        # path1_pct_list=[95]
        # path2_pct_list=[95]
        # # alpha1 <alpha2
        # # gamma1 <gamma2

        alpha1_list=[min(alpha_list),max(alpha_list)]
        alpha2_list=[min(alpha2_list),max(alpha2_list)]
        # alpha1_list=[40]
        # alpha2_list=[60]
        beta1_list=[0.9,1.1]
        beta2_list=[0.9,1.1]
        # beta1_list=[20]
        # beta2_list=[20]
        gamma1_list=[min(gamma_list),max(gamma_list)]
        gamma2_list=[min(gamma2_list),max(gamma2_list)]
        # gamma1_list=[40]
        # gamma2_list=[40]
        path1_pct_list=[95]
        path2_pct_list=[50,70]
        # alpha1 <alpha2
        # gamma1 <gamma2

        for beta1 in beta1_list:
            for beta2 in beta2_list:
                for gamma1 in gamma1_list:
                    for gamma2 in gamma2_list:
                        for path1_pct in path1_pct_list:
                            for path2_pct in path2_pct_list:
                                for alpha1 in alpha1_list:
                                    for alpha2 in alpha2_list:
                                        data = generate_onepath(gene_num=1, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta1, gamma1=gamma1, gamma2=gamma2, path1_pct=path1_pct, path2_pct=path2_pct, path1_sample=1000, path2_sample=1000, noise_level=0.2)
                                        data_g1 = data[data['gene_list']=="simulation000"]
                                        plt.scatter(data_g1['s0'], data_g1['u0'], c=data_g1['alpha'], s=1)
                                        plt.title('alpha('+str(alpha1)+','+str(alpha2)+')_beta('+str(beta1)+','+str(beta2)+')_gamma('+str(gamma1)+','+str(gamma2)+')_pathpct('+str(path1_pct)+','+str(path2_pct)+')')
                                        plt.axis('scaled')
                                        plt.show()
        ############## End - finding range
        ############## generating win_path_sim
        def gen_wing_path_sim(path2_sample,ratio):

            s_list=[70,80]
            u_list=[10,15]
            alpha_list=[]
            gamma_list=[]
            for s_value in s_list:
                for u_value in u_list:
                    alpha=u_value
                    gamma=u_value/s_value
                    alpha_list.append(alpha)
                    gamma_list.append(gamma)

            s_list2=[100,110]
            u_list2=[30,40]
            alpha2_list=[]
            gamma2_list=[]
            for s_value in s_list2:
                for u_value in u_list2:
                    alpha=u_value
                    gamma=u_value/s_value
                    alpha2_list.append(alpha)
                    gamma2_list.append(gamma)


            genn_amt=1000

            alpha1_list=np.random.uniform(low=min(alpha_list), high=max(alpha_list), size=(genn_amt,))
            alpha2_list=np.random.uniform(low=min(alpha2_list), high=max(alpha2_list), size=(genn_amt,))
            beta1_list=np.random.uniform(low=0.9, high=1.1, size=(genn_amt,)) 
            beta2_list=np.random.uniform(low=0.9, high=1.1, size=(genn_amt,)) 
            gamma1_list=np.random.uniform(low=min(gamma_list), high=min(gamma_list), size=(genn_amt,))
            gamma2_list=np.random.uniform(low=min(gamma2_list), high=min(gamma2_list), size=(genn_amt,))
            path1_pct_list=np.random.uniform(low=90, high=95, size=(genn_amt,))
            path2_pct_list=np.random.uniform(low=50, high=70, size=(genn_amt,))

            i=0
            data_df=pd.DataFrame()
            for alpha1,alpha2,beta1,beta2,gamma1,gamma2,path1_pct,path2_pct in zip(alpha1_list,alpha2_list,beta1_list,beta2_list,gamma1_list,gamma2_list,path1_pct_list,path2_pct_list):
                data = generate_onepath(gene_num=1, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, gamma1=gamma1, gamma2=gamma2, path1_pct=path1_pct, path2_pct=path2_pct, path1_sample=int(path2_sample/ratio), path2_sample=path2_sample, noise_level=0.2)
                data.gene_list = 'simulation'+str(i)
                data_df=data_df.append(data)
                # plt.scatter(data_g1['s0'], data_g1['u0'], c=data_g1['alpha'], s=1)
                # plt.title('alpha('+str(alpha1)+','+str(alpha2)+')_beta('+str(beta1)+','+str(beta2)+')_gamma('+str(gamma1)+','+str(gamma2)+')_pathpct('+str(path1_pct)+','+str(path2_pct)+')')
                # plt.axis('scaled')
                # plt.show()
                i=i+1
                print(i)
            data_df.to_csv(('/Users/shengyuli/Library/CloudStorage/OneDrive-HoustonMethodist/work/Velocity/data/simulation/data/wing_path/wing_path_20220218/raw/wing_path_Path2Upper_'+str(path2_sample)+'__R'+str(ratio)+'.csv'),index=False)

        ratio_list=[0.2,0.4,0.6]
        path2_sample=1000 #from (0,0) # 
        for ratio in ratio_list:
            print(ratio)
            gen_wing_path_sim(path2_sample,ratio)

##################################
########## END - SIM - Wings #####
##################################


