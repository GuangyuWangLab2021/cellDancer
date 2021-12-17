gene_choice=['Ank',
            'Btbd17',
            'Cdk1',
            'Cpe',
            'Gnao1',
            'Gng12',
            'Map1b',
            'Mapre3',
            'Nnat',
            'Ntrk2',
            'Pak3',
            'Pcsk2',
            'Ppp3ca',
            'Rap1b',
            'Rbfox3',
            'Smoc1',
            'Sulf2',
            'Tmem163',
            'Top2a',
            'Tspan7']

data_df=den_gyr[['gene_list', 'u0','s0']][den_gyr.gene_list.isin(gene_choice)]
data_df_downsampled=downsampling(data_df,gene_choice,para='neighbors')
datamodule = realDataMododule(data_fit = data_df_downsampled, data_predict=data_df, sampling_ratio=0.5)

brief_e0, detail_e0 = train(datamodule,model_path='../../data/model2', max_epoches=0, n_jobs=8)
brief_e5, detail_e5 = train(datamodule,model_path='../../data/model2', max_epoches=5, n_jobs=8)
brief_e10, detail_e10 = train(datamodule,model_path='../../data/model2', max_epoches=10, n_jobs=8)
brief_e50, detail_e50 = train(datamodule,model_path='../../data/model2', max_epoches=50, n_jobs=8)
brief_e100, detail_e100 = train(datamodule,model_path='../../data/model2', max_epoches=100, n_jobs=8)
brief_e200, detail_e200 = train(datamodule,model_path='../../data/model2', max_epoches=200, n_jobs=8)
brief_e300, detail_e300 = train(datamodule,model_path='../../data/model2', max_epoches=300, n_jobs=8)
brief_e400, detail_e400 = train(datamodule,model_path='../../data/model2', max_epoches=400, n_jobs=8)
brief_e500, detail_e500 = train(datamodule,model_path='../../data/model2', max_epoches=500, n_jobs=8)

detail_e0.to_csv("output/detailcsv/adj_e/detail_e0.csv")
detail_e5.to_csv("output/detailcsv/adj_e/detail_e5.csv")
detail_e10.to_csv("output/detailcsv/adj_e/detail_e10.csv")
detail_e50.to_csv("output/detailcsv/adj_e/detail_e50.csv")
detail_e100.to_csv("output/detailcsv/adj_e/detail_e100.csv")
detail_e200.to_csv("output/detailcsv/adj_e/detail_e200.csv")
detail_e300.to_csv("output/detailcsv/adj_e/detail_e300.csv")
detail_e400.to_csv("output/detailcsv/adj_e/detail_e400.csv")
detail_e500.to_csv("output/detailcsv/adj_e/detail_e500.csv")

brief_e0.to_csv("output/detailcsv/adj_e/brief_e0.csv")
brief_e5.to_csv("output/detailcsv/adj_e/brief_e5.csv")
brief_e10.to_csv("output/detailcsv/adj_e/brief_e10.csv")
brief_e50.to_csv("output/detailcsv/adj_e/brief_e50.csv")
brief_e100.to_csv("output/detailcsv/adj_e/brief_e100.csv")

list_e=[0,5,10,50,100,200,300,400,500]
list_e=[0,10,50,100,200]
g_list=gene_choice

for e_num in list_e:
    file_path="output/detailcsv/adj_e/detail_e"+str(e_num)+".csv"
    detail = pd.read_csv (file_path,index_col=False)
    detail["alpha_new"]=detail["alpha"]/detail["beta"]
    detail["beta_new"]=detail["beta"]/detail["beta"]
    detail["gamma_new"]=detail["gamma"]/detail["beta"]
    detailfinfo="e"+str(e_num)

    #color_map="Spectral"
    #color_map="PiYG"
    #color_map="RdBu"
    color_map="coolwarm"
    # color_map="bwr"
    alpha_inside=0.3
    alpha_inside=1
    vmin=0
    vmax=5
    for i in g_list:
        save_path="output/velo_plot_adj_e/"+i+"_"+"e"+str(e_num)+".pdf"# notice: changed
        velocity_plot(detail, [i],detailfinfo,color_scatter,pointsize,alpha_inside,color_map,vmin,vmax,save_path) # from cell dancer

brief_e100[brief_e100.gene_name=="Ntrk2"].head(50)