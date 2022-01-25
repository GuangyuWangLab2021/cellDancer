# set data_source

def select_gene_set(data_source):
    if data_source=="scv":
        gene_choice=["Ank","Abcc8","Tcp11","Nfib","Ppp3ca",
                "Rbfox3","Cdk1","Gng12","Map1b","Cpe",
                "Gnao1","Pcsk2","Tmem163","Pak3","Wfdc15b",
                "Nnat","Anxa4","Actn4","Btbd17","Dcdc2a",
                "Adk","Smoc1","Mapre3","Pim2","Tspan7",
                "Top2a","Rap1b","Sulf2"]
        #gene_choice=["Sulf2","Top2a","Abcc8"]
    elif data_source =='velocyto_selectByUs':
        gene_choice=['Adam23','Arid5b','Blcap','Coch','Dcx',
                'Elavl2','Elavl3','Elavl4','Eomes','Eps15',
                'Fam210b','Foxk2','Gpc6','Icam5','Kcnd2',
                'Pfkp','Psd3','Sult2b1','Thy1','Car2','Clip3','Ntrk2','Nnat'] #21 genes

    elif data_source =='velocyto':
        gene_choice=['Ank','Btbd17','Cdk1','Cpe','Gnao1',
                'Gng12','Map1b','Mapre3','Nnat','Ntrk2',
                'Pak3','Pcsk2','Ppp3ca','Rap1b','Rbfox3',
                'Smoc1','Sulf2','Tmem163','Top2a','Tspan7',
                "Pdgfra","Igfbpl1",#
                #Added GENE from page 11 of https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0414-6/MediaObjects/41586_2018_414_MOESM3_ESM.pdf
                "Syngr1","Fam210b","Meg3","Fam19a2","Kcnc3","Dscam"]#"Hagh"] time spent:  46.042665135860446  min
    elif data_source=="velocyto_in_storyboard":
        gene_choice=['Rimbp2','Dctn3','Psd3','Dcx','Elavl4','Ntrk2']
    elif data_source=='test':
        gene_choice=['simulation0']

    return(gene_choice)

