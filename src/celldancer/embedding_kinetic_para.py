import numpy as np
import pandas as pd

def embedding(
    cellDancer_df,
    kinetic_para,
    umap_n=25
):
    """Calculate the UMAP based on kinetic parameter(s).
        
    Arguments
    ---------
    cellDancer_df: `pandas.Dataframe`
        Data frame of velocity estimation results. Columns=['cellIndex', 'gene_name', 's0', 'u0', 's1', 'u1', 'alpha', 'beta', 'gamma', 'loss', 'cellID', 'clusters', 'embedding1', 'embedding2']
    kinetic_para: `str`
        Which parameter is used to calculate embedding space, could be selected from {'alpha', 'beta', 'gamma', 'alpha_beta_gamma'}.
    umap_n: `int` (optional, default: 25)
        The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation in UMAP.

    Returns
    -------
    cellDancer_df: `pandas.DataFrame`
        The updated cellDancer_df with additional column of UMAP based on kinetic parameter(s).

    """  
    import umap
    if set([(kinetic_para+'_umap1'),(kinetic_para+'_umap2')]).issubset(cellDancer_df.columns):
        cellDancer_df=cellDancer_df.drop(columns=[(kinetic_para+'_umap1'),(kinetic_para+'_umap2')])

    if kinetic_para=='alpha' or kinetic_para=='beta' or kinetic_para=='gamma':
        para_df=cellDancer_df.pivot(index='cellIndex', columns='gene_name', values=kinetic_para)
    elif kinetic_para=='alpha_beta_gamma':
        alpha_df=cellDancer_df.pivot(index='cellIndex', columns='gene_name', values='alpha')
        beta_df=cellDancer_df.pivot(index='cellIndex', columns='gene_name', values='beta')
        gamma_df=cellDancer_df.pivot(index='cellIndex', columns='gene_name', values='gamma')
        para_df=pd.concat([alpha_df,beta_df,gamma_df],axis=1)
    else:
        print('kinetic_para should be set in one of alpha, beta, gamma, or alpha_beta_gamma.')

    def get_umap(df,n_neighbors=umap_n, min_dist=0.1, n_components=2, metric='euclidean'):
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric
        )
        embed = fit.fit_transform(df);
        return(embed)
    umap_para=get_umap(para_df)
    umap_info=pd.DataFrame(umap_para,columns=[(kinetic_para+'_umap1'),(kinetic_para+'_umap2')])

    gene_amt=len(cellDancer_df.gene_name.drop_duplicates())
    umap_col=pd.concat([umap_info]*gene_amt)
    umap_col.index=cellDancer_df.index
    cellDancer_df=pd.concat([cellDancer_df,umap_col],axis=1)
    return(cellDancer_df)
