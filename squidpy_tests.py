#Test change here

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import harmonypy as hm

import os
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.4.2/'
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from scipy.sparse import csc_matrix, issparse

#Read in the data matrix into an annotated data object via scanpy read_10x
adata = sc.read_10x_mtx("C:/Users/HainBuche/Downloads/transcriptomics_example/", prefix='GSE252265_')

#Read in barcodes and positional data into dataframes
df_codes = pd.read_csv("C:/Users/HainBuche/Documents/PythonWorkspace/barcodes.tsv", sep="\t", header=None)
df = pd.read_csv("C:/Users/HainBuche/Downloads/transcriptomics_example/all/GSE252265_aggr_tissue_positions.csv")
#df_feats = pd.read_csv("C:/Users/HainBuche/Documents/PythonWorkspace/features.tsv", sep="\t", header=None)

#Assign the spatial information to the annotated data object
codes=df[df['barcode'].isin(df_codes[0])]
pos=np.array(codes[['pxl_row_in_fullres','pxl_col_in_fullres']])
adata.obsm["spatial"]=pos

#Label batches (samples)
df_agg = pd.read_csv("C:/Users/HainBuche/Downloads/transcriptomics_example/all/GSE252265_aggregation.csv")
batch_mapping = {str(idx+1): x for idx, x in enumerate(df_agg['library_id'])}
adata.obs['batch'] = adata.obs_names.str.split('-').str[-1].map(batch_mapping)
adata.obs['batch'] = adata.obs['batch'].astype('category')

#Create subset for different batches for scran 
batches = adata.obs["batch"].unique()
batch_subsets = [adata[adata.obs["batch"] == batch] for batch in batches]

##Perform scran
# Import required R packages
scran = importr('scran')
sce = importr('SingleCellExperiment')

for abatch in batch_subsets:
    
    #Perform Leiden clustering to generate "Cell" pools for scran normalization
    adata_pp = abatch
    sc.pp.normalize_total(adata_pp)
    sc.pp.log1p(adata_pp)
    sc.pp.pca(adata_pp, n_comps=15)
    sc.pp.neighbors(adata_pp)
    sc.tl.leiden(adata_pp, key_added="groups", resolution=0.5)

    # Convert AnnData to CSC in R and then to SingleCellExperiment
    data_mat = abatch.X.T.toarray() #if issparse(adata_pp.X) else adata_pp.X.T
    numpy2ri.activate()
    r_counts = ro.r.matrix(data_mat, nrow=data_mat.shape[0], ncol=data_mat.shape[1])
    sce_obj = sce.SingleCellExperiment(ro.ListVector({"counts": r_counts}))
    #sce_obj = sce.SingleCellExperiment(list(counts=data_mat)) #would need to run r things differrently for this
    # Assign Leiden clusters from adata_pp to R environment
    r_groups = ro.StrVector(adata_pp.obs["groups"])
    ro.r.assign("input_groups", r_groups)
    # Compute size factors using scran
    size_factors = scran.computeSumFactors(sce_obj, clusters=ro.r("input_groups"), **{"min.mean": 0.1})
    # Apply normalization
    size_factors_num = ro.r('as.numeric')(ro.r('sizeFactors')(size_factors))
    normalized_counts = abatch.X.toarray() / size_factors_num[:,None]
    #Update the original matrix accordingly
    abatch.X = csc_matrix(sc.pp.log1p(normalized_counts))
##

#Assign cell cylce scores G2M and S for every batch
cycle=pd.read_csv("C:/Users/HainBuche/Downloads/transcriptomics_example/Homo_sapiens.csv")
g2m_genes=np.array(cycle['geneID'][:54])
s_genes=np.array(cycle['geneID'][54:])
gene_id_to_symbol = dict(zip(adata.var['gene_ids'], adata.var_names))
s_genes_symbols = [gene_id_to_symbol[gene] for gene in s_genes if gene in gene_id_to_symbol]
g2m_genes_symbols = [gene_id_to_symbol[gene] for gene in g2m_genes if gene in gene_id_to_symbol]
for abatch in batch_subsets:
    sc.tl.score_genes_cell_cycle(abatch, s_genes=s_genes_symbols, g2m_genes=g2m_genes_symbols)

#Merge the batches again
adata_ori = adata.copy()
adata_norm = ad.concat(batch_subsets, join='outer', merge='same', label='batch')
adata = adata_norm.copy()

#Regress out those that scored G2M and S 
sc.pp.regress_out(adata, ['S_score', 'G2M_score'])
adata_reg = adata.copy()
#sc.pp.scale(adata)

#If not log1p transformed yet, shift values to positive and apply log1p?
#adata.X = adata.X.astype(float) - np.min(adata.X) + 1e-6
#sc.pp.normalize_total(adata)
#sc.pp.log1p(adata)

#Only keep the 3000 most variable genes in a seurat way
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat', subset=True)

#Derive first 100 principal components
sc.pp.pca(adata, n_comps=100)

#Perform Harmony batch correction
harmony_result = hm.run_harmony(
    adata.obsm['X_pca'], 
    adata.obs, 
    'batch', 
    max_iter_harmony=10
)
adata.obsm['X_pca_harmony'] = harmony_result.Z_corr.T

#Calculate k-nearest neighbors k=30 using the harmony corrected PCs
sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca_harmony')

#Caluclate the UMAP based on that
sc.tl.umap(adata, min_dist=0.3)

#Show the UMAP annotated with batch labels
sc.pl.umap(adata, color='batch')

#Perform leiden clustering at 0.6 resolution and show UMAP according to this
sc.tl.leiden(adata, resolution=0.6)
sc.pl.umap(adata, color=['leiden'])

#Compute differential expression across all clusters
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', n_genes=3000)

#Visualize top differentially expressed genes
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

#Matrix plot
sc.pl.rank_genes_groups_matrixplot(adata, n_genes=5)

#Dot plot
sc.pl.rank_genes_groups_dotplot(adata, n_genes=10)

#Dot plot of specified genes as in paper
adata_norm.obs['leiden'] = adata.obs['leiden'].copy()
genes_of_interest = ['KRT10','KRT15','PITX1','SOX2','ITGB1','TWIST1','AREG','TNC','FN1','THBS1',
                     'SNAI2','VIM','COL1A1','FAP','IGHG1','CD3E']
sc.pl.dotplot(adata_norm, var_names=genes_of_interest, groupby='leiden', vmin=0)


# other things to try from here
'''
#marker_genes_dict = {
#    'Epi': ['KRT10','KRT15','PITX1','SOX2'],
#    'SC': ['KRT15','PITX1','SOX2','ITGB1'],
#    'EMT': ['ITGB1','TWIST1','AREG','TNC','FN1','THBS1']
#}
#sc.pl.dotplot(adata_reg, marker_genes_dict, groupby='leiden')

#label leiden clusters ?
leiden_names = {
    '0': 'B-cells', 
    '1': 'T-cells', 
    '2': 'Monocytes'
}
adata.obs['leiden'] = adata.obs['leiden'].map(leiden_names)

sc.pl.rank_genes_groups_dotplot(
    adata, 
    n_genes=5,
    values_to_plot='logfoldchanges',
    min_logfoldchange=0.5,  # Minimum log fold change to include
    vmax=7,  
    vmin=-7,  
    cmap='bwr'  # Color map (blue-white-red)
)

sq.pl.spatial_scatter(
    adata,
    library_id="spatial",
    shape=None,
    color=[
        "batch",
    ],
    wspace=0.4,
)
'''

