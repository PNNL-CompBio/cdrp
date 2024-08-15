import coderdata as cd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, decomposition

hcmi = cd.DatasetLoader('hcmi', '../data')
beataml = cd.DatasetLoader('beataml', '../data')
cptac = cd.DatasetLoader('cptac', '../data')
depmap = cd.DatasetLoader('broad_sanger', '../data')
mpnst = cd.DatasetLoader('mpnst', '../data')

model_type_dict = {
    'Solid Tissue': 'tumor',
    'tumor': 'tumor',
    "organoid" : "organoid",
    'cell line': 'cell line',
    'Tumor': 'tumor',
    'ex vivo': 'tumor',
    '3D Organoid': 'organoid',
    'Peripheral Blood Components NOS': 'tumor',
    'Buffy Coat': np.nan,
     None: np.nan,
    'Peripheral Whole Blood': 'tumor',
    'Adherent Cell Line': 'cell line',
    '3D Neurosphere': 'organoid',
    '2D Modified Conditionally Reprogrammed Cells': 'cell line',
    'Pleural Effusion': np.nan,
    'Human Original Cells': 'cell line',
    'Not Reported': np.nan, 
    'Mixed Adherent Suspension': 'cell line',
    'Cell': 'cell line',
    'Saliva': np.nan
    }

# Join BeatAML and HCMI
joined_dataset0 = cd.join_datasets(beataml, hcmi)
# Join DepMap and CPTAC
joined_dataset1 = cd.join_datasets(depmap, cptac)
# Join Datasets
joined_dataset2 = cd.join_datasets(joined_dataset0,joined_dataset1)
# Final Join
joined_dataset3 = cd.join_datasets(joined_dataset2,mpnst)

joined_dataset3.proteomics= joined_dataset3.proteomics[["improve_sample_id", "proteomics", "entrez_id", "source", "study"]]
model_type_sample_map = dict(zip(joined_dataset3.samples['improve_sample_id'], joined_dataset3.samples['model_type']))
common_name_sample_map = dict(zip(joined_dataset3.samples['improve_sample_id'], joined_dataset3.samples['common_name']))
cancer_type_sample_map = dict(zip(joined_dataset3.samples['improve_sample_id'], joined_dataset3.samples['cancer_type']))
study_sample_map = dict(zip(joined_dataset3.proteomics['improve_sample_id'], joined_dataset3.proteomics['study']))




# save proteomics data
df = joined_dataset3.proteomics.pivot_table(index='improve_sample_id', columns='entrez_id', values='proteomics')
df = df.dropna(axis=1)
df.to_csv('proteomics.csv')
df = pd.read_csv('proteomics.csv')
df['cancer_type'] = df.improve_sample_id.map(cancer_type_sample_map)
df['study'] = df.improve_sample_id.map(study_sample_map)
df['model_type'] = df.improve_sample_id.map(model_type_sample_map)
df.to_csv('proteomics.csv', index=False)

# umap
reducer = umap.UMAP()
t_full_data = df.drop(['improve_sample_id', 'model_type', 'cancer_type', 'study'], axis=1)
scaled_t_full_data = StandardScaler().fit_transform(t_full_data)
embedding_t_full_data = reducer.fit_transform(scaled_t_full_data)
with open('umap_prot.pkl', 'wb') as f:
    pickle.dump(embedding_t_full_data, f)

# pca
pca = decomposition.PCA(n_components=10)
pca.fit(scaled_t_full_data)
X = pca.transform(scaled_t_full_data)
with open('pca_prot.pkl', 'wb') as f:
    pickle.dump(X,f)