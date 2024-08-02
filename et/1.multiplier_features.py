import pandas as pd
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
from sklearn import preprocessing
import os

os.makedirs('cl_features', exist_ok=True)

DATA_DIR='../process_data/data'

hcmi = cd.DatasetLoader('hcmi', DATA_DIR)
beataml = cd.DatasetLoader('beataml', DATA_DIR)
cptac = cd.DatasetLoader('cptac', DATA_DIR)
depmap = cd.DatasetLoader('broad_sanger', DATA_DIR)
mpnst = cd.DatasetLoader('mpnst', DATA_DIR)


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

for feature_type in ['transcriptomics', 'proteomics', 'copy_number']:
        
    df = pd.read_csv(f'../multiplier/b_matrices/all_{feature_type}_b.csv', index_col=0)
    df = df.T
    df['improve_sample_id'] = df.index

    df.improve_sample_id = df.improve_sample_id.astype(int)

    df['cancer_type'] = df.improve_sample_id.map(cancer_type_sample_map)
    df['study'] = df.improve_sample_id.map(study_sample_map)
    df['model_type'] = df.improve_sample_id.map(model_type_sample_map)

    df = df.drop(['study',	'model_type'], axis=1)



    df = df.loc[:, ['improve_sample_id'] + list(df.columns[:-2]) + ['cancer_type']]

    le = preprocessing.LabelEncoder()
    le.fit(df.cancer_type)
    df['cancer_type_int'] = le.transform(df.cancer_type)

    df.to_csv(f'cl_features/multiplier_{feature_type}_features.csv', index=False)
