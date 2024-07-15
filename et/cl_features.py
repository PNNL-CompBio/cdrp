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
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing


hcmi = cd.DatasetLoader('hcmi')
beataml = cd.DatasetLoader('beataml')
cptac = cd.DatasetLoader('cptac')
depmap = cd.DatasetLoader('broad_sanger')
mpnst = cd.DatasetLoader('mpnst')

# Join BeatAML and HCMI
joined_dataset0 = cd.join_datasets(beataml, hcmi)
# Join DepMap and CPTAC
joined_dataset1 = cd.join_datasets(depmap, cptac)
# Join Datasets
joined_dataset2 = cd.join_datasets(joined_dataset0,joined_dataset1)
# Final Join
joined_dataset3 = cd.join_datasets(joined_dataset2,mpnst)
joined_dataset3.transcriptomics= joined_dataset3.transcriptomics[["improve_sample_id", "transcriptomics", "entrez_id", "source", "study"]]


# select target type
data = joined_dataset3.experiments[joined_dataset3.experiments.dose_response_metric == 'auc']
# select study
ctrpv2 = data[data.study == 'CTRPv2']
# joined_dataset3.transcriptomics.study.unique()

# get cancer type map
cancer_type_sample_map = dict(zip(joined_dataset3.samples['improve_sample_id'], joined_dataset3.samples['cancer_type']))

# pivot
cl = joined_dataset3.transcriptomics.pivot_table(index='improve_sample_id', columns='entrez_id', values='transcriptomics')
cl = cl.dropna(axis=1)
cl.columns.name = None
cl.reset_index(inplace=True)



# add cancer type
cl['cancer_type'] = cl.improve_sample_id.map(cancer_type_sample_map)

# convert str labels to int
le = preprocessing.LabelEncoder()
le.fit(cl.cancer_type)
cl['cancer_type_int'] = le.transform(cl.cancer_type)
cl.to_csv('cl_features/cl.csv', index=False)



