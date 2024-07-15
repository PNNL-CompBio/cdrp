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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor


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

data = joined_dataset3.experiments[joined_dataset3.experiments.dose_response_metric == 'auc']
data.improve_drug_id.nunique()

ctrpv2 = data[data.study == 'CTRPv2']
ctrpv2.to_csv('cl_features/data.csv', index=False)
ctrpv2 = pd.read_csv('cl_features/data.csv')



#### load cl data

cl = pd.read_csv('cl_features/cl.csv')
cl_features = pd.read_pickle('cl_features/cl_1000.pkl')
# list(cl_features)[0]
cl_features = [str(i) for i in list(cl_features)]
cl = cl.loc[:, ['improve_sample_id'] + cl_features]



md = pd.read_csv('drugs/mdm.csv')
v = md.iloc[:, :-1]
remove=[]
for i in v.columns:
    
    v2 = v.loc[:, i]
    try:
        [float(i) for i in v2.values]
    except:
        remove.append(i)


v = md.drop(remove, axis=1)

id2smiles = dict(joined_dataset3.drugs[['improve_drug_id', 'canSMILES']].drop_duplicates().values)
smiles2id = {v:k for k,v in id2smiles.items()}
v['improve_drug_id'] = [smiles2id[i] for i in v.smiles]

v2 = v.drop(columns=['smiles'], axis=1)

# np.where(np.isnan(v3))
# sc = StandardScaler()


data2 = ctrpv2[['improve_sample_id', 'improve_drug_id','dose_response_value']]
d2 = pd.merge(data2, cl, on='improve_sample_id', how='left')
d3 = pd.merge(d2, v2, on='improve_drug_id', how='left')
d3 = d3.dropna(axis=0)


# sc = StandardScaler()
d3.reset_index(drop=True, inplace=True)
train, test = train_test_split(d3, test_size=.1)
# test, val = train_test_split(val, test_size=.5)

train_tmp = train.sample(frac=.1)
# train_x = train.iloc[:, 3:]
train_x = train_tmp.iloc[:, 3:]
test_x = test.iloc[:, 3:]

# train_y = train.dose_response_value.values
train_y = train_tmp.dose_response_value.values
test_y = test.dose_response_value.values

sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)


et = ExtraTreesRegressor()
et.fit(train_x, train_y)
et.predict(test_x)