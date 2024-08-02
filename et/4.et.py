import coderdata as cd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap
import seaborn as sns
import matplotlib.patches as mpatches
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import os
import argparse

DATA_DIR = '../process_data/data'
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
joined_dataset3.transcriptomics= joined_dataset3.transcriptomics[["improve_sample_id", "transcriptomics", "entrez_id", "source", "study"]]

data = joined_dataset3.experiments[joined_dataset3.experiments.dose_response_metric == 'auc']
data.improve_drug_id.nunique()

# NOTE: Currently using CTRPv2 data.
ctrpv2 = data[data.study == 'CTRPv2']




#### load cl data


if __name__ == '__main__':

    # N_IMP_FEATURES = 1000
    N_IMP_FEATURES = 'NONE'

    # for feature_type in ['transcriptomics', 'proteomics', 'copy_number']:
    for feature_type in ['multiplier_transcriptomics', 'multiplier_proteomics', 'multiplier_copy_number']:
            
        cl = pd.read_csv(f'cl_features/{feature_type}_features.csv')

        if 'multiplier' not in feature_type:
            
            cl_features = pd.read_pickle(f'cl_features/{feature_type}_fi_{N_IMP_FEATURES}.pkl')
            cl_features = [str(i) for i in list(cl_features)]
        else:
            cl_features = list(cl.columns[1:-2])
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




        data2 = ctrpv2[['improve_sample_id', 'improve_drug_id','dose_response_value']]
        d2 = pd.merge(data2, cl, on='improve_sample_id', how='left')
        d3 = pd.merge(d2, v2, on='improve_drug_id', how='left')
        d3 = d3.dropna(axis=0)


        # sc = StandardScaler()
        d3.reset_index(drop=True, inplace=True)

        # if feature_type=='transcriptomics':

        train, test = train_test_split(d3, test_size=.1)
        # test, val = train_test_split(val, test_size=.5)

        train_tmp = train.sample(frac=.1)

        test = test.set_index(['improve_sample_id','improve_drug_id'])
        train = train_tmp.set_index(['improve_sample_id','improve_drug_id'])

        test_index = test.index
        train_index = train.index

        with open(f'cl_features/{feature_type}_train_index.pkl', 'wb') as f:
            pickle.dump(train_index, f)
        
        with open(f'cl_features/{feature_type}_test_index.pkl', 'wb') as f:
            pickle.dump(test_index, f)

        # load the index values
        # train_index = pd.read_pickle('cl_features/train_tmp_index.pkl')
        # test_index = pd.read_pickle('cl_features/test_index.pkl')

        d3 = d3.set_index(['improve_sample_id','improve_drug_id'])

        train = d3.loc[train_index, :]
        test = d3.loc[test_index, :]

                
        train.reset_index(inplace=True)
        test.reset_index(inplace=True)



        # train_x = train.iloc[:, 3:]
        train_x = train.iloc[:, 3:]
        test_x = test.iloc[:, 3:]

        # train_y = train.dose_response_value.values
        train_y = train.dose_response_value.values
        test_y = test.dose_response_value.values

        sc = StandardScaler()
        train_x = sc.fit_transform(train_x)
        test_x = sc.transform(test_x)


        et = ExtraTreesRegressor()
        et.fit(train_x, train_y)
        pred = et.predict(test_x)

        os.makedirs('results', exist_ok=True)
        with open(f'results/{feature_type}_{N_IMP_FEATURES}_test_true.pkl', 'wb') as f:
            pickle.dump(test_y, f)
        
        with open(f'results/{feature_type}_{N_IMP_FEATURES}_test_pred.pkl', 'wb') as f:
            pickle.dump(pred, f)
            