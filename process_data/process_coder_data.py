import coderdata as cd
# import pandas as pd
import os
import matplotlib.pyplot as plt
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing



def create_features(joined_dataset3, feature_type):
        
    # select target type
    # data = joined_dataset3.experiments[joined_dataset3.experiments.dose_response_metric == 'auc']
    # select study
    # data = data[data.study == 'CTRPv2']
    # joined_dataset3.transcriptomics.study.unique()

    # get cancer type map
    cancer_type_sample_map = dict(zip(joined_dataset3.samples['improve_sample_id'], joined_dataset3.samples['cancer_type']))

    # pivot
    if feature_type == 'transcriptomics':
        cl = joined_dataset3.transcriptomics.pivot_table(index='improve_sample_id', columns='entrez_id', values='transcriptomics')
    elif feature_type == 'proteomics':
        cl = joined_dataset3.proteomics.pivot_table(index='improve_sample_id', columns='entrez_id', values='proteomics')
    elif feature_type == 'copy_number':
        cl = joined_dataset3.copy_number.pivot_table(index='improve_sample_id', columns='entrez_id', values='copy_number')


    cl = cl.dropna(axis=1)
    cl.columns.name = None
    cl.reset_index(inplace=True)



    # add cancer type
    cl['cancer_type'] = cl.improve_sample_id.map(cancer_type_sample_map)

    # convert str labels to int
    le = preprocessing.LabelEncoder()
    le.fit(cl.cancer_type)
    cl['cancer_type_int'] = le.transform(cl.cancer_type)

    cl.to_csv(f'features/{feature_type}_features.csv', index=False)



if __name__ == '__main__':

    hcmi = cd.DatasetLoader('hcmi', 'data')
    beataml = cd.DatasetLoader('beataml', 'data')
    cptac = cd.DatasetLoader('cptac', 'data')
    depmap = cd.DatasetLoader('broad_sanger', 'data')
    mpnst = cd.DatasetLoader('mpnst', 'data')

    # Join BeatAML and HCMI
    joined_dataset0 = cd.join_datasets(beataml, hcmi)
    # Join DepMap and CPTAC
    joined_dataset1 = cd.join_datasets(depmap, cptac)
    # Join Datasets
    joined_dataset2 = cd.join_datasets(joined_dataset0,joined_dataset1)
    # Final Join
    joined_dataset3 = cd.join_datasets(joined_dataset2,mpnst)
    joined_dataset3.transcriptomics = joined_dataset3.transcriptomics[["improve_sample_id", "transcriptomics", "entrez_id", "source", "study"]]
    joined_dataset3.proteomics = joined_dataset3.proteomics[["improve_sample_id", "proteomics", "entrez_id", "source", "study"]]
    joined_dataset3.copy_number = joined_dataset3.copy_number[["improve_sample_id", "copy_number", "entrez_id", "source", "study"]]


    os.makedirs('features', exist_ok=True)
    for feature_type in ['transcriptomics', 'proteomics', 'copy_number']:
        create_features(joined_dataset3, feature_type)
