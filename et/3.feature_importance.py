import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pickle


# for feature_type in ['transcriptomics', 'proteomics', 'copy_number']:
for feature_type in ['multiplier_copy_number', 'multiplier_proteomics', 'multiplier_transcriptomics']:


    cl = pd.read_csv(f'cl_features/{feature_type}_features.csv')
    sc = StandardScaler()
    cl.iloc[:, 1:-2] = sc.fit_transform(cl.iloc[:, 1:-2])



    train, test = train_test_split(cl, test_size=.1)
    train_y = train.cancer_type_int.values
    test_y = test.cancer_type_int.values
    train_x = train.iloc[:, 1:-2]
    test_x = test.iloc[:, 1:-2]


    et = ExtraTreesClassifier()
    et.fit(train_x, train_y)

    fi = et.feature_importances_
    features = cl.columns[1:-2]
    fi_srt = np.argsort(fi)[::-1]


    for i in [500, 1000, 2000, 3000, 4000, 5000 ]:
        
        selected = features[fi_srt][:i]
        with open(f'cl_features/{feature_type}_fi_{i}.pkl', 'wb') as f:

            pickle.dump(selected, f)


