# Evaluating the effectiveness of different ohmic features for drug response prediction

1. 1.cl_features.py: save 'transcriptomics', 'proteomics and 'copy_number' features. These will be used by the 'cancer data' encoder in the model.
2. 2.feature_importance.py: this script is used to find the important features for each feature type. the script will save 500, 1000, 2000, 3000, 4000 and 5000 most important features as determined by an extratreesclassifier model that predicts cancer type.
3. 3.et.py: Use a ExtraTreesRegressor model to make predictions for drug response.