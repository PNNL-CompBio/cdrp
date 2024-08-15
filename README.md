# Modeling
## model4
Contains a drug response prediction model with 4 drug enoders

### Running models
`python run_models.py --config configs/config_ccle.yaml`


# Data Exploration

## process_data
The scripts in this directory are used to create input 'transcriptomics', 'proteomics', 'copy_number' features for machine learning models.
Follow the steps below.

    1. Download coder-data from `https://figshare.com/s/fd20edec1955d82fabea`
    2. Extract data into the data directory
    3. run `process_coder_data.py`


### multiplier data
Scripts to obtain multiplier embeddings
Run the scripts in the following order.

    1. 1.save_coder_data_for_multiplier.py
    2. Copy the data and util folders from the multiplier repository at `https://github.com/greenelab/multi-plier`
    3. Rscript 2.obtain_b_matrix.R


## et
A light-weight extra trees regressor model for evaluation purposes. The scripts provided are used to evaluate the effectiveness of different ohmic features for drug response prediction.

    1. 1.multiplier_features.py
    2. 2.cl_features.py: save 'transcriptomics', 'proteomics and 'copy_number' features. These will be used by the 'cancer data' encoder in the model.
    3. 3.feature_importance.py: this script is used to find the important features for each feature type. the script will save 500, 1000, 2000, 3000, 4000 and 5000 most important features as determined by an extratreesclassifier model that predicts cancer type.
    4. 4.et.py: Use a ExtraTreesRegressor model to make predictions for drug response.


# Apps
apps/app_prot.py is an interactive proteomics data exploration app

    1. Create the neccessay input data by running `python  proteomics.py`
    2. Next, launch the app with `streamlit run app_prot.py`