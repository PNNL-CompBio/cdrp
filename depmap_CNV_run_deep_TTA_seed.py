# script that will run the deep learning model with TTA
from data_utils import DataProcessor, add_smiles, average_auc, average_dose_response_value
from gnn_utils import CreateData, EarlyStopping, test_fn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from depmap_CNV_model import Model
import pandas as pd 
import os

def split_df(df, seed):
    train, val = train_test_split(df, random_state=seed, test_size=0.2, train_size=0.8)

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    return train, val

# all the parameters are set here 
data_split_seed = 10 # adjust seed here
bs = 64
lr = 1e-4
n_epochs = 100 # test with small number of epochs 
mpnst_input_path = "./coderdata_input/MPNST_cnn_mutation_seq.csv.gz"
depmap_input_path = "./coderdata_input/depmap_copy_number.csv.gz"
# if you run the model for the first time, set load_trained_model to False
load_trained_model = False
selected_gene_df = pd.read_csv("./shared_input/graphDRP_landmark_genes_map.txt", sep='\t')
# def intersect_columns(mpnst_gene_exp, depmap_gene_exp, selected_gene_df):
#     selected_genes = selected_gene_df.iloc[:, 1].tolist()
#     common_columns = list("improve_sample_id",set(selected_genes).intersection(set(mpnst_gene_exp.columns), set(depmap_gene_exp.columns)))
#     mpnst_gene_exp = mpnst_gene_exp[common_columns]
#     depmap_gene_exp = depmap_gene_exp[common_columns]
#     return mpnst_gene_exp, depmap_gene_exp

# intersect_columns ISSUES: CANNOT FIND COMMON COLUMNS BETWEEN MPNST AND DEPMAP AND GRAPHDRP LANDMARK GENES
def CNV_intersect_columns(mpnst_gene_exp, depmap_gene_exp, selected_gene_df):
    # Extract the gene names from the second column of selected_gene_df
    selected_genes = selected_gene_df.iloc[:, 1].tolist()
    # Convert gene IDs to string because DataFrame columns are strings
    selected_genes_str = [str(int(gene)) for gene in selected_genes]
    # Find common genes between selected_genes and the columns of the input DataFrames
    common_columns = list(set(selected_genes_str).intersection(set(mpnst_gene_exp.columns), set(depmap_gene_exp.columns)))
    # Ensure "improve_sample_id" is included if it exists in both DataFrames
    if "improve_sample_id" in mpnst_gene_exp.columns and "improve_sample_id" in depmap_gene_exp.columns:
        common_columns = ["improve_sample_id"] + common_columns
    # Subset both DataFrames to only include the common columns
    mpnst_gene_exp_filtered = mpnst_gene_exp[common_columns]
    depmap_gene_exp_filtered = depmap_gene_exp[common_columns]
    return mpnst_gene_exp_filtered, depmap_gene_exp_filtered
##############################
# DATA PROCESSING
##############################
# Process gene expression data
# Downloads and saves the gene expression data into a separate directory "process_input"
def run_experiment(data_split_seed):
    if not os.path.exists("shared_input/MPNST_cnn_mutation_seq_wide.tsv"):
        DataProcessor.CNV_convert_long_to_wide_format(mpnst_input_path)
    mpnst_gene_exp = pd.read_csv("shared_input/MPNST_cnn_mutation_seq_wide.tsv", sep='\t')

    if not os.path.exists("shared_input/depmap_copy_number_wide.tsv"):
        DataProcessor.CNV_convert_long_to_wide_format(depmap_input_path)
    depmap_gene_exp = pd.read_csv("shared_input/depmap_copy_number_wide.tsv", sep='\t')

    mpnst_gene_exp, depmap_gene_exp = CNV_intersect_columns(mpnst_gene_exp, depmap_gene_exp, selected_gene_df)

    # mpnst: process experiment & drug data
    mpnst_exp = pd.read_csv("./coderdata_input/MPNST_experiments.csv.gz", compression='gzip')
    mpnst_drugs = pd.read_csv("./coderdata_input/MPNST_drugs.tsv.gz", sep='\t', compression='gzip')
    # average auc values for the same improve_sample_id and drug_id
    mpnst_exp = average_auc(mpnst_exp)
    # add smiles and split data
    mpnst_df_all = add_smiles(mpnst_drugs, mpnst_exp, "auc")

    # depmap: process experiment & drug data
    depmap_exp = pd.read_csv("./coderdata_input/depmap_experiments.tsv", sep='\t')
    depmap_drugs = pd.read_csv("./coderdata_input/depmap_drugs.tsv", sep='\t')
    # change depmap_exp column name "dose_response_value" to "auc" 
    depmap_exp.rename(columns={"dose_response_value": "auc"}, inplace=True)
    # average auc values for the same improve_sample_id and drug_id
    depmap_exp = average_auc(depmap_exp)
    # add smiles and split data
    depmap_df_all = add_smiles(depmap_drugs, depmap_exp, "auc")
    # depmap_df_all = add_smiles(depmap_drugs, depmap_exp, "dose_response_value") # new version use "dose_response_value" instead of "auc"

    # merge and split the data
    # Find the intersection of improve_sample_id in RNA & drug info
    mpnst_common_ids = set(mpnst_df_all['improve_sample_id']).intersection(set(mpnst_gene_exp['improve_sample_id']))
    # Filter RNA & drug to only include rows with improve_sample_id in the intersection
    mpnst_df = mpnst_df_all[mpnst_df_all['improve_sample_id'].isin(mpnst_common_ids)].reset_index(drop=True)
    mpnst_gene_exp = mpnst_gene_exp[mpnst_gene_exp['improve_sample_id'].isin(mpnst_common_ids)]
    # mpnst_df = mpnst_df.reset_index(drop=True, inplace=True)
    test = mpnst_df.reset_index(drop=True, inplace=True)

    # Find the intersection of improve_sample_id in RNA & drug
    depmap_common_ids = set(depmap_df_all['improve_sample_id'].unique()).intersection(set(depmap_gene_exp['improve_sample_id'].unique()))
    # Filter RNA & drug to only include rows with improve_sample_id in the intersection
    depmap_df = depmap_df_all[depmap_df_all['improve_sample_id'].isin(depmap_common_ids)].reset_index(drop=True)
    depmap_gene_exp = depmap_gene_exp[depmap_gene_exp['improve_sample_id'].isin(depmap_common_ids)]
    train, val= split_df(df=depmap_df, seed=data_split_seed)

    # mpnst: Ensure improve_sample_id is set as the index before scaling
    mpnst_gene_exp = mpnst_gene_exp.set_index('improve_sample_id')
    # Now perform the scaling operation on the DataFrame without the index column
    scaler = StandardScaler() # mean=0, unit variance
    mpnst_gene_exp_scaled = scaler.fit_transform(mpnst_gene_exp)
    # When creating the new DataFrame, use the same columns as the gene_exp DataFrame
    # Because gene_exp now does not include 'improve_sample_id' column, we don't need to adjust column names
    mpnst_gene_exp_scaled = pd.DataFrame(mpnst_gene_exp_scaled, index=mpnst_gene_exp.index, columns=mpnst_gene_exp.columns)
    data_creater = CreateData(gexp=mpnst_gene_exp_scaled, encoder_type='transformer', metric="auc", data_path= "shared_input/")
    test_ds = data_creater.create_data(mpnst_df)

    # bealaml: Ensure improve_sample_id is set as the index before scaling
    depmap_gene_exp = depmap_gene_exp.set_index('improve_sample_id')
    # Now perform the scaling operation on the DataFrame without the index column
    scaler = StandardScaler() # mean=0, unit variance
    depmap_gene_exp_scaled = scaler.fit_transform(depmap_gene_exp)
    # When creating the new DataFrame, use the same columns as the gene_exp DataFrame
    # Because gene_exp now does not include 'improve_sample_id' column, we don't need to adjust column names
    depmap_gene_exp_scaled = pd.DataFrame(depmap_gene_exp_scaled, index=depmap_gene_exp.index, columns=depmap_gene_exp.columns)
    data_creater = CreateData(gexp=depmap_gene_exp_scaled, encoder_type='transformer', metric="auc", data_path= "shared_input/")
    # bealaml: define the train and val datasets
    train_ds = data_creater.create_data(train)
    val_ds = data_creater.create_data(val)

    # bs = 64
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(gnn_features = None, encoder_type='transformer').to(device)
    # lr = 1e-4
    adam = torch.optim.Adam(model.parameters(), lr = lr)
    optimizer = adam

    # n_epochs = 100
    ckpt_path = 'tmp/best.pt'

    early_stopping = EarlyStopping(patience = n_epochs, verbose=True, chkpoint_name = ckpt_path)
    criterion = nn.MSELoss()

    # train the model
    # if we have a trained model, skip training and evaluate:
    if load_trained_model:
        model.load_state_dict(torch.load(ckpt_path))
        test_rmse, true, pred = test_fn(test_loader, model, device)
        # test['true'] = true
        # test['pred'] = pred
        print(test_rmse) # print the test dataframe with true and pred columns
        # if args.feature_path:
        #     test = test[['improve_sample_id', 'smiles', 'improve_chem_id', 'auc', 'true', 'pred']]
        # test.to_csv( os.path.join(out_dir, 'test_predictions.csv'), index=False )
        exit()
    # train the model  
    hist = {"train_rmse":[], "val_rmse":[]}
    for epoch in range(0, n_epochs):
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = output.reshape(-1,)

            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()


        # train_rmse = gnn_utils.test_fn(train_loader, model, device)
        val_rmse, _, _ = test_fn(val_loader, model, device)
        early_stopping(val_rmse, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # hist["train_rmse"].append(train_rmse)
        hist["val_rmse"].append(val_rmse)
        # print(f'Epoch: {epoch}, Train_rmse: {train_rmse:.3}, Val_rmse: {val_rmse:.3}')
        print(f'Epoch: {epoch}, Val_rmse: {val_rmse:.3}')
    model.load_state_dict(torch.load(ckpt_path))
    test_rmse, true, pred = test_fn(test_loader, model, device)
    # model_save_path = f'models/model_seed_{data_split_seed}.pt'
    # torch.save(model.state_dict(), model_save_path) # save the model for each seed
    return test_rmse

# Main loop for different seeds
results = {}
for seed in range(1, data_split_seed+1):
    test_rmse = run_experiment(data_split_seed=seed)
    results[seed] = test_rmse

# File path for saving the results
results_file_path = f'seed_{data_split_seed}_epoch_100_depmap_CNV_train_results_table.txt'

# Saving results to a file
with open(results_file_path, 'w') as file:
    file.write("Test RMSE for different seeds:\n")
    file.write("Seed\tTest RMSE\n")
    for seed, test_rmse in results.items():
        file.write(f"{seed}\t{test_rmse}\n")

print(f"Results saved to {results_file_path}")