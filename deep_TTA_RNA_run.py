# script that will run the deep learning model with TTA
from data_utils import DataProcessor, add_smiles, average_auc, average_dose_response_value, filter_exp_data
from gnn_utils import CreateData, EarlyStopping, test_fn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from deeptta_rna_model import Model
import pandas as pd 
import os
import argparse
def split_df(df, seed):
    train, val = train_test_split(df, random_state=seed, test_size=0.2, train_size=0.8)

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    return train, val

# all the parameters are set here
# data_split_seed = 10 # adjust seed here
# bs = 64
# lr = 1e-4
# n_epochs = 100 # test with small number of epochs 
# test_input_path = "./coderdata_input/MPNST_RNA_seq.csv.gz"
# train_input_path = "./coderdata_input/beataml_transcriptomics.csv.gz"
# if you run the model for the first time, set load_trained_model to False
load_trained_model = False
selected_gene_df = pd.read_csv("./shared_input/graphDRP_landmark_genes_map.txt", sep='\t')
def intersect_columns(test_gene_exp, train_gene_exp, selected_gene_df):
    # Extract the gene names from the second column of selected_gene_df
    selected_genes = selected_gene_df.iloc[:, 1].tolist()
    # Convert gene IDs to string because DataFrame columns are strings
    selected_genes_str = [str(float(gene)) for gene in selected_genes]
    # Find common genes between selected_genes and the columns of the input DataFrames
    common_columns = list(set(selected_genes_str).intersection(set(test_gene_exp.columns), set(train_gene_exp.columns)))
    # Ensure "improve_sample_id" is included if it exists in both DataFrames
    if "improve_sample_id" in test_gene_exp.columns and "improve_sample_id" in train_gene_exp.columns:
        common_columns = ["improve_sample_id"] + common_columns
    # Subset both DataFrames to only include the common columns
    test_gene_exp_filtered = test_gene_exp[common_columns]
    train_gene_exp_filtered = train_gene_exp[common_columns]
    return test_gene_exp_filtered, train_gene_exp_filtered

##############################
# DATA PROCESSING
##############################
# Process gene expression data
# Downloads and saves the gene expression data into a separate directory "process_input"
def run_experiment(data_split_seed, bs, lr, n_epochs,
                    train_input_path,
                    train_exp_input_path,
                    train_drugs_input_path,
                    test_input_path,
                    test_exp_input_path,
                    test_drugs_input_path, load_trained_model,
                    study_description, dose_response_metric
                    ):
    
    # Convert to absolute paths
    test_input_path = os.path.abspath(test_input_path)
    train_input_path = os.path.abspath(train_input_path)
    
    # Now, extract the filename for use in constructing new paths
    test_input_filename = os.path.basename(test_input_path)
    train_input_filename = os.path.basename(train_input_path)

    # # Create tmp directory if it doesn't exist, using the absolute path
    # tmp_dir = "./tmp"
    # if not os.path.exists(tmp_dir):
    #     os.makedirs(tmp_dir)    

    if not os.path.exists(os.path.join("./shared_input", test_input_filename+"_wide.tsv")):
        DataProcessor.convert_long_to_wide_format(test_input_path)
    test_gene_exp = pd.read_csv(os.path.join("./shared_input", test_input_filename+"_wide.tsv"), sep='\t')

    if not os.path.exists(os.path.join("./shared_input", train_input_filename+"_wide.tsv")):
        DataProcessor.convert_long_to_wide_format(train_input_path)
    train_gene_exp = pd.read_csv(os.path.join("./shared_input", train_input_filename+"_wide.tsv"), sep='\t')
    test_gene_exp, train_gene_exp = intersect_columns(test_gene_exp, train_gene_exp, selected_gene_df)

    # test: process experiment & drug data
    test_exp = pd.read_csv(test_exp_input_path, compression='gzip') if test_exp_input_path.endswith('.gz') else pd.read_csv(test_exp_input_path, sep='\t')
    test_drugs = pd.read_csv(test_drugs_input_path, sep='\t', compression='gzip') if test_drugs_input_path.endswith('.gz') else pd.read_csv(test_drugs_input_path, sep='\t')
    # average auc values for the same improve_sample_id and drug_id
    test_exp = average_auc(test_exp) #outdated
    # test_exp = average_dose_response_value(test_exp)
    # add smiles and split data
    test_df_all = add_smiles(test_drugs, test_exp, "auc") #outdated
    # test_df_all = add_smiles(test_drugs, test_exp, "dose_response_value")
    # train: process experiment & drug data
    train_exp = pd.read_csv(train_exp_input_path, compression='gzip') if train_exp_input_path.endswith('.gz') else pd.read_csv(train_exp_input_path, sep='\t')
    train_drugs = pd.read_csv(train_drugs_input_path, sep='\t', compression='gzip') if train_drugs_input_path.endswith('.gz') else pd.read_csv(train_drugs_input_path, sep='\t')
    
    # Use the function in your main logic
    try:
        train_exp = filter_exp_data(train_exp, study_description,dose_response_metric)
    except ValueError as e:
        print(e)
    
    # average auc values for the same improve_sample_id and drug_id
    # train_exp = average_auc(train_exp) #outdated
    train_exp = average_dose_response_value(train_exp)
    # add smiles and split data
    # train_df_all = add_smiles(train_drugs, train_exp, "auc") #outdated
    train_df_all = add_smiles(train_drugs, train_exp, "dose_response_value")
    # merge and split the data
    # Find the intersection of improve_sample_id in RNA & drug info
    test_common_ids = set(test_df_all['improve_sample_id']).intersection(set(test_gene_exp['improve_sample_id']))
    # Filter RNA & drug to only include rows with improve_sample_id in the intersection
    test_df = test_df_all[test_df_all['improve_sample_id'].isin(test_common_ids)].reset_index(drop=True) # this is EMPTY FIX!!!
    test_gene_exp = test_gene_exp[test_gene_exp['improve_sample_id'].isin(test_common_ids)]
    # test_df = test_df.reset_index(drop=True, inplace=True)
    test = test_df.reset_index(drop=True, inplace=True)

    # Find the intersection of improve_sample_id in RNA & drug
    train_common_ids = set(train_df_all['improve_sample_id'].unique()).intersection(set(train_gene_exp['improve_sample_id'].unique()))
    # Filter RNA & drug to only include rows with improve_sample_id in the intersection
    train_df = train_df_all[train_df_all['improve_sample_id'].isin(train_common_ids)].reset_index(drop=True)
    train_gene_exp = train_gene_exp[train_gene_exp['improve_sample_id'].isin(train_common_ids)]
    train, val= split_df(df=train_df, seed=data_split_seed)

    # test: Ensure improve_sample_id is set as the index before scaling
    test_gene_exp = test_gene_exp.set_index('improve_sample_id')
    # Now perform the scaling operation on the DataFrame without the index column
    scaler = StandardScaler() # mean=0, unit variance
    test_gene_exp_scaled = scaler.fit_transform(test_gene_exp)
    # When creating the new DataFrame, use the same columns as the gene_exp DataFrame
    # Because gene_exp now does not include 'improve_sample_id' column, we don't need to adjust column names
    test_gene_exp_scaled = pd.DataFrame(test_gene_exp_scaled, index=test_gene_exp.index, columns=test_gene_exp.columns)
    data_creater = CreateData(gexp=test_gene_exp_scaled, encoder_type='transformer', metric="auc", data_path= "shared_input/")
    test_ds = data_creater.create_data(test_df)

    # bealaml: Ensure improve_sample_id is set as the index before scaling
    train_gene_exp = train_gene_exp.set_index('improve_sample_id')
    # Now perform the scaling operation on the DataFrame without the index column
    scaler = StandardScaler() # mean=0, unit variance
    train_gene_exp_scaled = scaler.fit_transform(train_gene_exp)
    # When creating the new DataFrame, use the same columns as the gene_exp DataFrame
    # Because gene_exp now does not include 'improve_sample_id' column, we don't need to adjust column names
    train_gene_exp_scaled = pd.DataFrame(train_gene_exp_scaled, index=train_gene_exp.index, columns=train_gene_exp.columns) 
    data_creater = CreateData(gexp=train_gene_exp_scaled, encoder_type='transformer', metric="dose_response_value", data_path= "shared_input/") # metric needs to be adjusted based on dose_response_metric
    # bealaml: define the train and val datasets
    train_ds = data_creater.create_data(train)
    val_ds = data_creater.create_data(val)

    # bs = 64
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(gnn_features = None, encoder_type='transformer',n_genes=len(test_gene_exp.columns)).to(device)
    # lr = 1e-4
    adam = torch.optim.Adam(model.parameters(), lr = lr)
    optimizer = adam

    # n_epochs = 100
    ckpt_path = './tmp/best.pt'

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

def main():
    parser = argparse.ArgumentParser(description='Deep TTA RNA Model')
    parser.add_argument('--data_split_seed', type=int, default=10, help='Seed for data split')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--train_omics_input_path', type=str, default='./coderdata_input/beataml_transcriptomics.csv.gz', help='Path to train omics input data')
    parser.add_argument('--train_exp_input_path', type=str, default='./coderdata_input/train_exp.csv.gz', help='Path to train experiment input data')
    parser.add_argument('--train_drugs_input_path', type=str, default='./coderdata_input/train_drugs.tsv.gz', help='Path to train drugs input data')
    parser.add_argument('--test_omics_input_path', type=str, default='./coderdata_input/MPNST_RNA_seq.csv.gz', help='Path to test omics input data')
    parser.add_argument('--test_exp_input_path', type=str, default='./coderdata_input/MPNST_experiments.csv.gz', help='Path to test experiment input data')
    parser.add_argument('--test_drugs_input_path', type=str, default='./coderdata_input/MPNST_drugs.tsv.gz', help='Path to test drugs input data')
    parser.add_argument('--output_prefix', type=str, default='broad_CCLE', help='describe the study; output file will be named accordingly')
    parser.add_argument('--study_description', type=str, default='CCLE', help='For broad studies, specify the study name: CCLE or PRISM')
    parser.add_argument('--dose_response_metric', type=str, default='fit_auc', help='Choose dose response metric: fit_auc or fit_ic50')
    
    args = parser.parse_args()
    # Convert argparse arguments to variables
    data_split_seed = args.data_split_seed
    bs = args.bs
    lr = args.lr
    n_epochs = args.n_epochs
    train_input_path = args.train_omics_input_path
    train_exp_input_path = args.train_exp_input_path
    train_drugs_input_path = args.train_drugs_input_path
    test_input_path = args.test_omics_input_path
    test_exp_input_path = args.test_exp_input_path
    test_drugs_input_path = args.test_drugs_input_path
    output_prefix = args.output_prefix
    study_description = args.study_description
    dose_response_metric = args.dose_response_metric

    load_trained_model = False  # Assuming this is set elsewhere or needs to be added as an argparse argument

    # Call run_experiment or any other logic meant to run when the script is executed directly
    results = {}
    for seed in range(1, data_split_seed + 1):
        test_rmse = run_experiment(data_split_seed=seed,
                                    bs=bs, lr=lr,
                                    n_epochs=n_epochs,
                                    train_input_path=train_input_path,
                                    train_exp_input_path=train_exp_input_path,
                                    train_drugs_input_path=train_drugs_input_path,
                                    test_input_path=test_input_path,
                                    test_exp_input_path= test_exp_input_path,
                                    test_drugs_input_path= test_drugs_input_path,
                                    load_trained_model=load_trained_model,
                                    study_description=study_description,
                                    dose_response_metric=dose_response_metric)
        results[seed] = test_rmse

    # File path for saving the results
    results_file_path = f'seed_{data_split_seed}_epoch_{n_epochs}_{output_prefix}_rna_train_results_table.txt'

    # Saving results to a file
    with open(results_file_path, 'w') as file:
        file.write("Test RMSE for different seeds:\n")
        file.write("Seed\tTest RMSE\n")
        for seed, test_rmse in results.items():
            file.write(f"{seed}\t{test_rmse}\n")

    print(f"Results saved to {results_file_path}")

if __name__ == "__main__":
    main()