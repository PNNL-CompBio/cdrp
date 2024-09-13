from data_utils import DataProcessor, add_smiles, average_dose_response_value, filter_exp_data
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
import matplotlib.pyplot as plt
import numpy as np
# from pathlib import Path
# import coderdata as cd

from sklearn.model_selection import train_test_split
    
    
# def download_data_from_coderdata(dir="./coderdata_input"):
#     #Create Directory, Download Coderdata Data, Return Dir
#     original_dir = os.getcwd()
#     Path(dir).mkdir(parents=True, exist_ok=True)
#     os.chdir(dir)
#     cd.download_data_by_prefix()
#     os.chdir(original_dir)
#     return dir
    
def split_df(df, seed):
    """
    Splits a DataFrame into training and validation sets.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be split.
    - seed (int): The random seed for reproducibility.

    Returns:
    - tuple: A tuple containing two DataFrames:
        - train (pd.DataFrame): The training set.
        - val (pd.DataFrame): The validation set.
    """
    train, val = train_test_split(df, random_state=seed, test_size=0.2, train_size=0.8)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    return train, val

load_trained_model = False
selected_gene_df = pd.read_csv("./shared_input/graphDRP_landmark_genes_map.txt", sep='\t')

def aggregate_duplicate_columns(df):
    """
    Aggregates columns in a DataFrame that have duplicate names by taking their mean values.

    Parameters:
    - df (pd.DataFrame): The DataFrame with potential duplicate columns.

    Returns:
    - pd.DataFrame: A new DataFrame with duplicate columns aggregated and non-duplicated columns retained.
    """
    column_counts = df.columns.value_counts()
    duplicates = column_counts[column_counts > 1]
    aggregated_columns = {}
    non_duplicated_columns = []
    
    for column in duplicates.index:
        duplicate_columns = df.loc[:, df.columns == column]
        aggregated_values = duplicate_columns.mean(axis=1)
        aggregated_columns[column] = aggregated_values
    
    non_duplicated = column_counts[column_counts == 1].index

    for column in non_duplicated:
        non_duplicated_columns.append(df[column])

    aggregated_df = pd.concat([pd.DataFrame(aggregated_columns), pd.concat(non_duplicated_columns, axis=1)], axis=1)

    return aggregated_df
    
def intersect_columns(test_gene_exp, train_gene_exp, selected_gene_df):
    """
    Filters and aggregates columns in test and train gene expression DataFrames based on a common genes and improve_sample_ids.
    Align data with each other by ensuring that gene id and improve sample id both are converted to strings without decimals.

    Parameters:
    - test_gene_exp (pd.DataFrame): The DataFrame containing gene expression data for testing.
    - train_gene_exp (pd.DataFrame): The DataFrame containing gene expression data for training.
    - selected_gene_df (pd.DataFrame): DataFrame containing selected gene names for filtering.

    Returns:
    - tuple: A tuple containing two DataFrames:
        - test_gene_exp_filtered (pd.DataFrame): The filtered and aggregated test gene expression DataFrame.
        - train_gene_exp_filtered (pd.DataFrame): The filtered and aggregated train gene expression DataFrame.
    """
    selected_genes = selected_gene_df.iloc[:, 1].tolist()
    selected_genes_str = [str(gene).rstrip('.0') if str(gene).endswith('.0') else str(gene) for gene in selected_genes]
    test_gene_exp.columns = [str(col).rstrip('.0') if str(col).endswith('.0') else str(col) for col in test_gene_exp.columns]
    train_gene_exp.columns = [str(col).rstrip('.0') if str(col).endswith('.0') else str(col) for col in train_gene_exp.columns]

    common_columns = list(set(selected_genes_str).intersection(set(test_gene_exp.columns), set(train_gene_exp.columns)))
    common_columns = ["improve_sample_id"] + common_columns
    
    test_gene_exp_filtered = test_gene_exp[common_columns]
    train_gene_exp_filtered = train_gene_exp[common_columns]
    
    test_gene_exp_filtered = aggregate_duplicate_columns(test_gene_exp_filtered)
    train_gene_exp_filtered = aggregate_duplicate_columns(train_gene_exp_filtered)
    
    return test_gene_exp_filtered, train_gene_exp_filtered

def predict(model, test_loader, device):
    """
    Uses a model to make predictions on data from the test loader.

    Parameters:
    - model (torch.nn.Module): The trained model for making predictions.
    - test_loader (torch.utils.data.DataLoader): DataLoader providing the test data.
    - device (torch.device): The device (CPU or GPU) to which the data and model should be moved.

    Returns:
    - tuple: A tuple containing two lists:
        - true_values (list): The true labels from the test data.
        - predicted_values (list): The predicted labels from the model.
    """
    true_values = []
    predicted_values = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            predicted_values.extend(outputs.cpu().numpy())
            true_values.extend(data.y.cpu().numpy())
    return true_values, predicted_values

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
                    train_study_description, test_study_description, dose_response_metric,
                    ckpt_path, output_prefix, train_log_transform, test_log_transform
                    ):
    
    def _clean_improve_sample_id(value):
        ''' 
        Helper function to convert improve_sample_id to string and remove decimals 
        '''
        value_str = str(value)
        if value_str.endswith('.0'):
            return value_str[:-2]
        return value_str
    
    # Convert to absolute paths
    test_input_path = os.path.abspath(test_input_path)
    train_input_path = os.path.abspath(train_input_path)
    
    # Now, extract the filename for use in constructing new paths
    test_input_filename = os.path.basename(test_input_path)
    train_input_filename = os.path.basename(train_input_path)  

    # Retrieve wide format of data from shared_input. Otherwise, create it.
    if not os.path.exists(os.path.join("./shared_input", test_input_filename+"_wide.tsv")):
        DataProcessor.convert_long_to_wide_format(test_input_path)
    test_gene_exp = pd.read_csv(os.path.join("./shared_input", test_input_filename+"_wide.tsv"), sep='\t')
    if not os.path.exists(os.path.join("./shared_input", train_input_filename+"_wide.tsv")):
        DataProcessor.convert_long_to_wide_format(train_input_path)
    train_gene_exp = pd.read_csv(os.path.join("./shared_input", train_input_filename+"_wide.tsv"), sep='\t')

    train_gene_exp['improve_sample_id'] = train_gene_exp['improve_sample_id'].apply(_clean_improve_sample_id)
    test_gene_exp['improve_sample_id'] = test_gene_exp['improve_sample_id'].apply(_clean_improve_sample_id)

    test_gene_exp, train_gene_exp = intersect_columns(test_gene_exp, train_gene_exp, selected_gene_df)
    
    # test: process experiment & drug data
    test_exp = pd.read_csv(test_exp_input_path, compression='gzip') if test_exp_input_path.endswith('.gz') else pd.read_csv(test_exp_input_path, sep='\t')
    test_drugs = pd.read_csv(test_drugs_input_path, sep='\t', compression='gzip') if test_drugs_input_path.endswith('.gz') else pd.read_csv(test_drugs_input_path, sep='\t')
    
    # filter by dose_response_metric
    test_exp = test_exp[test_exp['dose_response_metric'] == dose_response_metric]
    
    ## Filter Broad Sanger data down to a single study
    if test_study_description in ['CCLE', 'CTRPv2', 'FIMM', 'gCSI', 'GDSCv1', 'GDSCv2','NCI60','PRISM']:
        test_exp = test_exp[test_exp['study'] == test_study_description]

    # average does response values for the same improve_sample_id and drug_id
    test_exp = average_dose_response_value(test_exp)
    
    # add smiles and split data
    test_df_all = add_smiles(test_drugs, test_exp, "dose_response_value")

    # train: process experiment & drug data
    train_exp = pd.read_csv(train_exp_input_path, compression='gzip') if train_exp_input_path.endswith('.gz') else pd.read_csv(train_exp_input_path, sep='\t')
    train_drugs = pd.read_csv(train_drugs_input_path, sep='\t', compression='gzip') if train_drugs_input_path.endswith('.gz') else pd.read_csv(train_drugs_input_path, sep='\t')
    
    # filter by dose_response_metric
    train_exp = train_exp[train_exp['dose_response_metric'] == dose_response_metric]

    ## Used to filter Broad Sanger data to a single study
    if train_study_description in ['CCLE', 'CTRPv2', 'FIMM', 'gCSI', 'GDSCv1', 'GDSCv2','NCI60','PRISM']:
        train_exp = train_exp[train_exp['study'] == train_study_description]
    
    # average dose response values for the same improve_sample_id and drug_id
    train_exp = average_dose_response_value(train_exp)
    
    # add smiles and split data
    train_df_all = add_smiles(train_drugs, train_exp, "dose_response_value")
    test_df_all['improve_sample_id'] = test_df_all['improve_sample_id'].apply(_clean_improve_sample_id)

    # Find the intersection of improve_sample_id in RNA & drug info
    test_common_ids = set(test_df_all['improve_sample_id']).intersection(set(test_gene_exp['improve_sample_id']))
    
    # Filter RNA & drug to only include rows with improve_sample_id in the intersection
    test_df = test_df_all[test_df_all['improve_sample_id'].isin(test_common_ids)].reset_index(drop=True)
    test_gene_exp = test_gene_exp[test_gene_exp['improve_sample_id'].isin(test_common_ids)]
    test_df.reset_index(drop=True, inplace=True)
    
    # Find the intersection of improve_sample_id in RNA & Drug
    train_gene_exp['improve_sample_id'] = train_gene_exp['improve_sample_id'].apply(_clean_improve_sample_id)
    train_df_all['improve_sample_id'] = train_df_all['improve_sample_id'].apply(_clean_improve_sample_id)
    train_common_ids = set(train_df_all['improve_sample_id'].unique()).intersection(set(train_gene_exp['improve_sample_id'].unique()))
    
    # Filter RNA & Drug data by common Improve_Samples IDs
    train_df = train_df_all[train_df_all['improve_sample_id'].isin(train_common_ids)].reset_index(drop=True)
    train_gene_exp = train_gene_exp[train_gene_exp['improve_sample_id'].isin(train_common_ids)]
    train, val= split_df(df=train_df, seed=data_split_seed)

    #Test Data: Scale and Log transform (if specified) 
    test_gene_exp = test_gene_exp.set_index('improve_sample_id')
    if test_log_transform == True:
        test_gene_exp = np.log1p(test_gene_exp)
    scaler = StandardScaler()
    test_gene_exp_scaled = scaler.fit_transform(test_gene_exp)
    test_gene_exp_scaled = pd.DataFrame(test_gene_exp_scaled, index=test_gene_exp.index, columns=test_gene_exp.columns)
    
    # Define the test dataset
    data_creater = CreateData(gexp=test_gene_exp_scaled, encoder_type='transformer', metric="dose_response_value", data_path= "shared_input/") 
    test_ds = data_creater.create_data(test_df)

    #Train Data: Scale and Log transform (if specified) 
    train_gene_exp = train_gene_exp.set_index('improve_sample_id')
    if train_log_transform == True:
        train_gene_exp = np.log1p(train_gene_exp)
    scaler = StandardScaler() 
    train_gene_exp_scaled = scaler.fit_transform(train_gene_exp)
        train_gene_exp_scaled = pd.DataFrame(train_gene_exp_scaled, index=train_gene_exp.index, columns=train_gene_exp.columns) 
    
    # Define the train and val datasets
    data_creater = CreateData(gexp=train_gene_exp_scaled, encoder_type='transformer', metric="dose_response_value", data_path= "shared_input/") 
    train_ds = data_creater.create_data(train)
    val_ds = data_creater.create_data(val)

    # Configure model specs
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(gnn_features = None, encoder_type='transformer',n_genes=len(test_gene_exp.columns)).to(device)
    adam = torch.optim.Adam(model.parameters(), lr = lr)
    optimizer = adam

    early_stopping = EarlyStopping(patience = n_epochs, verbose=True, chkpoint_name = ckpt_path)
    criterion = nn.MSELoss()

        
    # Create a dictionary to store losses for training and validation
    history = {
    "train_loss": [],
    "val_loss": []}
        
    # Train the model
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

            loss_all += loss.item()  # Accumulate loss
            
        # Calculate average training loss for this epoch
        train_loss = loss_all / len(train_loader)
        history["train_loss"].append(train_loss)  # Store training loss
        val_rmse, _, _, _, _ = test_fn(val_loader, model, device)
        val_loss = val_rmse 
        history["val_loss"].append(val_loss)  # Store validation loss
        early_stopping(val_rmse, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f'Epoch: {epoch}, Val_rmse: {val_rmse:.3}')
        if epoch % 33 == 0:
            model_save_path = f'models/{output_prefix}_model_seed_{data_split_seed}_epoch_{epoch}.pt'
            torch.save(model.state_dict(), model_save_path) # save the model for each seed & epoch
            print("Model saved at", model_save_path)
    
    model.load_state_dict(torch.load(ckpt_path))
    test_rmse, pearson_corr, spearman_corr, _, _= test_fn(test_loader, model, device)
    # Extract the losses from history
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        print(f"'{plots_folder}' directory created.")
        
    # Create plots
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(train_loss)), train_loss, label='Train Loss', linestyle='-', color='b')
    plt.plot(range(0, len(val_loss)), val_loss, label='Val Loss', linestyle='-', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plots_folder}/seed_{data_split_seed}_epoch_{n_epochs}_{output_prefix}_auc_train_val_plot.png')

    return test_rmse, pearson_corr, spearman_corr

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
    parser.add_argument('--train_study_description', type=str, default='CCLE', help='For broad studies, specify the study name: CCLE or PRISM')
    parser.add_argument('--test_study_description', type=str, default='CCLE', help='For broad studies, specify the study name: CCLE or PRISM')
    parser.add_argument('--dose_response_metric', type=str, default='fit_auc', help='Choose dose response metric: fit_auc or fit_ic50')
    parser.add_argument('--checkpoint_path', type=str, default='/people/moon515/mpnst_smile_model/tmp/best.pt', help='Path to the model checkpoint to evaluate')
    parser.add_argument('--train_log_transform', type=bool, default=False, help='Whether to log-transform the training data')
    parser.add_argument('--test_log_transform', type=bool, default=False, help='Whether to log-transform the test data')

    # Convert argparse arguments to variables
    args = parser.parse_args()
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
    train_study_description = args.train_study_description
    test_study_description = args.test_study_description 
    dose_response_metric = args.dose_response_metric
    ckpt_path = args.checkpoint_path
    train_log_transform = args.train_log_transform
    test_log_transform = args.test_log_transform

    load_trained_model = False  # Assuming this is set elsewhere or needs to be added as an argparse argument
    
    # File path for saving the models
    models_folder = "models"
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
        print(f"'{models_folder}' directory created.")
        
    # File path for saving the results
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"'{results_folder}' directory created.")
        
    # File path for saving tmp data
    tmp_folder = "tmp"
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
        print(f"'{tmp_folder}' directory created.")
            
    # Run experiment on each seed and save results.
    results = {}
    for seed in range(1, data_split_seed + 1):
        test_rmse, pearson_corr, spearman_corr = run_experiment(data_split_seed=seed,
                                    bs=bs, lr=lr,
                                    n_epochs=n_epochs,
                                    train_input_path=train_input_path,
                                    train_exp_input_path=train_exp_input_path,
                                    train_drugs_input_path=train_drugs_input_path,
                                    test_input_path=test_input_path,
                                    test_exp_input_path= test_exp_input_path,
                                    test_drugs_input_path= test_drugs_input_path,
                                    load_trained_model=load_trained_model,
                                    output_prefix=output_prefix,
                                    train_study_description=train_study_description,
                                    test_study_description=test_study_description,
                                    dose_response_metric=dose_response_metric,
                                    ckpt_path=ckpt_path,
                                    train_log_transform=train_log_transform,
                                    test_log_transform=test_log_transform)
       
        #Store results
        results[seed] = test_rmse, pearson_corr, spearman_corr 
        
        #Write Results to file
        results_file_path = f'{results_folder}/seed_{data_split_seed}_epoch_{n_epochs}_{output_prefix}_rna_train_results_table.txt'
        with open(results_file_path, 'w') as file:
            file.write("Results for different seeds:\n")
            file.write("Seed\tTest RMSE\tPearson Correlation\tSpearman Correlation\n")
            for seed, (test_rmse, pearson_corr, spearman_corr) in results.items():
                file.write(f"{seed}\t{test_rmse:.3f}\t{pearson_corr:.3f}\t{spearman_corr:.3f}\n")

        print(f"Results saved to {results_file_path}")

if __name__ == "__main__":
    main()