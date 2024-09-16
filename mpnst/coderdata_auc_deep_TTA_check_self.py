# this script will be used primarily to check the performance of the deep learning model on the CCLE dataset
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

from sklearn.model_selection import train_test_split

def split_df(df, seed):
    """
    Splits a DataFrame into training, validation, and test sets.

    The function first splits the DataFrame into 70% training and 30% test sets. Then, it further splits the training set into 50% training and 50% validation sets. The indices of all splits are reset.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        seed (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing three DataFrames:
            - train (pd.DataFrame): The training set.
            - val (pd.DataFrame): The validation set.
            - test (pd.DataFrame): The test set.
    """
    train, test = train_test_split(df, random_state=seed, test_size=0.3)
    train, val = train_test_split(train, random_state=seed, test_size=0.5)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    return train, val, test

selected_gene_df = pd.read_csv("./shared_input/graphDRP_landmark_genes_map.txt", sep='\t')

def intersect_columns(train_gene_exp, selected_gene_df):
    """
    Filters columns of a DataFrame to include only those that are present in a list of selected genes.

    This function takes a DataFrame containing gene expression data and filters its columns based on a list of selected genes provided in another DataFrame. It ensures that gene identifiers are compared in a consistent format and includes the "improve_sample_id" column if it exists.

    Args:
        train_gene_exp (pd.DataFrame): The DataFrame with gene expression data.
        selected_gene_df (pd.DataFrame): The DataFrame containing selected gene identifiers.

    Returns:
        pd.DataFrame: A DataFrame containing only the columns that are present in the list of selected genes.
    """
    selected_genes = selected_gene_df.iloc[:, 1].tolist()
    selected_genes_str = [str(gene).rstrip('.0') if str(gene).endswith('.0') else str(gene) for gene in selected_genes]
    train_gene_exp.columns = [str(col).rstrip('.0') if str(col).endswith('.0') else str(col) for col in train_gene_exp.columns]
    common_columns = list(set(selected_genes_str).intersection(set(train_gene_exp.columns)))

    if "improve_sample_id" in train_gene_exp.columns:
        common_columns = ["improve_sample_id"] + common_columns
        
    train_gene_exp_filtered = train_gene_exp[common_columns]
    
    return train_gene_exp_filtered

def predict(model, test_loader, device):
    """
    Makes predictions using a trained model on a test dataset.

    This function iterates over a DataLoader containing test data, performs inference using the provided model, and collects the predicted and true values.

    Args:
        model (torch.nn.Module): The trained model used for making predictions.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform the computations.

    Returns:
        tuple: A tuple containing two lists:
            - true_values (list): The true labels from the test dataset.
            - predicted_values (list): The predicted values from the model.
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
                    study_description, dose_response_metric,
                    ckpt_path, output_prefix, log_transform
                    ):
    
    # Convert to absolute paths
    train_input_path = os.path.abspath(train_input_path)
    # Now, extract the filename for use in constructing new paths
    train_input_filename = os.path.basename(train_input_path)   

    if not os.path.exists(os.path.join("./shared_input", train_input_filename+"_wide.tsv")):
        DataProcessor.convert_long_to_wide_format(train_input_path)
    train_gene_exp = pd.read_csv(os.path.join("./shared_input", train_input_filename+"_wide.tsv"), sep='\t')
    
    train_gene_exp = intersect_columns(train_gene_exp, selected_gene_df)
    
    # train: process experiment & drug data
    train_exp = pd.read_csv(train_exp_input_path, compression='gzip') if train_exp_input_path.endswith('.gz') else pd.read_csv(train_exp_input_path, sep='\t')
    train_drugs = pd.read_csv(train_drugs_input_path, sep='\t', compression='gzip') if train_drugs_input_path.endswith('.gz') else pd.read_csv(train_drugs_input_path, sep='\t')
    
    # filter exp data based on the stated study description and does response metric (if the dataset is from broad_sanger)
    try:
        train_exp = filter_exp_data(train_exp, study_description, dose_response_metric)
    except ValueError as e:
        print(e)
    
    # Average auc values for the same improve_sample_id and drug_id
    train_exp = average_dose_response_value(train_exp)
    
    # Add smiles and split data
    train_df_all = add_smiles(train_drugs, train_exp, "dose_response_value")

    # Find the intersection of improve_sample_id in RNA & Drug
    train_common_ids = set(train_df_all['improve_sample_id'].unique()).intersection(set(train_gene_exp['improve_sample_id'].unique()))
    
    # Filter RNA & Drug data by common Improve_Samples IDs
    train_df = train_df_all[train_df_all['improve_sample_id'].isin(train_common_ids)].reset_index(drop=True)
    train_gene_exp = train_gene_exp[train_gene_exp['improve_sample_id'].isin(train_common_ids)]
    
    train, val, test= split_df(df=train_df, seed=data_split_seed)

    #Training Data: Scale and Log transform (if specified) 
    train_gene_exp = train_gene_exp.set_index('improve_sample_id')
    if log_transform == True:
        train_gene_exp = np.log1p(train_gene_exp)
    scaler = StandardScaler()
    train_gene_exp_scaled = scaler.fit_transform(train_gene_exp)
    
    # When creating the new DataFrame, use the same columns as the gene_exp DataFrame
    train_gene_exp_scaled = pd.DataFrame(train_gene_exp_scaled, index=train_gene_exp.index, columns=train_gene_exp.columns) 
    
    # Remove columns with NaN values
    train_gene_exp_scaled = train_gene_exp_scaled.dropna(axis=1)

    data_creater = CreateData(gexp=train_gene_exp_scaled, encoder_type='transformer', metric="dose_response_value", data_path= "shared_input/") 
    
    # Define the train and val datasets
    train_ds = data_creater.create_data(train)
    val_ds = data_creater.create_data(val)
    test_ds = data_creater.create_data(test)

    # Configure model specs
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(gnn_features = None, encoder_type='transformer',n_genes=len(train_gene_exp_scaled.columns)).to(device)
    adam = torch.optim.Adam(model.parameters(), lr = lr)
    optimizer = adam

    early_stopping = EarlyStopping(patience = n_epochs, verbose=True, chkpoint_name = ckpt_path)
    criterion = nn.MSELoss()

    # Create a dictionary to store losses for training and validation
    history = {
    "train_loss": [],
    "val_loss": []}

    # Train the Model  
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

            loss_all += loss.item()
            
        # Calculate average training loss for this epoch
        if len(train_loader) == 0:
            train_loss = 0
        else:
            train_loss = loss_all / len(train_loader)
        
        history["train_loss"].append(train_loss)  # Store training loss

        val_rmse, _, _, _, _ = test_fn(val_loader, model, device)
        val_loss = val_rmse 
        history["val_loss"].append(val_loss)  # Store validation loss
        early_stopping(val_rmse, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        hist["val_rmse"].append(val_rmse)
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

    # Check for and create plots directory
    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        print(f"'{plots_folder}' directory created.")
        
    # Create plot for current status of model - This generates 3 plots for each seed.
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(train_loss)), train_loss, label='Train Loss', linestyle='-', color='b')
    plt.plot(range(0, len(val_loss)), val_loss, label='Val Loss', linestyle='-', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/seed_{data_split_seed}_epoch_{n_epochs}_{output_prefix}_auc_train_val_plot.png')

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
    parser.add_argument('--output_prefix', type=str, default='broad_CCLE', help='describe the study; output file will be named accordingly')
    parser.add_argument('--study_description', type=str, default='CCLE', help='For broad studies, specify the study name: CCLE or PRISM')
    parser.add_argument('--dose_response_metric', type=str, default='fit_auc', help='Choose dose response metric: fit_auc or fit_ic50')
    parser.add_argument('--checkpoint_path', type=str, default='models/model_seed_1_epoch_99.pt', help='Path to the model checkpoint to evaluate')
    parser.add_argument('--log_transform', type=bool, default=False, help='Whether to log-transform the training data')
    
    # Convert argparse arguments to variables
    args = parser.parse_args()
    data_split_seed = args.data_split_seed
    bs = args.bs
    lr = args.lr
    n_epochs = args.n_epochs
    train_input_path = args.train_omics_input_path
    train_exp_input_path = args.train_exp_input_path
    train_drugs_input_path = args.train_drugs_input_path
    output_prefix = args.output_prefix
    study_description = args.study_description
    dose_response_metric = args.dose_response_metric
    ckpt_path = args.checkpoint_path
    log_transform = args.log_transform

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

    # Call run_experiment or any other logic meant to run when the script is executed directly
    results = {}
    for seed in range(1, data_split_seed + 1):
        test_rmse, pearson_corr, spearman_corr = run_experiment(data_split_seed=seed,
                                    bs=bs, lr=lr,
                                    n_epochs=n_epochs,
                                    train_input_path=train_input_path,
                                    train_exp_input_path=train_exp_input_path,
                                    train_drugs_input_path=train_drugs_input_path,
                                    output_prefix=output_prefix,
                                    study_description=study_description,
                                    dose_response_metric=dose_response_metric,
                                    ckpt_path=ckpt_path,
                                    log_transform=log_transform)
        results[seed] = test_rmse, pearson_corr, spearman_corr 
            
        # File path for saving the results
        results_file_path = f'{results_folder}/seed_{data_split_seed}_epoch_{n_epochs}_{output_prefix}_rna_train_results_table.txt'

        with open(results_file_path, 'w') as file:
            file.write("Results for different seeds:\n")
            file.write("Seed\tTest RMSE\tPearson Correlation\tSpearman Correlation\n")
            for seed, (test_rmse, pearson_corr, spearman_corr) in results.items():
                file.write(f"{seed}\t{test_rmse:.3f}\t{pearson_corr:.3f}\t{spearman_corr:.3f}\n")

        print(f"Results saved to {results_file_path}")

if __name__ == "__main__":
    main()