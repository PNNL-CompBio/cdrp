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
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

def calculate_statistics(true_values, predicted_values):
    # Ensure both are numpy arrays and flatten predicted_values if it's a list of arrays
    true_values = np.array(true_values)
    if isinstance(predicted_values[0], np.ndarray):
        # Flatten the array if predicted_values consists of arrays (multiple outputs per instance)
        predicted_values = np.concatenate(predicted_values).ravel()
    else:
        predicted_values = np.array(predicted_values)

    # Calculate Pearson correlation coefficient
    if len(true_values) > 1 and len(predicted_values) > 1:  # Ensure there's enough data to calculate
        pcc, _ = pearsonr(true_values, predicted_values)
    else:
        pcc = np.nan  # Not enough data to calculate correlation

    # Calculate Spearman's rank correlation coefficient
    if len(true_values) > 1 and len(predicted_values) > 1:
        srcc, _ = spearmanr(true_values, predicted_values)
    else:
        srcc = np.nan

    # Calculate sample size
    sample_size = len(true_values)

    return pcc, srcc, sample_size


# load trained model and evaluate and plot the results
load_trained_model = True

def predict(model, test_loader, device):
    true_values = []
    predicted_values = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            predicted_values.extend(outputs.cpu().numpy())
            true_values.extend(data.y.cpu().numpy())
    return true_values, predicted_values

def plot_results(true_values, predicted_values, ckpt_path, test_df):
    # Flatten the predicted values list of lists
    predicted_values = [item[0] for item in predicted_values]
    # Debugging: Print lengths to identify inconsistencies
    print(f'Length of true_values: {len(true_values)}')
    print(f'Length of predicted_values: {len(predicted_values)}')
    print(f'Length of test_df: {len(test_df)}')
    # Calculate statistics
    pcc, srcc, sample_size = calculate_statistics(true_values, predicted_values)

    # Ensure the lengths match
    if len(true_values) != len(predicted_values):
        raise ValueError('Mismatch in length between true_values and predicted_values')
    if len(true_values) != len(test_df):
        raise ValueError('Mismatch in length between true_values/predicted_values and test_df')

    # Convert improve_sample_id and improve_drug_id to appropriate types if necessary
    test_df['improve_sample_id'] = test_df['improve_sample_id'].apply(str)
    test_df['improve_drug_id'] = test_df['improve_drug_id'].apply(str)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'True dss': true_values,
        'Predicted dss': predicted_values,
        'improve_sample_id': test_df['improve_sample_id'],
        'improve_drug_id': test_df['improve_drug_id']
    })
    print(plot_df)

    # Define enough unique filled markers
    # markers = ['o', 's', 'P', 'X', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_', '8', '1', '2', '3', '4']

    # Plotting
    plt.figure(figsize=(14, 10))
    # sns.scatterplot(data=plot_df, x='True dss', y='Predicted dss', hue='improve_sample_id', style='improve_drug_id', palette='deep', alpha=0.5)
    sns.scatterplot(
    data=plot_df, 
    x='True dss', 
    y='Predicted dss', 
    hue='improve_sample_id', 
    style='improve_drug_id', 
    palette='deep', 
    alpha=0.7, 
    s=100,  # Increase marker size
    # markers=markers  # Use the extended markers
    )
    plt.title(f'Comparison of True and Predicted dss\nCheckpoint: {ckpt_path}\nPCC: {pcc:.3f}, SRCC: {srcc:.3f}, Sample Size: {sample_size}')
    plt.xlabel('True dss')
    plt.ylabel('Predicted dss')
    plt.grid(True)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')

    # Adjust legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)  # Move legend outside the plot with two columns

    plt.tight_layout()
    plt.show()
    
    # Save the plot as a PDF
    ckpt_basename = os.path.basename(ckpt_path)
    plt.savefig(f'plots/zzz_rna_{ckpt_basename}_true_vs_predicted.pdf')

def split_df(df, seed):
    train, val = train_test_split(df, random_state=seed, test_size=0.2, train_size=0.8)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    return train, val

# load_trained_model = False
selected_gene_df = pd.read_csv("./shared_input/graphDRP_landmark_genes_map.txt", sep='\t')
def intersect_columns(test_gene_exp, train_gene_exp, selected_gene_df):
    # Extract the gene names from the second column of selected_gene_df
    selected_genes = selected_gene_df.iloc[:, 1].tolist()
    # Convert gene IDs to string because DataFrame columns are strings
    selected_genes_str = [str(float(gene)) for gene in selected_genes]
    # Convert test_gene_exp columns to strings and ensure they end with .0 except for column named "improve_sample_id"
    # test_gene_exp.columns = [str(col) if col == "improve_sample_id" else str(col) if '.' in str(col) else str(col) + '.0' for col in test_gene_exp.columns]
    test_gene_exp.columns = [str(col) for col in test_gene_exp.columns]
    # Convert train_gene_exp columns to strings if necessary
    train_gene_exp.columns = [str(col) for col in train_gene_exp.columns]
    # Find common genes between selected_genes and the columns of the input DataFrames
    common_columns = list(set(selected_genes_str).intersection(set(test_gene_exp.columns), set(train_gene_exp.columns)))
    # Ensure "improve_sample_id" is included if it exists in both DataFrames
    # if "improve_sample_id" in test_gene_exp.columns and "improve_sample_id" in train_gene_exp.columns:
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
                    test_drugs_input_path,
                    train_study_description, test_study_description, dose_response_metric,
                    ckpt_path, output_prefix, train_log_transform, test_log_transform,
                    load_trained_model=True, filter_drugs=None
                    ):
    
    # Convert to absolute paths
    test_input_path = os.path.abspath(test_input_path)
    train_input_path = os.path.abspath(train_input_path)
    
    # Now, extract the filename for use in constructing new paths
    test_input_filename = os.path.basename(test_input_path)
    train_input_filename = os.path.basename(train_input_path)   

    if not os.path.exists(os.path.join("./shared_input", test_input_filename+"_wide.tsv")):
        DataProcessor.convert_long_to_wide_format(test_input_path)
    test_gene_exp = pd.read_csv(os.path.join("./shared_input", test_input_filename+"_wide.tsv"), sep='\t')

    if not os.path.exists(os.path.join("./shared_input", train_input_filename+"_wide.tsv")):
        DataProcessor.convert_long_to_wide_format(train_input_path)
    train_gene_exp = pd.read_csv(os.path.join("./shared_input", train_input_filename+"_wide.tsv"), sep='\t')

    if train_study_description == 'MPNST':
        new_columns = ['improve_sample_id'] + [str(col) + '.0' for col in train_gene_exp.columns[1:]]
        train_gene_exp.columns = new_columns

    if test_study_description == 'MPNST':
        new_columns = ['improve_sample_id'] + [str(col) + '.0' for col in test_gene_exp.columns[1:]]
        test_gene_exp.columns = new_columns
    test_gene_exp, train_gene_exp = intersect_columns(test_gene_exp, train_gene_exp, selected_gene_df)

    # test: process experiment & drug data
    test_exp = pd.read_csv(test_exp_input_path, compression='gzip') if test_exp_input_path.endswith('.gz') else pd.read_csv(test_exp_input_path, sep='\t')
    
    # filter drugs if specified
    # if filter_drugs != None:
    #     test_exp = test_exp[test_exp['improve_drug_id'] == filter_drugs]

    test_drugs = pd.read_csv(test_drugs_input_path, sep='\t', compression='gzip') if test_drugs_input_path.endswith('.gz') else pd.read_csv(test_drugs_input_path, sep='\t')
    # filter by dose_response_metric
    test_exp = test_exp[test_exp['dose_response_metric'] == dose_response_metric]
    # filter exp data based on the stated study
    if test_study_description != 'MPNST':
        test_exp = test_exp[test_exp['study'] == test_study_description]
    # average does response values for the same improve_sample_id and drug_id
    test_exp = average_dose_response_value(test_exp)
    # add smiles and split data
    test_df_all = add_smiles(test_drugs, test_exp, "dose_response_value")
    
    # test_exp = filter_exp_data(test_exp, test_study_description,dose_response_metric)
    # try:
    #     test_exp = filter_exp_data(test_exp, test_study_description,dose_response_metric)
    # except ValueError as e:
    #     print(e)

    # train: process experiment & drug data
    train_exp = pd.read_csv(train_exp_input_path, compression='gzip') if train_exp_input_path.endswith('.gz') else pd.read_csv(train_exp_input_path, sep='\t')
    train_drugs = pd.read_csv(train_drugs_input_path, sep='\t', compression='gzip') if train_drugs_input_path.endswith('.gz') else pd.read_csv(train_drugs_input_path, sep='\t')
    # filter by dose_response_metric
    train_exp = train_exp[train_exp['dose_response_metric'] == dose_response_metric]
    # filter exp data based on the stated study
    train_exp = train_exp[train_exp['study'] == train_study_description]
    # filter exp data based on the stated study
    if train_study_description != 'MPNST':
        train_exp = train_exp[train_exp['study'] == train_study_description]
    # average does response values for the same improve_sample_id and drug_id
    train_exp = average_dose_response_value(train_exp)
    # add smiles and split data
    train_df_all = add_smiles(train_drugs, train_exp, "dose_response_value")
    
    # # filter exp data based on the stated study description and does response metric
    # try:
    #     train_exp = filter_exp_data(train_exp, train_study_description,dose_response_metric)
    # except ValueError as e:
    #     print(e)
    
    # average auc values for the same improve_sample_id and drug_id
    train_exp = average_dose_response_value(train_exp)
    # add smiles and split data
    train_df_all = add_smiles(train_drugs, train_exp, "dose_response_value")
    # merge and split the data
    # Find the intersection of improve_sample_id in RNA & drug info
    test_common_ids = set(test_df_all['improve_sample_id']).intersection(set(test_gene_exp['improve_sample_id']))
    # Filter RNA & drug to only include rows with improve_sample_id in the intersection
    test_df = test_df_all[test_df_all['improve_sample_id'].isin(test_common_ids)].reset_index(drop=True)
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
    # if the training data requires log_transform
    if test_log_transform == True:
        test_gene_exp = np.log1p(test_gene_exp)
    # Now perform the scaling operation on the DataFrame without the index column
    # scaler = StandardScaler() # mean=0, unit variance
    # test_gene_exp_scaled = scaler.fit_transform(test_gene_exp)
    test_gene_exp_scaled = test_gene_exp
    # When creating the new DataFrame, use the same columns as the gene_exp DataFrame
    # Because gene_exp now does not include 'improve_sample_id' column, we don't need to adjust column names
    test_gene_exp_scaled = pd.DataFrame(test_gene_exp_scaled, index=test_gene_exp.index, columns=test_gene_exp.columns)
    data_creater = CreateData(gexp=test_gene_exp_scaled, encoder_type='transformer', metric="dose_response_value", data_path= "shared_input/") 
    test_ds = data_creater.create_data(test_df)

    # Ensure improve_sample_id is set as the index before scaling
    train_gene_exp = train_gene_exp.set_index('improve_sample_id')
    # if the training data requires log_transform
    if train_log_transform == True:
        train_gene_exp = np.log1p(train_gene_exp)
    # Now perform the scaling operation on the DataFrame without the index column
    scaler = StandardScaler() # mean=0, unit variance
    train_gene_exp_scaled = scaler.fit_transform(train_gene_exp)
    # When creating the new DataFrame, use the same columns as the gene_exp DataFrame
    # Because gene_exp now does not include 'improve_sample_id' column, we don't need to adjust column names
    train_gene_exp_scaled = pd.DataFrame(train_gene_exp_scaled, index=train_gene_exp.index, columns=train_gene_exp.columns) 
    data_creater = CreateData(gexp=train_gene_exp_scaled, encoder_type='transformer', metric="dose_response_value", data_path= "shared_input/") 
    # define the train and val datasets
    train_ds = data_creater.create_data(train)
    val_ds = data_creater.create_data(val)

    # bs = 64
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
    # test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(gnn_features = None, encoder_type='transformer',n_genes=len(test_gene_exp.columns)).to(device)
    # lr = 1e-4
    adam = torch.optim.Adam(model.parameters(), lr = lr)
    optimizer = adam

    early_stopping = EarlyStopping(patience = n_epochs, verbose=True, chkpoint_name = ckpt_path)
    criterion = nn.MSELoss()

    # if we have a trained model, skip training and evaluate:
    model.load_state_dict(torch.load(ckpt_path))
    test_loader = DataLoader(test_ds, batch_size=999999, shuffle=False, drop_last=False) # use all for evaluation
    true_values, predicted_values = predict(model, test_loader, device)
    # print(true_values)  # Check the type of the first element if not empty
    # print(predicted_values)  # Same as above
    plot_results(true_values, predicted_values, ckpt_path, test_df)
    exit()

    # train the model    
    # Create a dictionary to store losses for training and validation
    history = {
    "train_loss": [],
    "val_loss": []}

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

            loss_all += loss.item()  # Accumulate loss
        # Calculate average training loss for this epoch
        train_loss = loss_all / len(train_loader)
        history["train_loss"].append(train_loss)  # Store training loss

        # train_rmse = gnn_utils.test_fn(train_loader, model, device)
        val_rmse, _, _, _, _ = test_fn(val_loader, model, device)
        # val_loss = val_rmse ** 2  # If using RMSE, convert to MSE (I think this is not right)
        val_loss = val_rmse 
        history["val_loss"].append(val_loss)  # Store validation loss
        early_stopping(val_rmse, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # hist["train_rmse"].append(train_rmse)
        hist["val_rmse"].append(val_rmse)
        # print(f'Epoch: {epoch}, Train_rmse: {train_rmse:.3}, Val_rmse: {val_rmse:.3}')
        print(f'Epoch: {epoch}, Val_rmse: {val_rmse:.3}')
        if epoch % 33 == 0:
            model_save_path = f'models/{output_prefix}_model_seed_{data_split_seed}_epoch_{epoch}.pt'
            torch.save(model.state_dict(), model_save_path) # save the model for each seed & epoch
            print("Model saved at", model_save_path)
    
    model.load_state_dict(torch.load(ckpt_path))
    # test_rmse, true, pred = test_fn(test_loader, model, device)
    test_rmse, pearson_corr, spearman_corr, _, _= test_fn(test_loader, model, device)
    # Extract the losses from history
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    # Create a plot
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
    parser.add_argument('--filter_drugs', type=str, default=False, help='State a drug name to filter the data')

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
    train_study_description = args.train_study_description
    test_study_description = args.test_study_description 
    dose_response_metric = args.dose_response_metric
    ckpt_path = args.checkpoint_path
    train_log_transform = args.train_log_transform
    test_log_transform = args.test_log_transform
    filter_drugs = args.filter_drugs

    # load_trained_model = False  # Assuming this is set elsewhere or needs to be added as an argparse argument

    # Call run_experiment or any other logic meant to run when the script is executed directly
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
                                    test_log_transform=test_log_transform,
                                    filter_drugs=filter_drugs)
        results[seed] = test_rmse, pearson_corr, spearman_corr 

        # File path for saving the results
        results_file_path = f'seed_{data_split_seed}_epoch_{n_epochs}_{output_prefix}_rna_train_results_table.txt'

        # Saving results to a file
        # with open(results_file_path, 'w') as file:
        #     file.write("Test RMSE for different seeds:\n")
        #     file.write("Seed\tTest RMSE\n")
        #     for seed, test_rmse in results.items():
        #         file.write(f"{seed}\t{test_rmse}\n")
            # Saving results to a file
        with open(results_file_path, 'w') as file:
            file.write("Results for different seeds:\n")
            file.write("Seed\tTest RMSE\tPearson Correlation\tSpearman Correlation\n")
            for seed, (test_rmse, pearson_corr, spearman_corr) in results.items():
                file.write(f"{seed}\t{test_rmse:.3f}\t{pearson_corr:.3f}\t{spearman_corr:.3f}\n")

        print(f"Results saved to {results_file_path}")

if __name__ == "__main__":
    main()