# script that will run the deep learning model with TTA
from data_utils import DataProcessor, add_smiles, average_auc
from gnn_utils import CreateData, EarlyStopping, test_fn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from model import Model
import pandas as pd 
import os

def split_df(df, seed):
    train, val = train_test_split(df, random_state=seed, test_size=0.2, train_size=0.8)

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    return train, val

# all the parameters are set here
data_split_seed = 1 # adjust seed here
bs = 64
lr = 1e-4
n_epochs = 10
mpnst_input_path = "./coderdata_input/MPNST_RNA_seq.csv.gz"
beataml_input_path = "./coderdata_input/beataml_transcriptomics.csv.gz"
# if you run the model for the first time, set load_trained_model to False
load_trained_model = True

##############################
# DATA PROCESSING
##############################
# Process gene expression data
# Downloads and saves the gene expression data into a separate directory "process_input"
if not os.path.exists("shared_input/MPNST_RNA_seq_wide.tsv"):
    DataProcessor.convert_long_to_wide_format(mpnst_input_path)
mpnst_gene_exp = pd.read_csv("shared_input/MPNST_RNA_seq_wide.tsv", sep='\t')

if not os.path.exists("shared_input/beataml_transcriptomics.csv.gz_wide.tsv"):
    DataProcessor.convert_long_to_wide_format(beataml_input_path)
beataml_gene_exp = pd.read_csv("shared_input/beataml_transcriptomics_wide.tsv", sep='\t')

#
def intersect_columns(mpnst_gene_exp, beataml_gene_exp):
    common_columns = list(set(mpnst_gene_exp.columns).intersection(set(beataml_gene_exp.columns)))
    mpnst_gene_exp = mpnst_gene_exp[common_columns]
    beataml_gene_exp = beataml_gene_exp[common_columns]
    return mpnst_gene_exp, beataml_gene_exp

mpnst_gene_exp, beataml_gene_exp = intersect_columns(mpnst_gene_exp, beataml_gene_exp)

# mpnst: process experiment & drug data
mpnst_exp = pd.read_csv("./coderdata_input/MPNST_experiments.csv.gz", compression='gzip')
mpnst_drugs = pd.read_csv("./coderdata_input/MPNST_drugs.tsv.gz", sep='\t', compression='gzip')
# average auc values for the same improve_sample_id and drug_id
mpnst_exp = average_auc(mpnst_exp)
# add smiles and split data
mpnst_df_all = add_smiles(mpnst_drugs, mpnst_exp, "auc")

# beataml: process experiment & drug data
beataml_exp = pd.read_csv("./coderdata_input/beataml_experiments.csv.gz", compression='gzip')
beataml_drugs = pd.read_csv("./coderdata_input/beataml_drugs.tsv.gz", sep='\t', compression='gzip')
# average auc values for the same improve_sample_id and drug_id
beataml_exp = average_auc(beataml_exp)
# add smiles and split data
beataml_df_all = add_smiles(beataml_drugs, beataml_exp, "auc")

# Find the intersection of improve_sample_id in RNA & drug info
mpnst_common_ids = set(mpnst_df_all['improve_sample_id'].unique()).intersection(set(mpnst_gene_exp['improve_sample_id'].unique()))
# Filter RNA & drug to only include rows with improve_sample_id in the intersection
mpnst_df = mpnst_df_all[mpnst_df_all['improve_sample_id'].isin(mpnst_common_ids)].reset_index(drop=True)
mpnst_gene_exp = mpnst_gene_exp[mpnst_gene_exp['improve_sample_id'].isin(mpnst_common_ids)]
test = mpnst_df.reset_index(drop=True, inplace=True)

# Find the intersection of improve_sample_id in RNA & drug
beataml_common_ids = set(beataml_df_all['improve_sample_id'].unique()).intersection(set(beataml_gene_exp['improve_sample_id'].unique()))
# Filter RNA & drug to only include rows with improve_sample_id in the intersection
beataml_df = beataml_df_all[beataml_df_all['improve_sample_id'].isin(beataml_common_ids)].reset_index(drop=True)
beataml_gene_exp = beataml_gene_exp[beataml_gene_exp['improve_sample_id'].isin(beataml_common_ids)]
train, val= split_df(df=beataml_df, seed=data_split_seed)

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
beataml_gene_exp = beataml_gene_exp.set_index('improve_sample_id')
# Now perform the scaling operation on the DataFrame without the index column
scaler = StandardScaler() # mean=0, unit variance
beataml_gene_exp_scaled = scaler.fit_transform(beataml_gene_exp)
# When creating the new DataFrame, use the same columns as the gene_exp DataFrame
# Because gene_exp now does not include 'improve_sample_id' column, we don't need to adjust column names
beataml_gene_exp_scaled = pd.DataFrame(beataml_gene_exp_scaled, index=beataml_gene_exp.index, columns=beataml_gene_exp.columns)
data_creater = CreateData(gexp=beataml_gene_exp_scaled, encoder_type='transformer', metric="auc", data_path= "shared_input/")
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
test['true'] = true
test['pred'] = pred

print(test) # print the test dataframe with true and pred columns

# if args.feature_path:
#     test = test[['improve_sample_id', 'smiles', 'improve_chem_id', 'auc', 'true', 'pred']]

# test.to_csv( os.path.join(out_dir, 'test_predictions.csv'), index=False )