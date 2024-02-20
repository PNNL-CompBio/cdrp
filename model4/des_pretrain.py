from data_utils import Downloader, DataProcessor
from data_utils import add_smiles
from gnn_utils import create_data_list
from torch_geometric.loader import DataLoader
from model import Model
import torch
import torch.nn as nn
from gnn_utils import EarlyStopping
# dw = Downloader('benchmark-data-imp-2023')
from gnn_utils import test_fn
import os
import argparse
import pandas as pd
from gnn_utils import CreateData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn.functional as F
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

def split_df(df, seed):
    
    train, test = train_test_split(df, random_state=seed, test_size=0.2)
    val, test = train_test_split(test, random_state=seed, test_size=0.5)
    
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    return train, val, test



# des = pd.read_csv('dataset.csv')
# des
# calc = Calculator(descriptors, ignore_3D=True)
# mols = [Chem.MolFromSmiles(i) for i in des.SMILES]

df_mdm = pd.read_csv('mdm_somas.csv')
df_mdm = df_mdm.iloc[:, :-2]

rem=[]
for i in df_mdm.columns:
    try:
        df_mdm[i].astype(float)
    except:
        rem.append(i)

feature_names = list(set(df_mdm.columns).difference(rem))
features = pd.read_csv('../drug_features_pilot1.csv')
feature_names = list(set(feature_names).intersection(features.columns[1:]))


df_mdm = df_mdm.loc[:, feature_names]
# 'GATS1Z' in df_mdm.columns
feature_names = list(set(df_mdm.columns).difference(['GATS1Z']))
sc = StandardScaler()
np.where(np.isnan(sc.fit_transform(df_mdm)))
# calc = Calculator(descriptors.Autocorrelation, ignore_3D=True)
# mols = [Chem.MolFromSmiles(i) for i in des.SMILES]

# df_des = calc.pandas(mols)
# df_des['smiles'] = des['SMILES']
# df_des['mfrags'] = [len(Chem.GetMolFrags(i)) for i in mols]
# df_des.to_csv('des_autoc.csv', index=False)
# df_des = pd.read_csv('des_autoc.csv')
# df_des.shape



with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

sc = StandardScaler()
df_mdm.loc[:, feature_names] = sc.fit_transform(df_mdm.loc[:, feature_names])
df_mdm.shape



train, test = train_test_split(df_mdm, test_size=.2)
val, test = train_test_split(test, test_size=.5)

target_name = 'GATS1Z'
'GATS1Z' in feature_names
def create_descriptor_data(df):

    df.reset_index(drop=True, inplace=True)
    # data = df.copy()
    # data.set_index('smiles', inplace=True)
    
    data_list = []
    for i in tqdm(range(df.shape[0])):
        # smiles = data.loc[i, 'smiles']
        y  = df.loc[i, target_name]
        # improve_sample_id = data.loc[i, 'improve_sample_id']
        # feature_list = self.features.loc[smiles, :].values.tolist()
        feature_list = df.loc[i, feature_names].values.tolist()
        # ge = self.gexp.loc[improve_sample_id, :].values.tolist()
    
        data = Data(fp=torch.tensor([feature_list], dtype=torch.float),
                y=torch.tensor([y],dtype=torch.float),)
        data_list.append(data)
        
    return data_list



# 'GATS1Z' in data.columns
bs = 64
lr = 1e-4
n_epochs = 500
train_ds = create_descriptor_data(train)
val_ds = create_descriptor_data(val)
test_ds = create_descriptor_data(test)
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False)
n_descriptors = len(feature_names)
from torch.nn import Linear
class DescriptorEncoder(nn.Module):
    def __init__(self, n_descriptors):
        super(DescriptorEncoder, self).__init__()

        self.fc1 = Linear( n_descriptors, 1024)
        self.fc2 = Linear( 1024, 512)
        self.fc3 = Linear( 512, 256)
        self.do1 = nn.Dropout(p = 0.1)
        self.do2 = nn.Dropout(p = 0.1)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.out = Linear(256, 1)

    def forward(self, data):
        fp = data.fp

        e = self.do1(self.act1(self.fc1(fp)))
        e = self.do2(self.act2(self.fc2(e)))
        e = self.fc3(e)
        out = self.out(e)

        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DescriptorEncoder(n_descriptors)
model.to(device);

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
adam = torch.optim.Adam(model.parameters(), lr = lr )
optimizer = adam

ckpt_path = 'pretrain.pt'
early_stopping = EarlyStopping(patience = 20, verbose=True, chkpoint_name = ckpt_path)
criterion = nn.MSELoss()


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

# print(f"training completed at {datetime.datetime.now()}")

#model.load_state_dict(torch.load(ckpt_path))


