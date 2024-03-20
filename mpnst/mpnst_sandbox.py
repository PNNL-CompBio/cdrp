# Author: Chang In Moon
# Test transformer model with MPNST dataset

# Not using graph neural network for my experiment
from gnn_utils import CreateData
# from gnn_utils import create_data_list
# from gnn_utils import EarlyStopping
# from gnn_utils import test_fn

# Refer to these for data processing
# from data_utils import DataProcessor
# from data_utils import add_smiles
from transformer import TransformerModel
from torch_geometric.loader import DataLoader
from model import Model
import torch
import torch.nn as nn
from torch.nn import Linear
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#-----------------------------------
# 1. Load and process data
#-----------------------------------
# Load drug & experiment files
drug = pd.read_csv('MPNST_input/MPNST_drugs.tsv', delimiter='\t')
experiment = pd.read_csv('MPNST_input/MPNST_experiments.csv')

# Load data gene expression data
gene_exp = pd.read_csv('MPNST_input/MPNST_RNA_seq.csv')

# figure out how to split the data starting with CreateData...
# gnn_utils::SmilesTokenizer requires vocab_path for BPE and subword_units files # ask Gihan about these files # how to get these files?
data_creater = CreateData(gexp=gexp, metric=metric, encoder_type='transformer', data_path=args.data_path, feature_names=feature_names)
train_ds = data_creater.create_data(train)
val_ds = data_creater.create_data(val)
test_ds = data_creater.create_data(test)

# Used DataLoader to load the data [train_ds, val_ds, test_ds are expected to be a pytorch dataset]
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False)



#-----------------------------------
# 2. Apply transformers
#-----------------------------------
# Hyperparameters
params = {'a1': 0, 'a2': 2, 'a3': 1, 'a4': 2, 'bs': 1, 'd1': 0.015105134306121593, 'd2': 0.3431295462686682, \
      'd3': 0.602688496976768, 'd4': 0.9532038077650021, 'e1': 256.0, 'eact1': 0, 'edo1': 0.4813038851902818,\
      'f1': 256.0, 'f2': 256.0, 'f3': 160.0, 'f4': 24.0, 'g1': 256.0, 'g2': 320.0, 'g21': 448.0,\
      'g22': 512.0, 'gact1': 2, 'gact2': 2, 'gact21': 2, 'gact22': 0, 'gact31': 2, 'gact32': 1, 'gact33': 1,\
      'gdo1': 0.9444250299450242, 'gdo2': 0.8341272742321129, 'gdo21': 0.7675340644596443,\
      'gdo22': 0.21498171859119775, 'gdo31': 0.8236003195596049, 'gdo32': 0.6040220843354102,\
      'gdo33': 0.21007469160431758, 'lr': 0, 'nfc': 0, 'ngl': 1, 'opt': 0}
act = {0: torch.nn.ReLU(), 1:torch.nn.SELU(), 2:torch.nn.Sigmoid()}

# Define Model
class Model(torch.nn.Module):
    
    def __init__(self, encoder_type=None, n_genes=958):
        super(Model, self).__init__()
        
        self.encoder_type= encoder_type
        args = {'vocab_size':2586,
                'masked_token_train': False,
                'finetune': False}
        self.drug_encoder = TransformerModel(args)
        
        # self.out2 = Linear(int(params['f2']), 1)
        # self.out3 = Linear(int(params['f3']), 1)
        # self.out4 = Linear(int(params['f4']), 1)
        
        self.dropout1 = nn.Dropout(p = params['d1'] )
        self.act1 = act[torch.nn.ReLU()]

        self.dropout2 = nn.Dropout(p = params['d2'] )
        self.act2 = act[torch.nn.Sigmoid()]

        self.transformer_lin = Linear(512,256)
        
        self.gexp_lin1 = Linear(n_genes, n_genes)
        self.gexp_lin2 = Linear(n_genes, 256)
        
        self.cat1 = Linear(512, 256)
        self.cat2 = Linear(256, 128)
        self.out = Linear(128, 1)
        
    def forward(self, data):
        # node_x, edge_x, edge_index = data.x, data.edge_attr, data.edge_index
        # if self.encoder_type in ['gnn', 'morganfp', 'descriptor']:
        #     drug = self.drug_encoder(data)
        _,_, drug = self.drug_encoder(data)
        drug = self.transformer_lin(drug)
        
        gexp = data.ge
        gexp = self.gexp_lin1(gexp)
        gexp = self.gexp_lin2(gexp)
        
        drug_gene = torch.cat((drug, gexp), 1)
        
        x3 = self.dropout1(self.act1(self.cat1( drug_gene )))
        x3 = self.dropout2(self.act2(self.cat2( x3 )))
        x3 = self.out(x3)
        return x3
