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
from data_utils import load_generic_expression_data

def split_df(df, seed):
    
    train, test = train_test_split(df, random_state=seed, test_size=0.2)
    val, test = train_test_split(test, random_state=seed, test_size=0.5)
    
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    return train, val, test
    
if __name__ == "__main__":


    parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does')
    parser.add_argument('--metric',  default='auc', help='')
    parser.add_argument('--run_id',  default='0', help='')
    parser.add_argument('--epochs',  default=1, help='')
    parser.add_argument('--batch_size', type=int,  default=64, help='')
    parser.add_argument('--data_split_seed',  default=1, help='')
    parser.add_argument('--data_split_id',  default=0, help='')
    parser.add_argument('--encoder_type',  default='gnn', help='')
    parser.add_argument('--data_path',  default='', help='')
    parser.add_argument('--data_type',  default='CCLE', help='')
    parser.add_argument('--data_version',  default='benchmark-data-pilot1', help='benchmark-data-imp-2023 or benchmark-data-pilot1')
    parser.add_argument('--out_dir',  default='', help='')
    parser.add_argument('--feature_path', type=str, default=None, help='')
    parser.add_argument('--scale_gexp',  type=bool, default=False, help='')
    parser.add_argument('--use_proteomics_data',  type=int, default=1, help='if 1 use proteomics data elif 0, use gexp')

    args = parser.parse_args()



    pc = DataProcessor(args.data_version)

    metric = args.metric
    bs = args.batch_size
    lr = 1e-4
    n_epochs = int(args.epochs)
    out_dir = args.out_dir
    run_id = args.run_id
    data_split_seed = int(args.data_split_seed)
    encoder_type = args.encoder_type
    data_type = args.data_type
    data_split_id = args.data_split_id

    out_dir = os.path.join(out_dir, run_id )
    os.makedirs( out_dir, exist_ok=True )
    os.makedirs( args.data_path, exist_ok=True )
    ckpt_path = os.path.join(out_dir, 'best.pt')

    dw = Downloader(args.data_version)
    dw.download_candle_data(data_type=data_type, split_id=data_split_id, data_dest=args.data_path)
    dw.download_deepttc_vocabs(data_dest=args.data_path)



    train = pc.load_drug_response_data(data_path=args.data_path, data_type=data_type,
        split_id=data_split_id, split_type='train', response_type=metric, sep="\t",
        dropna=True)

    val = pc.load_drug_response_data(data_path=args.data_path, data_type=data_type,
        split_id=data_split_id, split_type='val', response_type=metric, sep="\t",
        dropna=True)

    test = pc.load_drug_response_data(data_path=args.data_path, data_type=data_type,
        split_id=data_split_id, split_type='test', response_type=metric, sep="\t",
        dropna=True)

    smiles_df = pc.load_smiles_data(args.data_path)

    train = add_smiles(smiles_df=smiles_df, df=train, metric=metric)
    val = add_smiles(smiles_df=smiles_df, df=val, metric=metric)
    test = add_smiles(smiles_df=smiles_df, df=test, metric=metric)

    gene_exp_ = load_generic_expression_data('proteomics_restructure_with_knn_impute.tsv')


    if args.use_proteomics_data == 1:
        print('using proteomics data')
        gene_exp = gene_exp_
    else:
        print('using gene expression data')
        gene_exp  = pc.load_gene_expression_data(args.data_path)

    
    use_improve_ids = gene_exp_.index.values
    train = train[train.improve_sample_id.isin(use_improve_ids)]
    val = val[val.improve_sample_id.isin(use_improve_ids)]
    test = test[test.improve_sample_id.isin(use_improve_ids)]

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    

    
    df_all = pd.concat([train, val, test], axis=0)
    df_all.reset_index(drop=True, inplace=True)

    if data_split_seed > -1:
        print("using random splitting")
        train, val, test = split_df(df=df_all, seed=data_split_seed)
    else:
        print("using predefined splits")


    # gene_exp  = pc.load_gene_expression_data(args.data_path)
   

    lm = pc.load_landmark_genes(args.data_path)
    lm = list(set(lm).intersection(gene_exp.columns))
    gexp = gene_exp.loc[:, lm]

    n_genes = len(lm)

    if args.scale_gexp:
        scgexp = StandardScaler()
        gexp.loc[:,:] = scgexp.fit_transform(gexp)


    n_descriptors=None
    features=None
    if args.feature_path:
        print("feature path exists")
        features = pd.read_csv(args.feature_path)
        n_descriptors = features.shape[1] - 1
        feature_names = features.drop(['smiles'], axis=1).columns.tolist()
    
        test = pd.merge(test, features, on='smiles', how='left')
        train = pd.merge(train, features, on='smiles', how='left')
        val = pd.merge(val, features, on='smiles', how='left')

        sc = StandardScaler()
        train.loc[:, feature_names] = sc.fit_transform(train.loc[:, feature_names])
        test.loc[:, feature_names] = sc.transform(test.loc[:, feature_names])
        val.loc[:, feature_names] = sc.transform(val.loc[:, feature_names])
    else:
        feature_names=None


    data_creater = CreateData(gexp=gexp, metric=metric, encoder_type=encoder_type, data_path=args.data_path, feature_names=feature_names)
    
    train_ds = data_creater.create_data(train)
    val_ds = data_creater.create_data(val)
    test_ds = data_creater.create_data(test)
    
    # train_ds = create_data_list(train, gexp, metric=metric)
    # val_ds = create_data_list(val, gexp, metric=metric)
    # test_ds = create_data_list(test, gexp, metric=metric)


    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False)

    # train_loader_no_shuffle = DataLoader(train_X, batch_size = bs, shuffle=False, drop_last=False)
    # val_loader_no_shuffle = DataLoader(val_X, batch_size = bs, shuffle=False, drop_last=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(gnn_features = 65, n_descriptors=n_descriptors, encoder_type=encoder_type,
                 n_genes=n_genes).to(device)
    adam = torch.optim.Adam(model.parameters(), lr = lr )
    optimizer = adam

    
    early_stopping = EarlyStopping(patience = n_epochs, verbose=True, chkpoint_name = ckpt_path)
    criterion = nn.MSELoss()


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

    # print(f"training completed at {datetime.datetime.now()}")

    model.load_state_dict(torch.load(ckpt_path))

    test_rmse, true, pred = test_fn(test_loader, model, device)
    test['true'] = true
    test['pred'] = pred

    if args.feature_path:
        test = test[['improve_sample_id', 'smiles', 'improve_chem_id', 'auc', 'true', 'pred']]

    test.to_csv( os.path.join(out_dir, 'test_predictions.csv'), index=False )
    # r2_score(y_pred=pred, y_true=true)
