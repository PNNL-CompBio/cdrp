import os
import pickle
import argparse
from omegaconf import OmegaConf



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='config.yaml')
    args = parser.parse_args()

    if args.config:  # args priority is higher than yaml
        args_ = OmegaConf.load(args.config)
        OmegaConf.resolve(args_)
        args=args_


    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)

    epochs = args.epochs
    data_type = args.data_type
    metric = args.metric
    out_dir = args.out_dir
    use_proteomics_data = args.use_proteomics_data

 

    # for i, seed in enumerate(seeds):
    for i in range(10):
    

        #os.system(f"python train_prot.py --encoder_type gnn --out_dir GNNenc/{out_dir} --data_version benchmark-data-pilot1  \
        #--data_path GNNenc/{out_dir}/Data --data_split_seed -10 --data_split_id {i} --metric {metric} --data_type {data_type} --epochs {epochs} --run_id {i} \
        # --use_proteomics_data {use_proteomics_data}")
  
        os.system(f"python train_prot.py --encoder_type transformer --out_dir Trnsfenc/{out_dir} --data_version benchmark-data-pilot1 \
        --data_path Trnsfenc/{out_dir}/Data --data_split_seed -10 --data_split_id {i} --metric {metric} --data_type {data_type} --epochs {epochs} --run_id {i} \
        --use_proteomics_data {use_proteomics_data}")

        #os.system(f"python train_prot.py --encoder_type morganfp --out_dir Morganfpenc/{out_dir} --data_version benchmark-data-pilot1  \
        #--data_path Morganfpenc/{out_dir}/Data --data_split_seed -10 --data_split_id {i} --metric {metric} --data_type {data_type} --epochs {epochs} --run_id {i} \
        #--use_proteomics_data {use_proteomics_data}")

        #os.system(f"python train_prot.py --encoder_type descriptor --out_dir Descriptorenc/{out_dir} --data_version benchmark-data-pilot1  \
        #--data_path Descriptorenc/{out_dir}/Data --data_split_seed -10 --data_split_id {i} --metric {metric}  --data_type {data_type} --epochs {epochs} --run_id {i} \
        #--feature_path ../drug_features_pilot1.csv \
        #--use_proteomics_data {use_proteomics_data}")
        break 
