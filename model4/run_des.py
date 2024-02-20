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


    os.environ["CUDA_VISIBLE_DEVICES"] = str(3)

    epochs = args.epochs
    data_type = args.data_type
    metric = args.metric
    out_dir = args.out_dir

    # for i, seed in enumerate(seeds):
    for i in range(10):
    

        os.system(f"python train.py --encoder_type descriptor --out_dir Descriptorenc/{out_dir} --data_version benchmark-data-pilot1  \
        --data_path Descriptorenc/{out_dir}/Data --data_split_seed -10 --data_split_id {i} --metric {metric}  --data_type {data_type} --epochs {epochs} --run_id {i} \
        --feature_path ../drug_features_pilot1.csv")
        # break 
