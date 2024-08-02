import pandas as pd
import os


os.makedirs('coder_data', exist_ok=True)

def save_coder_data_in_the_multiplier_format(data_path, save_path):
        
    # genes should be in the rows
    
    # tp = pd.read_csv('../coder_data/transcriptomics.csv')
    tp = pd.read_csv(data_path)
    tp = tp.drop(['cancer_type', 'cancer_type_int'], axis=1)
    tp = tp.T
    tp.columns = tp.iloc[0, :].values.astype(int)
    tp=tp.iloc[1:,:]
    tp.index.name = 'Gene'
    tp = tp.reset_index()
    
    tp.Gene = tp.Gene.astype(float)
    tp.Gene = tp.Gene.astype(int)
    
    # tp.to_csv('coder_data/all_transcriptomics_mp.csv', index=False)
    tp.to_csv(save_path, index=False)



save_coder_data_in_the_multiplier_format('../process_data/features/transcriptomics_features.csv', 'coder_data/all_transcriptomics_mp.csv')
save_coder_data_in_the_multiplier_format('../process_data/features/proteomics_features.csv', 'coder_data/all_proteomics_mp.csv')
save_coder_data_in_the_multiplier_format('../process_data/features/copy_number_features.csv', 'coder_data/all_copy_number_mp.csv')