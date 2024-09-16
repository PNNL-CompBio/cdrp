import pandas as pd
import numpy as np

# Read in files from local dir.

samples = pd.read_csv("mpnst_samples.csv")
experiments = pd.read_csv("mpnst_experiments.tsv",sep = "\t")

# Seperate out organoid and pdx samples. We will swap the improve_sample_id for these in the updated experiments files
organoid_samples = samples[samples['model_type'] == 'organoid']
pdx_samples = samples[samples['model_type'] == 'patient derived xenograft']

# find mappings
mapping_df = pd.merge(organoid_samples[['common_name', 'improve_sample_id']],
                      pdx_samples[['common_name', 'improve_sample_id']],
                      on='common_name', 
                      suffixes=('_organoid', '_pdx'))

# convert to dict
mapping_dict = dict(zip(mapping_df['improve_sample_id_organoid'], mapping_df['improve_sample_id_pdx']))

# Remap in experiments
experiments['improve_sample_id'] = experiments['improve_sample_id'].replace(mapping_dict)

# Write new experiments file
experiments.to_csv('mpnst_remapped_experiments.tsv', index=False,sep = "\t")

