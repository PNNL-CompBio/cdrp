import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Importing numpy for log transformation

# The goal of this script is to compare the distribution of transcriptomics data for a specific entrez_id
entrez_id=4609
gene_symbol='MYC'

# Read the datasets
df1 = pd.read_csv('/people/moon515/mpnst_smile_model/3_31_24/beataml_transcriptomics.csv.gz', compression='gzip')
df2 = pd.read_csv('/people/moon515/mpnst_smile_model/old_input/MPNST_input/MPNST_RNA_seq.csv')

# Filter data for entrez_id
df1_filtered = df1[df1['entrez_id'] == entrez_id]
df2_filtered = df2[df2['entrez_id'] == entrez_id]

# Extract transcriptomics data for comparison
transcriptomics_1 = df1_filtered['transcriptomics']
# transcriptomics_2 = df2_filtered['transcriptomics']
transcriptomics_2_log = np.log1p(df2_filtered['transcriptomics'])  # np.log1p(x) = log(x + 1), useful if data has zeroes or negative values


# Plot histograms for both datasets
plt.figure(figsize=(10, 6))
plt.hist(transcriptomics_1, bins=20, alpha=0.5, label='beataml_transcriptomics')
# plt.hist(transcriptomics_2, bins=20, alpha=0.5, label='MPNST_RNA_seq')
plt.hist(transcriptomics_2_log, bins=20, alpha=0.5, label='MPNST_RNA_seq (log-transformed)')

# Add titles and labels
plt.title(f'Transcriptomics Distribution for Entrez ID {entrez_id} ({gene_symbol})')
plt.xlabel('Transcriptomics')
plt.ylabel('Frequency')
plt.legend()
# save as png
plt.savefig(f'/people/moon515/mpnst_smile_model/tmp/distribution_fig/transcriptomics_distribution_{entrez_id}_{gene_symbol}.png')