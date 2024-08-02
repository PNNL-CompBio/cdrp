import pandas as pd
import numpy as np

# Read gene length data
gene_lengths = pd.read_csv('/people/moon515/mpnst_smile_model/tmp/gencode_v29_gene_lengths.csv')

# Read mapping of Entrez IDs to gene symbols
entrez_to_symbol = pd.read_csv('/people/moon515/mpnst_smile_model/tmp/genes.csv')

# Clean and prepare the Entrez to gene symbol mapping
entrez_to_symbol = entrez_to_symbol[['entrez_id', 'gene_symbol']]
entrez_to_symbol.drop_duplicates(inplace=True)

# Add entrez_id column to gene_lengths based on the entrez_to_symbol mapping
gene_lengths = pd.merge(gene_lengths, entrez_to_symbol, how='left', on='gene_symbol')

# remove rows with NaN entrez_id
gene_lengths = gene_lengths[gene_lengths['entrez_id'].notnull()]

# Read the TPM count data
tpm = pd.read_csv('/people/moon515/mpnst_smile_model/old_input/MPNST_input/MPNST_RNA_seq.csv')

# Merge TPM data with gene length based on entrez_id
tpm = pd.merge(tpm, gene_lengths[['entrez_id', 'gene_length']], how='left', on='entrez_id')

# remove rows with NaN gene_length
tpm = tpm[tpm['gene_length'].notnull()]

# Calculate RPKM from TPM
# RPKM = (transcriptomics * 10^6) / (gene_length_in_kb * total_RNA_seq_depth_in_millions)
tpm['gene_length_kb'] = tpm['gene_length'] / 1000  # Convert gene length to kilobases
total_RNA_seq_depth = 10**6  # Placeholder for total depth in millions
tpm['rpkm'] = (tpm['transcriptomics'] * total_RNA_seq_depth) / tpm['gene_length_kb']

# Convert RPKM to log-transformed RPKM
tpm['log_rpkm'] = np.log2(tpm['rpkm'] + 1)  # Adding 1 to avoid log(0)

# Maintain original 5 columns plus the new log(RPKM) column
output_data = tpm[['improve_sample_id', 'log_rpkm', 'entrez_id', 'source', 'study']]
output_data.rename(columns={'log_rpkm': 'transcriptomics'}, inplace=True)

# replace 'transcriptomics' value with 'log_rpkm' then remove log_rpkm column

# Write the output to a CSV file
output_data.to_csv('/people/moon515/mpnst_smile_model/old_input/MPNST_input/MPNST_RNA_seq_log2_rpkm.csv', index=False)

# Display the first few rows to confirm
print(output_data.head())
