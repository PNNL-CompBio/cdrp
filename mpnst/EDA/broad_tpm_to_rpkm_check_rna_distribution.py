import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Required for log transformation

# Given data and gene lengths
df1 = pd.read_csv('/people/moon515/mpnst_smile_model/3_31_24/broad_sanger_transcriptomics.csv.gz', compression='gzip')
df2 = pd.read_csv('/people/moon515/mpnst_smile_model/old_input/MPNST_input/MPNST_RNA_seq.csv')
gene_lengths = pd.read_csv('/people/moon515/mpnst_smile_model/tmp/gencode_v29_gene_lengths.csv')  # CSV file with 'entrez_id' and 'gene_length'

# Define entrez_id and gene symbol
entrez_id = 4609
gene_symbol = 'MYC'

# Filter the data for a specific entrez_id
df1_filtered = df1[df1['entrez_id'] == entrez_id]
df2_filtered = df2[df2['entrez_id'] == entrez_id]

# Extract transcriptomics data for conversion
transcriptomics_1 = df1_filtered['transcriptomics']
transcriptomics_2 = df2_filtered['transcriptomics']

# Get the gene length for this specific gene
gene_length = gene_lengths[gene_lengths['gene_symbol'] == gene_symbol]['gene_length'].iloc[0]

# Calculate RPKM for transcriptomics_2
# Convert counts into RPKM
total_counts = sum(transcriptomics_2)  # Total count across samples
rpkm = 10**9 * transcriptomics_2 / gene_length / total_counts  # RPKM formula

# Apply log transformation
rpkm_log = np.log1p(rpkm)  # Log transformation with np.log1p for stability

# Plot the histogram of RPKM (log-transformed)
plt.figure(figsize=(10, 6))
plt.hist(transcriptomics_1, bins=20, alpha=0.5, label='beataml_transcriptomics')
plt.hist(rpkm_log, bins=20, alpha=0.5, label='MPNST_RNA_seq (RPKM, log-transformed)')

# Add titles and labels
plt.title(f'RPKM Distribution for Entrez ID {entrez_id} ({gene_symbol})')
plt.xlabel('Log-transformed RPKM')
plt.ylabel('Frequency')
plt.legend()

# Save the plot
output_path = f'/people/moon515/mpnst_smile_model/tmp/distribution_fig/rpkm_distribution_{entrez_id}_{gene_symbol}.png'
plt.savefig(output_path)

print(f"Plot saved at: {output_path}")
