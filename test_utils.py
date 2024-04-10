# from data_utils import DataProcessor

# # Specify the input long format file path and the desired output wide format file path
# input_file_path = 'MPNST_input/MPNST_RNA_seq.csv'
# output_file_path = 'cancer_gene_expression.tsv'

# # Convert the data from long to wide format
# DataProcessor.convert_long_to_wide_format(input_file_path, output_file_path)

###############################################################################################
import pandas as pd

from data_utils import add_smiles  # Adjust the import path according to your project structure

exp = pd.read_csv('MPNST_input/MPNST_experiments.csv')
drugs = pd.read_csv('MPNST_input/MPNST_drugs.tsv', sep='\t')

output_df = add_smiles(drugs, exp, "auc")
print(output_df.head())