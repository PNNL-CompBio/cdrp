import pandas as pd

# Load the GTF file
gtf_file = "gencode.v29.basic.annotation.gtf"
gtf_df = pd.read_csv(gtf_file, comment='#', delimiter='\t', header=None)

# Define column names for GTF data
gtf_df.columns = ["chr", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]

# Filter for gene features
gene_df = gtf_df[gtf_df['type'] == 'gene']

# Function to parse attributes and return a dictionary of attribute keys and values
def parse_attributes(attribute_string):
    attributes = {}
    attr_list = attribute_string.split(";")
    for attr in attr_list:
        attr = attr.strip()
        if attr:
            key, value = attr.split(" ", 1)
            attributes[key] = value.replace('"', '')  # Remove quotes from values
    return attributes

# Apply parsing function to get a new column with parsed attributes
gene_df['parsed_attributes'] = gene_df['attributes'].apply(parse_attributes)

# Extract gene symbols and calculate gene lengths
gene_df['gene_symbol'] = gene_df['parsed_attributes'].apply(lambda x: x.get('gene_name', ''))
gene_df['gene_id'] = gene_df['parsed_attributes'].apply(lambda x: x.get('gene_id', ''))
gene_df['gene_length'] = gene_df['end'] - gene_df['start']

# Create a DataFrame with gene symbols and lengths
gene_lengths = gene_df[['gene_symbol', 'gene_id', 'gene_length']]

# save the gene lengths to a CSV file
gene_lengths.to_csv("gencode_v29_gene_lengths.csv", index=False)