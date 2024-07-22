import numpy as np

def calculate_statistics(file_path):
    # Read data from file
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]  # Skip header
        for line in lines:
            values = line.strip().split('\t')
            rmse = float(values[1])
            pearson = float(values[2])
            data.append((rmse, pearson))

    # Separate RMSE and Pearson Correlation
    rmse_values = [item[0] for item in data]
    pearson_values = [item[1] for item in data]

    # Calculate mean and standard deviation
    mean_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    mean_pearson = np.mean(pearson_values)
    std_pearson = np.std(pearson_values)

    print(f"Mean ± Standard Deviation of RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")
    print(f"Mean ± Standard Deviation of Pearson Correlation: {mean_pearson:.3f} ± {std_pearson:.3f}")

# File path
# file_path = 'path_to_your_file/seed_10_epoch_100_GDSCv2_early_exit_rna_train_results_table.txt'

file_path = 'seed_10_epoch_100_CCLE_MPNST_dss_rna_train_results_table.txt'

calculate_statistics(file_path)