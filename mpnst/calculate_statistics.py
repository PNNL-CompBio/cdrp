import os
import argparse
import numpy as np

def calculate_statistics(file_path):
    # Read data from file
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]  # Skip header
        for line in lines:
            values = line.strip().split('\t')
            try:
                rmse = float(values[1])
                pearson = float(values[2])
                data.append((rmse, pearson))
            except ValueError:
                # Skip lines with non-numeric data
                continue

    # Ensure there is valid data to process
    num_seeds = len(data)
    if num_seeds == 0:
        print(f"No valid data found in file: {file_path}")
        return

    # Separate RMSE and Pearson Correlation
    rmse_values = [item[0] for item in data]
    pearson_values = [item[1] for item in data]

    # Calculate mean and standard deviation
    mean_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    mean_pearson = np.mean(pearson_values)
    std_pearson = np.std(pearson_values)

    # Print results
    print(f"Results for file: {file_path}")
    print(f"Number of seeds used: {num_seeds}")
    print(f"Mean ± Standard Deviation of RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")
    print(f"Mean ± Standard Deviation of Pearson Correlation: {mean_pearson:.3f} ± {std_pearson:.3f}")
    print("-" * 40)

def process_all_files_in_folder(results_folder):
    # Loop through all files in the folder
    for filename in os.listdir(results_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(results_folder, filename)
            calculate_statistics(file_path)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process result files in a folder to calculate statistics.")
    parser.add_argument('results_folder', type=str, help="Path to the folder containing the result files.")

    # Parse the arguments
    args = parser.parse_args()

    # Process all files in the given folder
    process_all_files_in_folder(args.results_folder)

if __name__ == "__main__":
    main()
