#!/bin/csh
#SBATCH -A moon515_intern                # Replace with your actual project name
#SBATCH -t 12:00:00                    # Set a shorter time limit for the test
#SBATCH -N 1                          # Number of nodes
#SBATCH -n 4                          # Number of cores (adjust as needed)
#SBATCH -J deeptta     # Job name for debugging
#SBATCH -o /people/moon515/CCLE_BeatAML_check.out  # Standard output file for debugging
#SBATCH -e /people/moon515/CCLE_BeatAML_check_%j.err  # Standard error file, %j will be replaced by job ID
#SBATCH --partition=dlv               # Partition to submit to
#SBATCH --gres=gpu:1                  # Request 1 GPU     
# Make sure the module commands are available
source /etc/profile.d/modules.csh
source /people/moon515/miniconda3/envs/CDRP # location of conda 

# Set up your environment you wish to run in with module commands
module purge                          # Clear all loaded modules
conda activate CDRP

# Unlimit system resources (for csh/tcsh)
unlimit

# It is extremely useful to record the modules you have loaded, your limit settings, 
# your current environment variables and the dynamically loaded libraries 
# that your executable is linked against in your job output file.
echo "loaded modules"
module list >& _modules.lis_
cat _modules.lis_
/bin/rm -f _modules.lis_

echo "limits"
limit

echo "Environment Variables"
printenv

# Execute your command with specified arguments for a short debugging session (modify for batch size for aml)
python /people/moon515/mpnst_smile_model/coderdata_proteomics_deep_TTA_cross_test.py \
    --data_split_seed 10 \
    --n_epochs 100 \
    --bs 64 \
    --train_omics_input_path /people/moon515/mpnst_smile_model/2024_05_17/broad_sanger_proteomics.csv.gz \
    --train_exp_input_path /people/moon515/mpnst_smile_model/2024_05_17/broad_sanger_experiments.tsv \
    --train_drugs_input_path /people/moon515/mpnst_smile_model/2024_05_17/broad_sanger_drugs.tsv \
    --test_omics_input_path /people/moon515/mpnst_smile_model/2024_05_17/beataml_proteomics.csv.gz \
    --test_exp_input_path /people/moon515/mpnst_smile_model/2024_05_17/beataml_experiments.tsv \
    --test_drugs_input_path /people/moon515/mpnst_smile_model/2024_05_17/beataml_drugs.tsv \
    --output_prefix CCLE_BeatAML_proteomics_dss \
    --train_study_description CCLE \
    --test_study_description BeatAML \
    --dose_response_metric dss \
    --train_log_transform False \
    --test_log_transform False \
    --checkpoint_path /people/moon515/mpnst_smile_model/tmp/best.pt
# Wait for background processes to finish
wait
