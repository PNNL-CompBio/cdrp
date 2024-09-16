#!/bin/csh
#SBATCH -A coderdata_mpnst                # Replace with your actual project name
#SBATCH -t 12:00:00                    # Set a shorter time limit for the test
#SBATCH -N 1                          # Number of nodes
#SBATCH -n 4                          # Number of cores (adjust as needed)
#SBATCH -J deeptta_BeatAML_CCLE       # Job name for debugging
#SBATCH -o /people/jaco059/CoderData_Moon_Project/logs/BeatAML_CCLE_cross_%j.out  # Standard output file for debugging
#SBATCH -e /people/jaco059/CoderData_Moon_Project/logs/BeatAML_CCLE_cross_%j.err  # Standard error file, %j will be replaced by job ID
#SBATCH --partition=dlv               # Partition to submit to
#SBATCH --gres=gpu:1                  # Request 1 GPU     
# Make sure the module commands are available
source /etc/profile.d/modules.csh
source /people/jaco059/.conda/envs/CDRP # location of conda environment

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

# Execute your command with specified arguments for a short debugging session
python /people/jaco059/CoderData_Moon_Project/cdrp/mpnst/coderdata_auc_deep_TTA_cross_test.py \
    --data_split_seed 10 \
    --n_epochs 100 \
    --train_omics_input_path /people/jaco059/CoderData_Moon_Project/data_coderdata_0_1_40/beataml_transcriptomics.csv \
    --train_exp_input_path /people/jaco059/CoderData_Moon_Project/data_coderdata_0_1_40/beataml_experiments.tsv \
    --train_drugs_input_path /people/jaco059/CoderData_Moon_Project/data_coderdata_0_1_40/beataml_drugs.tsv \
    --test_omics_input_path /people/jaco059/CoderData_Moon_Project/data_coderdata_0_1_40/broad_sanger_transcriptomics.csv \
    --test_exp_input_path /people/jaco059/CoderData_Moon_Project/data_coderdata_0_1_40/broad_sanger_experiments.tsv \
    --test_drugs_input_path /people/jaco059/CoderData_Moon_Project/data_coderdata_0_1_40/broad_sanger_drugs.tsv \
    --output_prefix BeatAML_CCLE_DSS \
    --train_study_description BeatAML \
    --test_study_description CCLE \
    --dose_response_metric dss \
    --train_log_transform True \
    --test_log_transform True \
    --checkpoint_path /people/jaco059/CoderData_Moon_Project/cdrp/mpnst/tmp/beataml_ccle_best.pt
# Wait for background processes to finish
wait
