#!/bin/csh
#SBATCH -A br24_moon515                # Replace with your actual project name
#SBATCH -t 4:00:00                    # Set a shorter time limit for the test
#SBATCH -N 1                          # Number of nodes
#SBATCH -n 4                          # Number of cores (adjust as needed)
#SBATCH -J ccle_check     # Job name for debugging
#SBATCH -o /people/moon515/ccle_check.out  # Standard output file for debugging
#SBATCH -e /people/moon515/ccle_check_%j.err  # Standard error file, %j will be replaced by job ID
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

# Execute your command with specified arguments for a short debugging session
python /people/moon515/mpnst_smile_model/coderdata_evaluate_epochs_check_ccle.py \
    --data_split_seed 1 \
    --n_epochs 100 \
    --train_omics_input_path /people/moon515/mpnst_smile_model/coderdata_0_1_26/broad_sanger_transcriptomics.csv.gz \
    --train_exp_input_path /people/moon515/mpnst_smile_model/coderdata_0_1_26/45458812_broad_sanger_experiments.tsv \
    --train_drugs_input_path /people/moon515/mpnst_smile_model/coderdata_0_1_26/broad_sanger_drugs.tsv \
    --output_prefix ccle_check \
    --study_description CCLE \
    --dose_response_metric fit_auc \
    --train_log_transform True \
    --test_log_transform True \
    --checkpoint_path /people/moon515/mpnst_smile_model/models/ccle_check_model_seed_1_epoch_0.pt \
    
# Wait for background processes to finish
wait