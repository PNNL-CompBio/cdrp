#!/bin/csh
#SBATCH -A br24_moon515                # Replace with your actual project name
#SBATCH -t 20:00:00                         # Set to your required time limit
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 4                                # Number of cores (adjust as needed)
#SBATCH -J dlv_beataml_deep_TTA                  # Job name
#SBATCH -o /people/moon515/SLURM_test_beataml.out            # Standard output file
#SBATCH -e /people/moon515/SLURM_beataml_job_errors_%j.err                # Standard error file, %j will be replaced by job ID
#SBATCH --partition=dlv                    # Partition to submit to
#SBATCH --gres=gpu:1                        # Request 1 GPU     
# Make sure the module commands are available
source /etc/profile.d/modules.csh
source /people/moon515/miniconda3/envs/CDRP # location of conda 

# Set up your environment you wish to run in with module commands
module purge                               # Clear all loaded modules
conda activate CDRP

# Add any other modules or environment settings you need

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

# Execute your command with specified arguments
python /people/moon515/mpnst_smile_model/deep_TTA_RNA_run.py \
    --data_split_seed 10 \
    --n_epochs 100 \
    --train_omics_input_path /people/moon515/mpnst_smile_model/coderdata_input/beataml_transcriptomics.csv.gz \
    --train_exp_input_path /people/moon515/mpnst_smile_model/coderdata_input/beataml_experiments.csv.gz \
    --train_drugs_input_path /people/moon515/mpnst_smile_model/coderdata_input/beataml_drugs.tsv.gz \
    --test_omics_input_path /people/moon515/mpnst_smile_model/coderdata_input/MPNST_RNA_seq.csv.gz \
    --test_exp_input_path /people/moon515/mpnst_smile_model/coderdata_input/MPNST_experiments.csv.gz \
    --test_drugs_input_path /people/moon515/mpnst_smile_model/coderdata_input/MPNST_drugs.tsv.gz \
    --study_description beataml_test
# Wait for background processes to finish
wait
