## Setup: Conda
conda create --name CDRP python=3.8
conda activate CDRP
pip install -r requirements.txt

<!-- ## Setup dependencies with docker
docker build -f Dockerfile -t deeptta . --build-arg HTTPS_PROXY=$HTTPS_PROXY -->

## How to Run SLURMS_script

To run the algorithm using the provided SLURM scripts (SLURM script located in SLURM_script need to be in the same directory of the python file), use the following command:

```bash
sbatch coderdata_SLURM_cross_PRISM_MPNST.sh
```

## Example SLURM Script

Here is an example of a SLURM script (`coderdata_SLURM_cross_PRISM_MPNST.sh`) to help you understand how to set up and run your jobs:

```csh
#!/bin/csh
#SBATCH -A moon515_intern                # Replace with your actual project name
#SBATCH -t 12:00:00                      # Set a shorter time limit for the test
#SBATCH -N 1                             # Number of nodes
#SBATCH -n 4                             # Number of cores (adjust as needed)
#SBATCH -J PRISM_check                   # Job name for debugging
#SBATCH -o /people/moon515/PRISM_check.out  # Standard output file for debugging
#SBATCH -e /people/moon515/PRISM_check_%j.err  # Standard error file, %j will be replaced by job ID
#SBATCH --partition=dlv                  # Partition to submit to
#SBATCH --gres=gpu:1                     # Request 1 GPU     

# Make sure the module commands are available
source /etc/profile.d/modules.csh
source /people/moon515/miniconda3/envs/CDRP  # Location of conda 

# Set up your environment you wish to run in with module commands
module purge                             # Clear all loaded modules
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
python /people/moon515/mpnst_smile_model/coderdata_auc_deep_TTA_check_self.py \
    --data_split_seed 10 \
    --n_epochs 100 \
    --train_omics_input_path /people/moon515/mpnst_smile_model/coderdata_0_1_26/broad_sanger_transcriptomics.csv.gz \
    --train_exp_input_path /people/moon515/mpnst_smile_model/coderdata_0_1_26/45458812_broad_sanger_experiments.tsv \
    --train_drugs_input_path /people/moon515/mpnst_smile_model/coderdata_0_1_26/broad_sanger_drugs.tsv \
    --output_prefix PRISM_early_exit \
    --study_description PRISM \
    --dose_response_metric fit_auc \
    --train_log_transform True \
    --test_log_transform True

# Wait for background processes to finish
wait
```

## Modify Python Script

To modify the Python script to run and include required arguments, follow these steps:

1. **Open the SLURM script**: Open the SLURM script file you want to modify (e.g., `coderdata_SLURM_cross_PRISM_MPNST.sh`).

2. **Locate the Python script call**: Find the line where the Python script is called. It should look something like this:
    ```bash
    python your_script.py
    ```

3. **Add required arguments**: Modify this line to include the necessary arguments for running your algorithm. For example:
    ```bash
    python your_script.py --arg1 value1 --arg2 value2
    ```

4. **Save the SLURM script**: After making the necessary modifications, save the SLURM script file.

Example SLURM script after modification:

```bash
#!/bin/bash
#SBATCH --job-name=your_job_name
#SBATCH --output=your_output_file.out
#SBATCH --error=your_error_file.err
#SBATCH --time=XX:XX:XX
#SBATCH --partition=your_partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=XX
#SBATCH --mem=XXG
#SBATCH --gpus=XX

# Load necessary modules
module load python/3.x
module load other_module

# Run the Python script with arguments
python your_script.py --arg1 value1 --arg2 value2
```

Replace `your_job_name`, `your_output_file.out`, `your_error_file.err`, `XX:XX:XX`, `your_partition`, `XX`, and `your_script.py --arg1 value1 --arg2 value2` with your specific job details and script arguments.

## Google Slide Presentation - James Moon Internship Project

You can view the presentation with outputs and detailed tutorial on how to use the code [here](https://docs.google.com/presentation/d/1eeSRc0E2Ran2mKjM57sjimsRYh83B3M1/edit?usp=sharing&ouid=111167841248759724862&rtpof=true&sd=true).
