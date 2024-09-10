#!/usr/bin/env bash
# Leave only one comment symbol on selected options
# Those with two commets will be ignored:
# The name to show in queue lists for this job:
#SBATCH -J samples_256_ddpm_unconditional_celebA.sh

# Number of desired cpus (can be in any node):
#SBATCH --ntasks=4

# Number of desired cpus (all in same node):
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4

# Amount of RAM needed for this job:
#SBATCH --mem=50gb

# The time the job will be running:
#SBATCH --time=48:00:00

# To use GPUs you have to request them:
#SBATCH --gres=gpu:2

#SBATCH --constraint=dgx
#SBATCH --nodelist=exa02

# Set output and error files
#SBATCH --error=samples_256_ddpm_unconditional_celebA.%J.err
#SBATCH --output=samples_256_ddpm_unconditional_celebA.%J.out


# To load some software (you can show the list with 'module avail'):

module load pytorch
module load singularity

export OPENAI_LOGDIR="./maskGDM/maskgdm/diffusion_model/guided-diffusion/results"
export PYTHONPATH="./maskGDM/maskgdm/diffusion_model/guided-diffusion"

# the program to execute with its parameters:
echo "Init job: Samples images celebA"

time python ./maskGDM/maskgdm/diffusion_model/guided-diffusion/scripts/image_sample_intermediateLayers.py --model_path <path_model> --save_results <path_results> --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 3 --use_fp16 False --batch_size 2 --num_samples 4 --timestep_respacing 250 --path_dataset <path_datset>