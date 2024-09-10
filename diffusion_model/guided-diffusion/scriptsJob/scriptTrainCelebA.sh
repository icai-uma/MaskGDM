#!/usr/bin/env bash
# Leave only one comment symbol on selected options
# Those with two commets will be ignored:
# The name to show in queue lists for this job:
#SBATCH -J train_ddpm_unconditional_celebA_256.sh

# Number of desired cpus (can be in any node):
#SBATCH --ntasks=4

# Number of desired cpus (all in same node):
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4

# Amount of RAM needed for this job:
#SBATCH --mem=50gb

# The time the job will be running:
#SBATCH --time=96:00:00

# To use GPUs you have to request them:
#SBATCH --gres=gpu:2
#SBATCH --constraint=dgx

# Set output and error files
#SBATCH --error=train_ddpm_unconditional_celebA_256.%J.err
#SBATCH --output=train_ddpm_unconditional_celebA_256.%J.out


# To load some software (you can show the list with 'module avail'):

module load pytorch
module load singularity

export OPENAI_LOGDIR="<path_save_results>/modelCifarUnconditionalCelebA_256_30K"
export PYTHONPATH="<path_python_project>/guided-diffusion"

# the program to execute with its parameters:
echo "Init job: Train images celebA"

time python <path_project>/guided-diffusion/scripts/image_train.py --data_dir <path_dataset>/CelebA_modelDiffusion/CelebA-HQ-img --attention_resolutions 32,16,8 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 3 --learn_sigma True --class_cond False --diffusion_steps 1000 --noise_schedule linear --use_kl False --lr 1e-5 --batch_size 2 --lr_anneal_steps 200000 --microbatch -1 --use_fp16 False --dropout 0.3 --pretrained False