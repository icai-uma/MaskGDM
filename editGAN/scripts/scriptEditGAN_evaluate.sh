#!/usr/bin/env bash
# Leave only one comment symbol on selected options
# Those with two commets will be ignored:
# The name to show in queue lists for this job:
#SBATCH -J editGAN_evaluate_face.sh

# Number of desired cpus (can be in any node):
#SBATCH --ntasks=4

# Number of desired cpus (all in same node):
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4

# Amount of RAM needed for this job:
#SBATCH --mem=20gb

# The time the job will be running:
#SBATCH --time=96:00:00

# To use GPUs you have to request them:
#SBATCH --gres=gpu:1
#SBATCH --constraint=dgx
##SBATCH --nodelist=exa04


##SBATCH --constraint=cal

# Set output and error files
#SBATCH --error=editGAN_evaluate_face.%J.err
#SBATCH --output=editGAN_evaluate_face.%J.out

# To load some software (you can show the list with 'module avail'):
module load pytorch
module load singularity

pip install imageio
pip install scikit-image

export PYTHONPATH="./maskGDM/maskgdm/editGAN"

time python ./maskGDM/maskgdm/editGAN/evaluateClassifier.py --exp ./maskGDM/maskgdm/editGAN/experiments/tool_face.json