#!/bin/sh

#BSUB -J dog_to_cat
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00


# load a scipy module
module load python3/3.12.4

# activate the virtual environment
source .venv/bin/activate

python3 train.py +experiment=mnist_to_emnist logger.offline=False trainer.log_every_n_steps=40