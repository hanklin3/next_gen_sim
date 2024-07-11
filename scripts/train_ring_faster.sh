#!/bin/bash
#SBATCH -n 2 --gres=gpu:volta:1 -o results/training/behavior_net/ring_1.log-%j

source /etc/profile
source activate NNDE

python3 main_training.py --config ./configs/ring_behavior_net_training_all.yml --experiment-name ring_1
