#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1 -o results/training/behavior_net/ring_speed.log-%j

source /etc/profile
source activate NNDE

python3 main_training.py --config ./configs/ring_behavior_net_training_speed.yml --experiment-name ring_speed_test

# python3 main_training.py --config ./configs/ring_behavior_net_training_speed.yml --experiment-name ring_speed
