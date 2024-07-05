#!/bin/bash
#SBATCH -n 2 --gres=gpu:volta:1 -o results/training/behavior_net/ring_faster_no_jump.log-%j

source /etc/profile
source activate NNDE

python run_training_behavior_net.py --config ./configs/ring_behavior_net_training_faster.yml --experiment-name ring_behavior_net_training_faster_no_jump
