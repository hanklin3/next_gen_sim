#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1 -o results/training/behavior_net/ring_position_dxdy_test.l-%j

source /etc/profile
# source activate NNDE


export PYTHONPATH=$PYTHONPATH:/home/gridsan/thlin/cameraculture_shared/hanklin/next_gen_sim/

export LIBSUMO=0 && python3 main_training.py --config ./configs/ring_behavior_net_training_position_dxdy.yml --experiment-name ring_position_dxdy_test

# python3 main_training.py --config ./configs/ring_behavior_net_training_position.yml --experiment-name ring_position_close_multiConf_libsumo_9-16
