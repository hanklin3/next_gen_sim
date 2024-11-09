#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1 -o results/training/behavior_net/ring_position_close_loop.log-%j

source /etc/profile
source activate NNDE

export PYTHONPATH=$PYTHONPATH:/home/gridsan/thlin/cameraculture_shared/hanklin/next_gen_sim/
export LIBSUMO=0
# export LIBSUMO=0 && python3 main_training.py --config ./configs/ring_behavior_net_training_position.yml --experiment-name ring_position_test

# python3 main_training.py --config ./configs/ring_behavior_net_training_position.yml --experiment-name ring_position_close_multiConf_libsumo_9-16

# export LIBSUMO=0 && python3 main_training.py --config ./results/training/behavior_net/0007_local_0006__refine_pos_close_dxdy_setSpeed_11-04/config_load.yml --experiment-name 0007_local_0006__refine_pos_close_dxdy_setSpeed_11-04
# export LIBSUMO=0 && python3 main_training.py --config ./configs/ring_behavior_net_training_position.yml --experiment-name 0007_local_0006__refine_pos_close_dxdy_setSpeed_11-04
# export LIBSUMO=0 && python3 main_training.py --config ./configs/ring_behavior_net_training_position.yml --experiment-name 0007-2_local_0007__refine_pos_close_xy_moveToXY_cossin2deg_11-05

# export LIBSUMO=0 && python3 main_training.py --config ./configs/ring_behavior_net_training_position.yml --experiment-name 0008_pos_open_pred35_11-04

# export LIBSUMO=0 && python3 main_training.py --config ./configs/ring_behavior_net_training_position.yml --experiment-name 0009_pos_open_pred35_goals_11-04

# export LIBSUMO=0 && python3 main_training.py --config ./configs/ring_behavior_net_training_position.yml --experiment-name 0008-2__0008_refine_pos_close_pred35_xy_moveToXY_cossin2deg_11-05

python3 main_training.py --experiment-name 0009-2__0009_refine_pos_close_pred35_goals_moveToXY_cossin2deg_11-05