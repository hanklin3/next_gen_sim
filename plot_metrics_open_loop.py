# %%
import argparse
import numpy as np
import os
import shutil
import sys
import yaml

import logging
logging.basicConfig(level=logging.WARNING)

from utils import set_sumo
# from behavior_net import datasets
from behavior_net.model_inference import Predictor
from trajectory_pool import TrajectoryPool
from vehicle import Vehicle
from utils import time_buff_to_traj_pool, to_vehicle
from behavior_net import datasets
from behavior_net import Trainer

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')

parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', type=str, required=True,
                    help='The name of the experiment folder where the data will be stored')
parser.add_argument('--save-result-path', type=str, default=f'',
                    help='The path to save the training results, a folder with experiment_name will be created in the path')
parser.add_argument('--config', type=str, required=True,
                    help='The path to the training config file. E.g., ./configs\ring_inference.yml')
args = parser.parse_args()

# python plot_metrics_open_loop.py --experiment-name ring_speed_test --config configs/ring_inference.yml

if __name__ == '__main__':
    
    with open(args.config) as file:
        try:
            configs = yaml.safe_load(file)
            print(f"Loading config file: {args.config}")
        except yaml.YAMLError as exception:
            print(exception)
            
    sumo_cmd = ['sumo', '-c', 'data/sumo/ring/circles.sumocfg', '--step-length', "0.4"]
    sumo_cmd = set_sumo(configs['gui'], 
                    configs['sumocfg_file_name'], configs['max_steps'], configs['sim_resol'])
    print('sumo_cmd', sumo_cmd)
                
    # Checkpoints and training process visualizations save paths
    experiment_name = args.experiment_name
    save_result_path = args.save_result_path
    if save_result_path == '':
        save_result_path = f'./results/inference/behavior_net/{experiment_name}'
    path = f'{save_result_path}/df_traj_1000.csv'
    path_pred = f'{save_result_path}/df_traj_pred_1000.csv'

    configs["checkpoint_dir"] = os.path.join(save_result_path, experiment_name, "checkpoints")  # The path to save trained checkpoints
    configs["vis_dir"] = os.path.join(save_result_path, experiment_name, "vis_training")  # The path to save training visualizations

    # Save the config file of this experiment
    # os.makedirs(os.path.join(save_result_path), exist_ok=True)
    save_path = os.path.join(save_result_path, "config.yml")
    shutil.copyfile(args.config, save_path)
    
    # Initialize the DataLoader
    dataloaders = datasets.get_loaders(configs, sumo_cmd)
    
    # m = Trainer(configs=configs, dataloaders=dataloaders)
    # m.train_models()

    dataloaders['val']
    
    # for batch_id, batch in enumerate(dataloaders['train'], 0):
    #     print('batch_id', batch_id)
    #     print('batch', batch)