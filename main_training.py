# %%
import argparse
import numpy as np
import os
import shutil
import sys
import yaml

import logging
logging.basicConfig(level=logging.ERROR)
if os.environ['LIBSUMO'] == "1":
    # sys.path.append(os.path.join(os.environ['W'], 'sumo-1.12.0', 'tools'))
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import libsumo as traci
    print('Using libsumo')
else:
    import traci
    print('Traci')

from utils import set_sumo
# from behavior_net import datasets
from behavior_net.model_inference import Predictor
from trajectory_pool import TrajectoryPool, time_buff_to_traj_pool
from vehicle import Vehicle
from vehicle.utils_vehicle import to_vehicle
from behavior_net import datasets
from behavior_net import Trainer

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')

parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', type=str, required=True,
                    help='The name of the experiment folder where the data will be stored')
parser.add_argument('--save-result-path', type=str, default=r'./results/training/behavior_net',
                    help='The path to save the training results, a folder with experiment_name will be created in the path')
parser.add_argument('--config', type=str,
                    help='The path to the training config file. E.g., ./configs\ring_inference.yml',
                    default=r'./configs/ring_behavior_net_training_position.yml')
args = parser.parse_args()

if __name__ == '__main__':
    
    experiment_name = args.experiment_name
    save_result_path = args.save_result_path
    config_to_load = args.config
    config_save_path = os.path.join(save_result_path, experiment_name, "config.yml")
    # if config_save_path exists, load
    if os.path.exists(config_save_path):
        config_to_load = config_save_path
        print(f"Model config exists, loading from {config_to_load}")
    with open(config_to_load) as file:
        try:
            configs = yaml.safe_load(file)
            print(f"Loading config file: {config_to_load}")
        except yaml.YAMLError as exception:
            print(exception)
            
    sumo_cmd = ['sumo', '-c', 'data/sumo/ring/circles.sumocfg', '--step-length', "0.4"]
    print('Sumo configs found: ', len(configs['sumocfg_files']), configs['sumocfg_files'])
                
    # Checkpoints and training process visualizations save paths
    configs["checkpoint_dir"] = os.path.join(save_result_path, experiment_name, "checkpoints")  # The path to save trained checkpoints
    configs["vis_dir"] = os.path.join(save_result_path, experiment_name, "vis_training")  # The path to save training visualizations
    configs["experiment_name"] = experiment_name

    # Save the config file of this experiment
    os.makedirs(os.path.join(save_result_path, experiment_name), exist_ok=True)
    if not os.path.exists(config_save_path):
        shutil.copyfile(config_to_load, config_save_path)
    
    # Initialize the DataLoader
    dataloaders = datasets.get_loaders(configs)
    
    m = Trainer(configs=configs, dataloaders=dataloaders)
    m.train_models()
    
    # for batch_id, batch in enumerate(dataloaders['train'], 0):
    #     print('batch_id', batch_id)
    #     print('batch', batch)
# %%
