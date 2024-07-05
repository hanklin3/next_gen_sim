from __future__ import absolute_import
from __future__ import print_function

import argparse
from behavior_net.networks import define_G, define_safety_mapper
from behavior_net.model_inference import Predictor
import os
import torch
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--folder-idx', type=int, default='1', metavar='N',
                    help='Worker id of the running experiment')
parser.add_argument('--experiment-name', type=str, required=True,
                    help='The name of the experiment folder where the data will be stored')
parser.add_argument('--save-result-path', type=str, default=r'./results/inference/',
                    help='The path to save the simulation results, a folder with experiment_name will be created in the path')
parser.add_argument('--config', type=str, required=True,
                    help='The path to the simulation config file. E.g., ./configs/AA_rdbt_inference.yml')
parser.add_argument('--viz-flag', action='store_true', help='Default is False, adding this argument will overwrite the same flag in config file')

args = parser.parse_args()


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)



# Load config file
with open(args.config) as file:
    try:
        configs = yaml.safe_load(file)
        print(f"Loading config file: {args.config}")
    except yaml.YAMLError as exception:
        print(exception)
        # configs = yaml.load(file)

# Settings
configs["device"] = torch.device("cuda:0" if configs["use_gpu"] else "cpu")
print(f"Using device: {configs['device']}...")
print(f"Simulating {configs['dataset']} using {configs['model']} model...")
print('Using conflict critic module!') if configs["use_conflict_critic_module"] else print('Not using conflict critic module!')
print(f'Using neural safety mapper!' if configs["use_neural_safety_mapping"] else 'Not using neural safety mapper!')
print("Checkpointd_dir", configs['behavior_model_ckpt_dir'])
assert (configs["rolling_step"] <= configs["pred_length"])
configs["viz_flag"] = configs["viz_flag"] or args.viz_flag  # The visualization flag can be easily modified through input argument.

assert os.path.exists(configs['behavior_model_ckpt_dir']), f"Not found {configs['behavior_model_ckpt_dir']}"

history_length, pred_length, m_tokens = configs["history_length"], configs["pred_length"], configs["m_tokens"]
rolling_step = configs["rolling_step"]

sim = Predictor(model=configs["model"], history_length=history_length, pred_length=pred_length, m_tokens=m_tokens,
                                    checkpoint_dir=configs["behavior_model_ckpt_dir"],
                                    safety_mapper_ckpt_dir=configs["safety_mapper_ckpt_dir"], device=device,)
