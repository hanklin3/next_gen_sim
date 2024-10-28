# %%
import argparse
import numpy as np
import os
import pandas as pd
import shutil
import sys
import time
import yaml

import matplotlib.pyplot as plt

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
from vehicle.utils_vehicle import (to_vehicle, 
    traci_get_vehicle_data, traci_set_vehicle_state, cossin2deg)

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
sumo_cmd = ['sumo', '-c', 'data/sumo/ring/circles.sumocfg', '--step-length', "0.4"]

parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', type=str, default=r'',
                    help='The name of the experiment folder where the data will be stored')
parser.add_argument('--save-result-path', type=str, default=r'./results/inference/behavior_net',
                    help='The path to save the training results, a folder with experiment_name will be created in the path')
parser.add_argument('--config', type=str, required=True,
                    help='The path to the training config file. E.g., ./configs\ring_inference.yml')
args = parser.parse_args()

with open(args.config) as file:
    try:
        configs = yaml.safe_load(file)
        print(f"Loading config file: {args.config}")
    except yaml.YAMLError as exception:
        print(exception)
            


save_result_path = args.save_result_path
experiment_name = args.experiment_name
if experiment_name == '':
    experiment_name = configs['behavior_model_ckpt_dir'].split('/')[4]
print('experiment_name:', experiment_name)
configs["checkpoint_dir"] = os.path.join(save_result_path, experiment_name, "checkpoints")  # The path to save trained checkpoints
configs["vis_dir"] = os.path.join(save_result_path, experiment_name, "vis_training")  # The path to save training visualizations
os.makedirs(os.path.join(save_result_path, experiment_name), exist_ok=True)
save_path = os.path.join(save_result_path, experiment_name, "config.yml")
shutil.copyfile(args.config, save_path)

def converter(x, y, angles_deg):
    lat, lon = x, y
    # sumo: north, clockwise
    # neuralNDE: east, counterclockwise
    # convert from sumo to neuralNDE format
    heading = (-angles_deg + 90 ) % 360
    heading = np.radians(heading)  # Convert degrees to radians
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)
    return lat, lon, cos_heading, sin_heading

########################################
# Initialize the predictor process
# Predictor will load model
model = Predictor(model=configs['model'], history_length=configs['history_length'], 
                  pred_length=configs['pred_length'], m_tokens=configs['m_tokens'], 
                  checkpoint_dir=configs['behavior_model_ckpt_dir'],
                  safety_mapper_ckpt_dir=configs['safety_mapper_ckpt_dir'])

# traj_pool = TrajectoryPool()


# veh_num = configs['m_tokens']
# time_length = configs['pred_length']
# buff_lat = np.empty([veh_num, time_length])
# buff_lat[:] = np.nan
# buff_lon = np.empty([veh_num, time_length])
# buff_lon[:] = np.nan
# buff_cos_heading = np.empty([veh_num, time_length])
# buff_cos_heading[:] = np.nan
# buff_sin_heading = np.empty([veh_num, time_length])
# buff_sin_heading[:] = np.nan
# buff_vid = np.empty([veh_num, time_length])
# buff_vid[:] = np.nan


############## SUMO ##################
# %%
sumo_cmd = set_sumo(configs['gui'], 
                    configs['sumocfg_file_name'], configs['max_steps'], configs['sim_resol'])
# sumo_cmd = set_sumo(configs['gui'], 
#                     configs['sumocfg_file_name'], configs['max_steps'])
print('sumo_cmd', sumo_cmd)

DF_HEADER = ['Simulation No', 'Time', 'Car', 'x', 'y', 'Speed', 'Heading']

use_gt_prediction = configs['use_gt_prediction']
print('use_gt_prediction', use_gt_prediction)

traci.start(sumo_cmd, label="sim1")
if use_gt_prediction:
    traci.start(sumo_cmd, label="gt")
# %%

TIME_BUFF = []
rolling_step = configs['rolling_step']
history_length = configs['history_length']
sim_resol = configs['sim_resol']
model_output = configs['model_output']

dataf = []
df_predicted = []

if use_gt_prediction:
    # gt one ahead
    traci.switch("gt")
    traci.simulationStep() # run 1 step for sim2

time_start = time.time()
step_max = configs['max_steps']
step = 0
while step < step_max:
    print(step)

    if use_gt_prediction:
        traci.switch("gt")
        traci.simulationStep() # run 1 step for sim2
    traci.switch("sim1")
    traci.simulationStep() # run 1 step for sim1

    step += 1
        
    vehicle_list = traci_get_vehicle_data()

    if step > history_length:
        for veh in vehicle_list:  
            dataf.append([int(0), step * sim_resol, int(veh.id), 
                        float(veh.location.x), float(veh.location.y), float(veh.speed), float(veh.speed_heading_deg)])
                
    TIME_BUFF.append(vehicle_list)
        
    if step < history_length:
        continue
    
    assert len(TIME_BUFF) == history_length
    traj_pool = time_buff_to_traj_pool(TIME_BUFF)
    
    buff_lat, buff_lon, buff_cos_heading, buff_sin_heading, \
            buff_vid, buff_speed, buff_acc, buff_road_id, buff_lane_id, buff_lane_index, buff_time = \
        traj_pool.flatten_trajectory(
            time_length=model.history_length, max_num_vehicles=model.m_tokens, output_vid=True)

    pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, buff_lat, buff_lon = \
        model.run_forwardpass(buff_lat, buff_lon, buff_cos_heading, buff_sin_heading)

    if use_gt_prediction:
        traci.switch("gt")
        vehicle_list_gt = traci_get_vehicle_data()
        traj_pool_gt = time_buff_to_traj_pool([vehicle_list_gt])

        pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, \
            buff_vid, buff_speed, buff_acc, buff_road_id, buff_lane_id, buff_lane_index, buff_time = \
        traj_pool_gt.flatten_trajectory(
                time_length=model.history_length, max_num_vehicles=model.m_tokens, output_vid=True)
        pred_lat = pred_lat[:, ::-1]
        pred_lon = pred_lon[:, ::-1]
        pred_cos_heading = pred_cos_heading[:, ::-1]
        pred_sin_heading = pred_sin_heading[:, ::-1]
        buff_vid = buff_vid[:, ::-1]
        buff_speed = buff_speed[:, ::-1]
        buff_acc = buff_acc[:, ::-1]
        buff_road_id = buff_road_id[:, ::-1]
        buff_lane_id = buff_lane_id[:, ::-1]
        buff_lane_index = buff_lane_index[:, ::-1]
        buff_time = buff_time[:, ::-1]
        # print('buff_time before', buff_time)
        # print('buff_time after', buff_time)
        # assert False
        traci.switch("sim1")
    
    pred_speed = pred_lat
    pred_acceleration = pred_lon
        
    TIME_BUFF = TIME_BUFF[rolling_step:]

    print('pred_lat', pred_lat.shape, pred_lat) # [32, 5]
    print('buff_vid', buff_vid.shape, buff_vid) # [32, 5], each row same car id
    # print('buff_road_id', buff_road_id.shape, buff_road_id, buff_road_id[0,0].decode('utf-8'))
    print('buff_lane_index', buff_lane_index.shape, buff_lane_index, buff_lane_index[0,0])
    print('buff_speed', buff_speed.shape, buff_speed)
    print('buff_time', buff_time.shape, buff_time)

    ## Record prediction to dataframe for metrics later
    rows, cols = buff_vid.shape
    assert pred_lat.shape == buff_vid.shape
    for irow in range(rows):
        # for icol in range(cols):
            icol = 0
            nextx, nexty = pred_lat[irow, icol], pred_lon[irow, icol]
            next_vid = buff_vid[irow, icol]
            next_speed = pred_speed[irow, icol]
            
            sin_heading = pred_sin_heading[irow][icol]
            cos_heading = pred_cos_heading[irow][icol]
            angle_deg = cossin2deg(sin_heading, cos_heading)
            # print('next_vid', next_vid)
            if np.isnan(next_vid):
                # reach max vehicles
                continue
            df_predicted.append([int(0), (step+1) * sim_resol, int(next_vid), 
                                float(nextx), float(nexty), float(next_speed), float(angle_deg)])
    
    traci_set_vehicle_state(model_output, buff_vid,
                            pred_lat, pred_lon, 
                            pred_cos_heading, pred_sin_heading,
                            pred_speed, pred_acceleration, 
                            buff_lat, buff_lon, buff_cos_heading, buff_sin_heading,
                            configs['sim_resol'])

traci.close()

time_end = time.time()
print(f"Inference time: {time_end-time_start}s")

#libsumo: Inference time: 8.278995275497437s 
#traci: Inference time: 20.589370489120483s
################################################################
# %%

save_path = os.path.join(save_result_path, experiment_name)
os.makedirs(save_path, exist_ok=True)

arr = np.asarray(dataf)
df_traj = pd.DataFrame(arr,
                       columns=DF_HEADER)
df_traj['Simulation No'] = df_traj['Simulation No'].astype(int)
df_traj['Car'] = df_traj['Car'].astype(int)

df = df_traj[df_traj['Car']==13]
plt.scatter(df['x'].values, df['y'].values)
plt.gca().set_aspect("equal")

##################Compute distance by circle arc ###############################
dia =  np.max(df_traj['x']) - np.min(df_traj['x'])
dia2 =  np.max(df_traj['y']) - np.min(df_traj['y'])
dia = max(dia,dia2)
r = dia/2
print('dia', dia, 'radius', r)
# %%
Xstart, Ystart = (np.min(df_traj['x']) + np.max(df_traj['x']))/2, np.min(df_traj['y'])
Xstart, Ystart
# %%
def dist(x1,x2, y1,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_pos(x, y):
    d = dist(Xstart,x, Ystart, y)
    # print('1-d**2/2*r**2', 1 - d**2/(2*r**2))
    rad = np.arccos(1 - d**2/(2*r**2))

    
    length = r * rad
    if np.isnan(length):
        print('nan',x, y)
        print('1-d**2/2*r**2', 1 - d**2/(2*r**2))
        print('rad', rad)
    return length

def get_pos2(x, y):
    rad = np.arctan2(y-Ystart, x-Xstart)
    print('rad', rad)
    rad[rad<0] = rad[rad<0] + 2*np.pi
    # if rad < 0:
    #     rad += 2*np.pi
    length = r * rad
    return length

# %%
for index, row in df.iterrows():
    #print(row['x'], row['y'])
    # pos = get_pos(row['x'], row['y'])
    # print('pos', get_pos(row['x'], row['y']))
    print('pos2', get_pos2(np.asarray([row['x']]), np.asarray([row['y']])))
    # break
# %%
poss = get_pos2(df['x'].values, df['y'].values)
poss
# %%
np.arccos([1, -1])
# %%
poss = get_pos2(df_traj['x'].values, df_traj['y'].values)
poss
#######################################################
# %%
df_traj.insert(len(df_traj.columns), "Position", poss)
df_traj
# %%
df_traj[df_traj['Car']==21]['Position']
# %%
print(np.unique(df_traj['Car']))

# %%
###########################
# Save sumo simuilation dataframe
if model_output == 'no_set':
    save_file = os.path.join(save_path, f'df_traj_sumo_gt_{step_max}.csv')  # sumo gt log data
elif use_gt_prediction:
    save_file = os.path.join(save_path, f'df_traj_sumo_close_sumoPRED_{step_max}.csv') # sumo sim close loop - perfect model
else:
    save_file = os.path.join(save_path, f'df_traj_sumo_close_{step_max}.csv')  # sumo sim data (from model output)
df_traj.to_csv(save_file)
print('saved to ', save_file)
########################################
# Save model prediction dataframe
arr_predicted = np.asarray(df_predicted)
df_traj_predicted = pd.DataFrame(arr_predicted,
                       columns=DF_HEADER)
df_traj_predicted['Simulation No'] = df_traj['Simulation No'].astype(int)
df_traj_predicted['Car'] = df_traj['Car'].astype(int)
pos2 = get_pos2(df_traj_predicted['x'].values, df_traj_predicted['y'].values)
df_traj_predicted.insert(len(df_traj_predicted.columns), "Position", pos2)

if model_output == 'no_set':
    save_file = os.path.join(save_path, f'df_traj_pred_open_loop_{step_max}.csv')  # model open loop pred on log data
elif use_gt_prediction:
    save_file = os.path.join(save_path, f'df_traj_pred_close_loop_sumoPRED_{step_max}.csv') # sumo close loop pred on log data
else:
    save_file = os.path.join(save_path, f'df_traj_pred_close_loop_{step_max}.csv') # model close loop pred on sim data
df_traj_predicted.to_csv(save_file)
print('saved to ', save_file)
##########################
# Initialize the training process

# dataloaders = datasets.get_loaders(configs)
# m = Trainer(configs=configs, dataloaders=dataloaders)
# m.train_models()

