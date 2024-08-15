# %%
import argparse
import numpy as np
import os
import pandas as pd
import sys
import yaml

import matplotlib.pyplot as plt

from utils import set_sumo
# from behavior_net import datasets
from behavior_net.model_inference import Predictor
from trajectory_pool import TrajectoryPool
from vehicle import Vehicle
from utils import time_buff_to_traj_pool, to_vehicle

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
sumo_cmd = ['sumo', '-c', 'data/sumo/ring/circles.sumocfg', '--step-length', "0.4"]

parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', type=str, required=True,
                    help='The name of the experiment folder where the data will be stored')
parser.add_argument('--save-result-path', type=str, default=r'./results/inference/behavior_net',
                    help='The path to save the training results, a folder with experiment_name will be created in the path')
parser.add_argument('--config', type=str, required=True,
                    help='The path to the training config file. E.g., ./configs\ring_inference.yml')
args = parser.parse_args()

path = 'configs/ring_inference.yml'

with open(args.config) as file:
    try:
        configs = yaml.safe_load(file)
        print(f"Loading config file: {args.config}")
    except yaml.YAMLError as exception:
        print(exception)
            


save_result_path = args.save_result_path
experiment_name = args.experiment_name
configs["checkpoint_dir"] = os.path.join(save_result_path, experiment_name, "checkpoints")  # The path to save trained checkpoints
configs["vis_dir"] = os.path.join(save_result_path, experiment_name, "vis_training")  # The path to save training visualizations


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


# def to_vehicle(x, y, angle_deg, id, speed, road_id, lane_id, lane_index):

#     v = Vehicle()
#     v.location.x = x
#     v.location.y = y
#     v.id = id
#     # sumo: north, clockwise
#     # yours: east, counterclockwise
#     v.speed_heading = (-angle_deg + 90 ) % 360
#     v.speed = speed
#     v.road_id = road_id
#     v.lane_id = lane_id
#     v.lane_index = lane_index

#     factor = 1
#     v.size.length, v.size.width = 3.6*factor, 1.8*factor
#     v.safe_size.length, v.safe_size.width = 3.8*factor, 2.0*factor
#     v.update_poly_box_and_realworld_4_vertices()
#     v.update_safe_poly_box()
#     return v

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

import traci

traci.start(sumo_cmd)
# %%

TIME_BUFF = []
rolling_step = configs['rolling_step']
history_length = configs['history_length']
sim_resol = configs['sim_resol']
output_type = configs['model_output_type']

dataf = []
df_predicted = []

step = 0
while step < 1000:
    print(step)

    traci.simulationStep()
    
    step += 1

    car_list = traci.vehicle.getIDList()
    print('car_list', car_list)
    vehicle_list = []
    
    for car_id in car_list:
        x,y = traci.vehicle.getPosition(car_id)
        angle_deg = traci.vehicle.getAngle(car_id)
        # speed = traci.vehicle.getSpeed(car_id)
        speed = traci.vehicle.getLateralSpeed(car_id)
        acceleration = traci.vehicle.getAcceleration(car_id)
        road_id = traci.vehicle.getRoadID(car_id)
        lane_id = traci.vehicle.getLaneID(car_id)
        lane_index = traci.vehicle.getLaneIndex(car_id)
        # speed = traci.getSpeed(car_id)
        # print(car_id, '(', x, y, ')', angle_deg, road_id, lane_index)
        # print('road_id, lane_id', road_id, lane_index, type(road_id), type(lane_index))

        # lat, lon, cos_heading, sin_heading = converter(x, y, angle_deg)
        # print('lat, lon, cos_heading, sin_heading', 
        #       lat, lon, cos_heading, sin_heading)
        vehicle_list.append(to_vehicle(x, y, angle_deg, car_id, speed, road_id, 
                                        lane_id, lane_index, acceleration))
        
        if step > history_length:
            dataf.append([int(0), step * sim_resol, int(car_id), 
                        float(x), float(y)])
                
    TIME_BUFF.append(vehicle_list)
        
    if step < history_length:
        continue
    
    assert len(TIME_BUFF) == history_length
    traj_pool = time_buff_to_traj_pool(TIME_BUFF)
    
    buff_lat, buff_lon, buff_cos_heading, buff_sin_heading, \
            buff_vid, buff_speed, buff_acc, buff_road_id, buff_lane_id, buff_lane_index = \
        traj_pool.flatten_trajectory(
            time_length=model.history_length, max_num_vehicles=model.m_tokens, output_vid=True)

    pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, buff_lat, buff_lon = \
        model.run_forwardpass(buff_lat, buff_lon, buff_cos_heading, buff_sin_heading)

    pred_speed = pred_lat
    pred_acceleration = pred_lon
        
    TIME_BUFF = TIME_BUFF[rolling_step:]

    print('pred_lat', pred_lat.shape, pred_lat) # [32, 5]
    print('buff_vid', buff_vid.shape, buff_vid) # [32, 5], each row same car id
    print('buff_road_id', buff_road_id.shape, buff_road_id, buff_road_id[0,0].decode('utf-8'))
    print('buff_lane_id', buff_lane_index.shape, buff_lane_index, buff_lane_index[0,0])

    ## Record prediction to dataframe for metrics later
    rows, cols = buff_vid.shape
    assert pred_lat.shape == buff_vid.shape
    for irow in range(rows):
        # for icol in range(cols):
            icol = 0
            nextx, nexty = pred_lat[irow, icol], pred_lon[irow, icol]
            next_vid = buff_vid[irow, icol]
            if np.isnan(next_vid):
                # reach max vehicles
                continue
            df_predicted.append([int(0), (step+1) * sim_resol, int(next_vid), 
                                float(nextx), float(nexty)])
    
    for row_idx, row in enumerate(buff_vid):
        print('row_idx, row', row_idx, row)
        vid = row[0]
        if np.isnan(vid):
            continue

        # output_type = 'position'
        # output_type = 'speed'
        # output_type = 'no_set'
        if output_type == 'position':
            dx = np.diff(pred_lat[row_idx,:])
            dy = np.diff(pred_lon[row_idx,:])
            speed = np.sqrt(dx**2 + dy**2) / configs['sim_resol']
            # speed = max(dx / configs['sim_resol'], dy / configs['sim_resol'])
            print('dx', dx.shape, dx)
            # print('speed', speed)
            
            # assert speed[0] > 0, (speed, pred_speed[row_idx,:], pred_speed[row_idx,:])
            traci.vehicle.setSpeed(str(int(vid)), speed[0])
            # traci.setPreviousSpeed(str(int(vid)), speed[0])
        
        elif output_type == 'speed':
            ####################Speed
            # print('pred_speed', pred_speed[row_idx,0])
            traci.vehicle.setSpeed(str(int(vid)), pred_speed[row_idx,0])
        elif output_type == 'no_set':
            pass
        elif output_type == 'acceleration':
            ####################Acce
            # print('pred_acceleration', pred_acceleration[row_idx,0])
            # traci.vehicle.setAcceleration(str(int(vid)), pred_acceleration[row_idx,0], 0.4)
            traci.vehicle.setAccel(str(int(vid)), pred_acceleration[row_idx,0])
            #####################
            # dx = np.diff(buff_lat[row_idx,:], n=2)
            # dy = np.diff(buff_lon[row_idx,:], n=2)
            # acceleration = np.sqrt(dx**2 + dy**2) / configs['sim_resol']
            
            # print('acceleration', acceleration)
            
            # traci.vehicle.setAcceleration(str(int(vid)), acceleration[0], 0.1)

        #####################
        # traci.vehicle.moveToXY(str(int(vid)), buff_road_id[row_idx, 0].decode('utf-8'), 
        #                 buff_lane_index[row_idx, 0], 
        #                 x=pred_lat[row_idx,0], y=pred_lon[row_idx,0])
        # traci.vehicle.moveToXY(str(int(vid)), buff_road_id[row_idx, 0].decode('utf-8'), 
        #         0, 
        #         x=buff_lat[row_idx,1], y=buff_lon[row_idx,1])
        else:
            assert False, "Unsupported model output type"

traci.close()

################################################################
# %%

save_path = os.path.join(save_result_path, experiment_name)
os.makedirs(save_path, exist_ok=True)

arr = np.asarray(dataf)
df_traj = pd.DataFrame(arr,
                       columns=['Simulation No', 'Time', 'Car', 'x', 'y'])
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
if output_type == 'no_set':
    save_file = os.path.join(save_path, f'df_traj_gt_{1000}.csv')
else:
    save_file = os.path.join(save_path, f'df_traj_{1000}.csv')
df_traj.to_csv(save_file)
print('saved to ', save_file)
########################################
# Save model prediction dataframe
arr_predicted = np.asarray(df_predicted)
df_traj_predicted = pd.DataFrame(arr_predicted,
                       columns=['Simulation No', 'Time', 'Car', 'x', 'y'])
df_traj_predicted['Simulation No'] = df_traj['Simulation No'].astype(int)
df_traj_predicted['Car'] = df_traj['Car'].astype(int)

if output_type == 'no_set':
    save_file = os.path.join(save_path, f'df_traj_pred_open_loop_{1000}.csv')
else:
    save_file = os.path.join(save_path, f'df_traj_pred_{1000}.csv')
df_traj_predicted.to_csv(save_file)
print('saved to ', save_file)
##########################
# Initialize the training process

# dataloaders = datasets.get_loaders(configs)
# m = Trainer(configs=configs, dataloaders=dataloaders)
# m.train_models()

