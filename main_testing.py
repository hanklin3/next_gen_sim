# %%
import argparse
import numpy as np
import os
import sys
import yaml

from utils import set_sumo
# from behavior_net import datasets
from behavior_net.model_inference import Predictor
from trajectory_pool import TrajectoryPool
from vehicle import Vehicle


if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
sumo_cmd = ['sumo', '-c', 'data/sumo/ring/circles.sumocfg', '--step-length', "0.4"]

path = 'configs/ring_inference.yml'

with open(path) as file:
    try:
        configs = yaml.safe_load(file)
        print(f"Loading config file: {path}")
    except yaml.YAMLError as exception:
        print(exception)
            
parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', type=str, required=True,
                    help='The name of the experiment folder where the data will be stored')
parser.add_argument('--save-result-path', type=str, default=r'./results/training/behavior_net',
                    help='The path to save the training results, a folder with experiment_name will be created in the path')
parser.add_argument('--config', type=str, required=True,
                    help='The path to the training config file. E.g., ./configs\ring_inference.yml')
args = parser.parse_args()

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


def to_vehicle(x, y, angle_deg, id, speed, road_id, lane_id, lane_index):

    v = Vehicle()
    v.location.x = x
    v.location.y = y
    v.id = id
    # sumo: north, clockwise
    # yours: east, counterclockwise
    v.speed_heading = (-angle_deg + 90 ) % 360
    v.speed = speed
    v.road_id = road_id
    v.lane_id = lane_id
    v.lane_index = lane_index

    factor = 1
    v.size.length, v.size.width = 3.6*factor, 1.8*factor
    v.safe_size.length, v.safe_size.width = 3.8*factor, 2.0*factor
    v.update_poly_box_and_realworld_4_vertices()
    v.update_safe_poly_box()
    return v

########################################
# Initialize the predictor process
model = Predictor(model=configs['model'], history_length=configs['history_length'], 
                  pred_length=configs['pred_length'], m_tokens=configs['m_tokens'], 
                  checkpoint_dir=configs['behavior_model_ckpt_dir'],
                  safety_mapper_ckpt_dir=configs['safety_mapper_ckpt_dir'])
model.initialize_net_G()
model.initialize_net_safety_mapper()

traj_pool = TrajectoryPool()

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
step = 0
while step < 600:
    print(step)

    traci.simulationStep()
    
    step += 1

    car_list = traci.vehicle.getIDList()
    print('car_list', car_list)
    vehicle_list = []
    
    for car_id in car_list:
        x,y = traci.vehicle.getPosition(car_id)
        angle_deg = traci.vehicle.getAngle(car_id)
        speed = traci.vehicle.getSpeed(car_id)
        road_id = traci.vehicle.getRoadID(car_id)
        lane_id = traci.vehicle.getLaneID(car_id)
        lane_index = traci.vehicle.getLaneIndex(car_id)
        # speed = traci.getSpeed(car_id)
        print(car_id, '(', x, y, ')', angle_deg, road_id, lane_index)
        print('road_id, lane_id', road_id, lane_index, type(road_id), type(lane_index))

        # lat, lon, cos_heading, sin_heading = converter(x, y, angle_deg)
        # print('lat, lon, cos_heading, sin_heading', 
        #       lat, lon, cos_heading, sin_heading)
        vehicle_list.append(to_vehicle(x, y, angle_deg, car_id, speed, road_id, lane_id, lane_index))

    traj_pool.update(vehicle_list)
    
    if step < 10:
        continue
    
    buff_lat, buff_lon, buff_cos_heading, buff_sin_heading, buff_vid, buff_road_id, buff_lane_id, buff_lane_index = \
        traj_pool.flatten_trajectory(
        time_length=model.history_length, max_num_vehicles=model.m_tokens, output_vid=True)
    pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, buff_lat, buff_lon = \
        model.run_forwardpass(buff_lat, buff_lon, buff_cos_heading, buff_sin_heading)

    print('pred_lat', pred_lat.shape, pred_lat)
    print('buff_vid', buff_vid.shape, buff_vid)
    print('buff_road_id', buff_road_id.shape, buff_road_id, buff_road_id[0,0].decode('utf-8'))
    print('buff_lane_id', buff_lane_index.shape, buff_lane_index, buff_lane_index[0,0])
    
    for row_idx, row in enumerate(buff_vid):
        print('row_idx, row', row_idx, row)
        vid = row[0]
        if np.isnan(vid):
            continue
        dx = np.diff(buff_lat[row_idx,:])
        dy = np.diff(buff_lon[row_idx,:])
        speed = np.sqrt(dx**2 + dy**2) / configs['sim_resol']
        # speed = [0]
        print('speed', speed)
        # traci.vehicle.setSpeed(str(int(vid)), speed[0])
        

        traci.vehicle.moveToXY(str(int(vid)), buff_road_id[row_idx, 1].decode('utf-8'), 
                        buff_lane_index[row_idx, 1], 
                        x=buff_lat[row_idx,1], y=buff_lon[row_idx,1])
        # traci.vehicle.moveToXY(str(int(vid)), buff_road_id[row_idx, 0].decode('utf-8'), 
        #         0, 
        #         x=buff_lat[row_idx,1], y=buff_lon[row_idx,1])
        

traci.close()
# %%







########################################
# Initialize the training process

# dataloaders = datasets.get_loaders(configs)
# m = Trainer(configs=configs, dataloaders=dataloaders)
# m.train_models()

