# %%
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
# from shapely.geometry import Point
import os
import time
# import shutil
from PIL import Image, ImageDraw 
from vehicle import Vehicle
os.environ['LIBSUMO'] = '0'
from vehicle.utils_vehicle import to_vehicle, traci_get_vehicle_data
from trajectory_pool import TrajectoryPool, time_buff_to_traj_pool
from utils import set_sumo
import pickle
import yaml


import traci

# %%

base_dir = '/mnt/d/OneDrive - Massachusetts Institute of Technology/Research/sumo/'
base_dir = './data/sumo/'
assert os.path.exists(base_dir)

# experiment='ring_faster'
experiment='ring'
# experiment='ring_larger'
# experiment='ring_w_goals'

# %%
config = './configs/ring_behavior_net_training_position.yml'

with open(config) as file:
    try:
        configs = yaml.safe_load(file)
        print(f"Loading config file: {config}")
    except yaml.YAMLError as exception:
        print(exception)

sumo_cmd = set_sumo(configs['gui'], 
                    configs['sumocfg_files'][0], configs['max_steps'], configs['sim_resol'])
print('sumo_cmd', sumo_cmd)

time_start = time.time()
max_steps = configs['max_steps']
history_length = configs['history_length']
sim_resol = configs['sim_resol']
step = 0
dataf = []

# %%
traci.start(sumo_cmd)

while step < max_steps:
    traci.simulationStep() # run 1 step for sim1

    step += 1
        
    vehicle_list = traci_get_vehicle_data()

    for veh in vehicle_list:  
        dataf.append([int(0), step * sim_resol, int(veh.id), 
                    float(veh.location.x), float(veh.location.y), float(veh.speed), 
                    float(veh.speed_heading_deg), float(veh.acceleration),
                    veh.road_id, veh.lane_id, veh.lane_index])

traci.close()

arr = np.asarray(dataf)
DF_HEADER = ['Simulation No', 'Time', 'Car', 'x', 'y', 'Speed', 'Heading', 
             'Acceleration', 'Road ID', 'Lane ID', 'Lane Index']
sim_df = pd.DataFrame(arr,
                       columns=DF_HEADER)

# %%
# plot_save_path = f'./data/sumo/{experiment}/plots'
# t = 0.
# count = 0
# while t < np.max(sim_df['time']):
#     t = round(t, 2)
#     init_df = sim_df[sim_df['time']==t]
#     vxs = init_df['x'].tolist()
#     vys = init_df['y'].tolist()
#     ids = init_df['id'].tolist()
#     plt.figure()
#     plt.plot(vxs, vys, '.')
#     for iid in range(len(ids)):
#         plt.text(vxs[iid], vys[iid], ids[iid])
#     plt.axis('equal')
#     # plt.savefig(f'{plot_save_path}/{count:02d}.jpg')
#     if count % 10 == 0:
#         print('Saved to', f'{plot_save_path}/{count:02d}.jpg')

#     count += 1
#     t += 0.4

#     if count > 5:
#         break


minx, maxx = min(sim_df['x']), max(sim_df['x'])
miny, maxy = min(sim_df['y']), max(sim_df['y'])
print('minx, maxx', minx, maxx)
print('miny, maxy', miny, maxy)

sim_df
# %%

plt.scatter(sim_df['x'],sim_df['y'])
plt.axis('equal')

minx, maxx = min(sim_df['x']), max(sim_df['x'])
miny, maxy = min(sim_df['y']), max(sim_df['y'])
print('minx, maxx', minx, maxx)
print('miny, maxy', miny, maxy)
print('Car id', np.unique(sim_df['Car']))

print('max time', max(sim_df['Time']))


# %%

def init_pickle(vehicle_list):
    path = f'./data/inference/{experiment}/simulation_initialization/gen_veh_states/ring'
    output_file_path = os.path.join(path, 'initial_vehicle_dict.pickle')
    vehicle_dict = {'n_in1': vehicle_list}
    pickle.dump(vehicle_dict, open(output_file_path, "wb"))

# %%
traci.start(sumo_cmd)

t = 0.0
count = 0
while t < max_steps:
    t = round(t, 2)

    traci.simulationStep() # run 1 step for sim1

    vehicle_list = traci_get_vehicle_data()

    if t==0:
        init_pickle(vehicle_list)
        # t += 0.4
        # count += 1
        # continue
    
    path = f'./data/inference/{experiment}/simulation_initialization/initial_clips/ring-01/01/'
    output_file_path = os.path.join(path,f"{count:06d}.pickle")
    print('output_file_path', output_file_path)
    pickle.dump(vehicle_list, open(output_file_path, "wb"))

    if t <= 900.0:
        path = f'./data/training/behavior_net/{experiment}/ring257/train/01/01'
        output_file_path = os.path.join(path,f"{count:06d}.pickle")
        print('output_file_path', output_file_path)
        pickle.dump(vehicle_list, open(output_file_path, "wb"))
    else:
        path = f'./data/training/behavior_net/{experiment}/ring257/val/01/01'
        output_file_path = os.path.join(path,f"{count:06d}.pickle")
        print('output_file_path', output_file_path)
        pickle.dump(vehicle_list, open(output_file_path, "wb"))

    if int(t) %100 == 0:
        print('t', t)

    t += 0.4
    count += 1

traci.close()
# %%
traci.close()

# %%
t=6
# vehicle_list = get_veh_list(t)
vxs = [v.location.x for v in vehicle_list]
vys = [v.location.y for v in vehicle_list]
plt.plot(vxs, vys, '.')
plt.axis('equal')
plt.grid()

# %%
# check data x y range
plt.scatter(sim_df['x'],sim_df['y'])
plt.axis('equal')
plt.grid()
minx, maxx = min(sim_df['x']), max(sim_df['x'])
miny, maxy = min(sim_df['y']), max(sim_df['y'])
print('minx, maxx', minx, maxx)
print('miny, maxy', miny, maxy)




# pos = [float(x) for x in  sim_df['pos'][0].split()]
# x, y, z = pos
# print(pos)

# positions = []
# for index, row in sim_df.iterrows():
#     positions.append([float(x) for x in  row['pos'].split()])

# print(positions)
# positions = np.asarray(positions)
# print(positions.shape, positions)

assert False, 'Done'
# %%
plt.imshow(img)
img2 = Image.fromarray(img)
img3 = ImageDraw.Draw(img2)
# %%
img3 = ImageDraw.Draw(img2)
# %%
plt.imshow(np.asarray(img3))
# %%
img = np.zeros(img_size)
# plt.imsave(os.path.join(base_dir, 'ring-sim-remove-vehicle-area-map.jpg'), img)
# %%
