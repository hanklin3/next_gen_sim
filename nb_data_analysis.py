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

sim_df['Time'] = sim_df['Time'].astype(float)
sim_df['x'] = sim_df['x'].astype(float)
sim_df['y'] = sim_df['y'].astype(float)
sim_df['Car'] = sim_df['Car'].astype(int)
sim_df['Heading'] = sim_df['Heading'].astype(float)
sim_df['Speed'] = sim_df['Speed'].astype(float)
sim_df['Lane ID'] = sim_df['Lane ID'] # str


# %%



minx, maxx = min(sim_df['x']), max(sim_df['x'])
miny, maxy = min(sim_df['y']), max(sim_df['y'])
print('minx, maxx', minx, maxx)
print('miny, maxy', miny, maxy)

sim_df
# %%
plot_save_path = f'./data/sumo/{experiment}/plots'
t = 0.
count = 0
while t < np.max(sim_df['Time']):
    t = round(t, 2)
    print('t', t)
    init_df = sim_df[sim_df['Time']==t]
    vxs = init_df['x'].tolist()
    vys = init_df['y'].tolist()
    ids = init_df['Car'].tolist()
    plt.figure()
    plt.plot(vxs, vys, '.')
    for iid in range(len(ids)):
        plt.text(vxs[iid], vys[iid], ids[iid])
    plt.axis('equal')
    # plt.savefig(f'{plot_save_path}/{count:02d}.jpg')
    # if count % 10 == 0:
    #     print('Saved to', f'{plot_save_path}/{count:02d}.jpg')

    count += 1
    t += 0.4

    if count > 10:
        break
# %%
times = np.unique(sim_df['Time'])
plot_save_path = f'./data/sumo/{experiment}/plots'
t = 0.
count = 0
# for index, row in sim_df.iterrows():
#     t = row['Time']
for t in times:
    
    init_df = sim_df[sim_df['Time']==t]
    vxs = init_df['x'].tolist()
    vys = init_df['y'].tolist()
    ids = init_df['Car'].tolist()
    if count % 5 == 0:
        print('t', t, 'count', count)
        plt.figure()
        plt.plot(vxs, vys, '.')
        for iid in range(len(ids)):
            plt.text(vxs[iid], vys[iid], ids[iid])
        plt.axis('equal')
        # plt.savefig(f'{plot_save_path}/{count:02d}.jpg')
        # if count % 10 == 0:
        #     print('Saved to', f'{plot_save_path}/{count:02d}.jpg')

    count += 1

    if count > 100:
        break

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

t=6
vehicle_list = get_veh_list(t)
vxs = [v.location.x for v in vehicle_list]
vys = [v.location.y for v in vehicle_list]
plt.plot(vxs, vys, '.')
plt.axis('equal')

# %%
# img_size = (round(maxx)+1, round(maxy)+1, 3)
img_size = (80,80, 3)
img_resize = (200, 200)
# img_resize = img_size
img = np.zeros(img_size)
# for x, y in zip(sim_df['x'], sim_df['y']):
for x, y in zip(xs, ys):
    img[int(x+to_zero), int(y+to_zero), :] = 255
img = img.astype(np.uint8)
img = cv2.resize(img, img_resize)
ksize = (15, 15) 
# Using cv2.blur() method  
img = cv2.blur(img, ksize)  
img[img > 0] = 255

plt.imshow(img)
print(img.min(), img.max(), np.unique(img))

# %%
base_dir = './data/inference/ring/'
save_img_path = os.path.join(base_dir,'drivablemap', 'ring-drivablemap.jpg')
plt.imsave(save_img_path, img)
save_img_path = os.path.join(base_dir,'basemap', 'ring-official-map.jpg')
plt.imsave(save_img_path, img)
for filename in ['circle_1_q-map.jpg', 'circle_2_q-map.jpg', 'circle_3_q-map.jpg', 'circle_4_q-map.jpg']:
    save_img_path = os.path.join(base_dir,'ROIs-map','circle', filename)
    plt.imsave(save_img_path, img)

for filename in ['circle_inner_lane-map.jpg','circle_outer_lane-map.jpg'
                 ]:
    save_img_path = os.path.join(base_dir,'ROIs-map','at-circle-lane', filename)
    plt.imsave(save_img_path, img)
print('save_img_path', save_img_path)

# %%
json_text = f'"tl": [{minx}, {maxy}], "bl": [{minx}, {miny}], "tr": [{maxx}, {maxy}], "br": [{maxx}, {miny}]'
json_text = '{' + json_text + '}'
print(json_text)
buf=10
json_text = f'"tl": [{minx-buf}, {maxy+buf}], "bl": [{minx-buf}, {miny+buf}], "tr": [{maxx+buf}, {maxy+buf}], "br": [{maxx+buf}, {miny-buf}]'
json_text = '{' + json_text + '}'
print(json_text)
json_text = f'"tl": [{0.0}, {img_size[1]}], "bl": [{0.0}, {0.0}], "tr": [{img_size[0]}, {img_size[1]}], "br": [{img_size[0]}, {0.0}]'
json_text = '{' + json_text + '}'
print(json_text)

# %%
empty_img = np.zeros(img_resize)
save_img_path = os.path.join(base_dir,'ROIs-map', 'ring-sim-remove-vehicle-area-map.jpg')
plt.imsave(save_img_path, empty_img)
print('save_img_path', save_img_path)

for filename in ['exit_n-map.jpg','exit_e-map.jpg', 'exit_s-map.jpg',
                 'exit_w-map.jpg', 'exit_n_rightturn-map.jpg', 
                 'exit_s_rightturn-map.jpg'
                 ]:
    save_img_path = os.path.join(base_dir,'ROIs-map','exit', filename)
    plt.imsave(save_img_path, empty_img)
for filename in ['entrance_n_1-map.jpg','entrance_n_2-map.jpg', 'entrance_n_rightturn-map.jpg',
                 'entrance_e_1-map.jpg', 'entrance_e_2-map.jpg', 
                 'entrance_s_1-map.jpg', 'entrance_s_2-map.jpg',
                 'entrance_s_rightturn-map.jpg', 
                 'entrance_w_1-map.jpg', 'entrance_w_2-map.jpg'
                 ]:
    save_img_path = os.path.join(base_dir,'ROIs-map','entrance', filename)
    plt.imsave(save_img_path, empty_img)
for filename in ['yielding_n-map.jpg','yielding_e-map.jpg', 'yielding_s-map.jpg',
                 'yielding_w-map.jpg'
                 ]:
    save_img_path = os.path.join(base_dir,'ROIs-map','yielding-area', filename)
    plt.imsave(save_img_path, empty_img)