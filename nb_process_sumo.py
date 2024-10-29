# %%
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
# from shapely.geometry import Point
import os
# import shutil
from PIL import Image, ImageDraw 
from vehicle import Vehicle
os.environ['LIBSUMO'] = '0'
from vehicle.utils_vehicle import to_vehicle
import pickle

# %%

base_dir = '/mnt/d/OneDrive - Massachusetts Institute of Technology/Research/sumo/'
base_dir = './data/sumo/'
assert os.path.exists(base_dir)

# experiment='ring_faster'
experiment='ring'
# experiment='ring_larger'
# experiment='ring_w_goals'

sumo_exp = experiment if experiment != 'ring_w_goals' else 'ring'
simulated_file = os.path.join(base_dir, f"{sumo_exp}/out.xml")
# simulated_file = os.path.join(base_dir,f"{experiment}/out.xml")

assert os.path.exists(simulated_file)


# %%
tree = ET.parse(simulated_file)

print(tree)
sim_root = tree.getroot()

print(sim_root)

sim_data = []
for child in sim_root:
    print(child)
    print('child.attrib', child.attrib)
    for child2 in child:
        print(child2)
        if child2.tag == 'vehicle':
            print(child2.attrib)
            rowd = child2.attrib
            rowd['time'] = child.attrib['time']
            sim_data.append(child2.attrib)
            print('child2.attrib', child2.attrib)
sim_df = pd.DataFrame(sim_data)

# %%
sim_df['time'] = sim_df['time'].astype(float)
sim_df['x'] = sim_df['x'].astype(float)
sim_df['y'] = sim_df['y'].astype(float)
sim_df['id'] = sim_df['id'].astype(int)
sim_df['angle'] = sim_df['angle'].astype(float)
sim_df['speed'] = sim_df['speed'].astype(float)
sim_df['lane'] = sim_df['lane'] # str

sim_df

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
# %%

plt.scatter(sim_df['x'],sim_df['y'])
plt.axis('equal')

minx, maxx = min(sim_df['x']), max(sim_df['x'])
miny, maxy = min(sim_df['y']), max(sim_df['y'])
print('minx, maxx', minx, maxx)
print('miny, maxy', miny, maxy)
print('id', np.unique(sim_df['id']))

print('max time', max(sim_df['time']))

# %%

def get_veh_list(t, debug=False):
    init_df = sim_df[sim_df['time']==t]
    ixs = init_df['x'].tolist()
    iys = init_df['y'].tolist()
    ids = init_df['id'].tolist()
    angles = init_df['angle'].tolist()
    speeds = init_df['speed'].tolist()
    lanes = init_df['lane'].tolist()

    v = Vehicle()
    # print(v.location.x, v.location.y)

    # change to 0.4s
    assert len(ixs) > 0

    vehicle_list = []
    for i in range(len(ixs)):
        # v = Vehicle()
        # v.location.x = ixs[i]
        # v.location.y = iys[i]
        # v.id = ids[i]
        # # sumo: north, clockwise
        # # yours: east, counterclockwise
        # # atan2(y, x): 0 corresponds to a point directly on the positive x-axis. 
        # #              Positive values indicate angles above the x-axis (counterclockwise).
        # v.speed_heading_deg = (-angles[i] + 90 ) % 360
        # # yours to sumo: (90-angle) % 360
        # v.speed = speeds[i]

        # factor = 1
        # v.size.length, v.size.width = 3.6*factor, 1.8*factor
        # v.safe_size.length, v.safe_size.width = 3.8*factor, 2.0*factor
        # v.update_poly_box_and_realworld_4_vertices()
        # v.update_safe_poly_box()
        
        v = to_vehicle(ixs[i], iys[i], angles[i], ids[i], speeds[i], road_id=np.nan, lane_id=lanes[i], 
                       lane_index=np.nan, acceleration=np.nan)
        vehicle_list.append(v)

    if not vehicle_list:
        print('t', t)
        print(init_df)

    if debug:
        print('t', t)
        v = vehicle_list[0]
        print('x, y, heading', v.location.x, v.location.y, v.id, v.speed_heading)
        print('poly_box', vars(v.poly_box))
        print('v.size.length, v.size.width', v.size.length, v.size.width)
    return vehicle_list


def init_pickle(vehicle_list):
    path = f'./data/inference/{experiment}/simulation_initialization/gen_veh_states/ring'
    output_file_path = os.path.join(path, 'initial_vehicle_dict.pickle')
    vehicle_dict = {'n_in1': vehicle_list}
    pickle.dump(vehicle_dict, open(output_file_path, "wb"))

# %%
t = 0.0
count = 0
while t < 1000.0:
    t = round(t, 2)

    vehicle_list = get_veh_list(t)
    if t==0:
        init_pickle(vehicle_list)
        # t += 0.4
        # count += 1
        # continue
    
    path = f'./data/inference/{experiment}/simulation_initialization/initial_clips/ring-01/01/'
    output_file_path = os.path.join(path,f"{count:06d}.pickle")
    print('output_file_path', output_file_path)
    pickle.dump(vehicle_list, open(output_file_path, "wb"))

    if t >= 200.0:
        path = f'./data/training/behavior_net/{experiment}/ring257/train/01/01'
        output_file_path = os.path.join(path,f"{count:06d}.pickle")
        print('output_file_path', output_file_path)
        pickle.dump(vehicle_list, open(output_file_path, "wb"))

    if t < 200.0:
        path = f'./data/training/behavior_net/{experiment}/ring257/val/01/01'
        output_file_path = os.path.join(path,f"{count:06d}.pickle")
        print('output_file_path', output_file_path)
        pickle.dump(vehicle_list, open(output_file_path, "wb"))

    if int(t) %100 == 0:
        print('t', t)

    t += 0.4
    count += 1


# %%
t=6
vehicle_list = get_veh_list(t)
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
