# %%
import cv2
import glob
import os
import pickle
from pprint import pprint
import matplotlib.pyplot as plt



# %%
path = './data/inference/rounD/simulation_initialization/initial_clips/rounD-09/01'
path = './data/training/behavior_net/AA_rdbt/AA_rdbt-10h-data-local-heading-size-36-18/train/01/02/'
path = './data/training/behavior_net/ring/ring257/train/01/01/'
experiment = 'ring_faster'
experiment = 'ring'
path = f'./data/inference/{experiment}/simulation_initialization/initial_clips/ring-01/01'


def get_vehicle_list(max_timestep=0):
    TIME_BUFF=[]
    datafolder_dirs = sorted(glob.glob(os.path.join(path, '*.pickle')))
    print('datafolder_dirs', datafolder_dirs)
    for step, datafolder_dir in enumerate(datafolder_dirs):
        vehicle_list = pickle.load(open(datafolder_dir, "rb"))
        TIME_BUFF.append(vehicle_list)
        if max_timestep > 0 and step > max_timestep:
            break
    return TIME_BUFF

# %%

TIME_BUFF = get_vehicle_list(max_timestep=1)

# TIME_BUFF=[]
# TIME_BUFF.append(vehicle_list)

print(TIME_BUFF[0])

# %%
# for i in range(len(TIME_BUFF)):
i=0
for j in range(len(TIME_BUFF[i])):
    v = TIME_BUFF[i][j]
    print('v', v)
    print('x, y, id', v.location.x, v.location.y, v.id)

# %%
plot_save_path = f'./data/inference/{experiment}/imgs_plot'
for i in range(len(TIME_BUFF)):
    vehicle_list = TIME_BUFF[i]
    vxs = [v.location.x for v in vehicle_list]
    vys = [v.location.y for v in vehicle_list]
    ids = [str(v.id) for v in vehicle_list]
    plt.figure()
    plt.plot(vxs, vys, '.')
    for iid in range(len(ids)):
        plt.text(vxs[iid], vys[iid], ids[iid])
    plt.axis('equal')
    # plt.savefig(f'{plot_save_path}/{i:02d}.jpg')
    # if i % 10:
    #     print('Saved to', f'{plot_save_path}/{i:02d}.jpg')
# %%


# %%
# # Demo of our trajectory format
# import pickle
# import os
# import glob

# def load_traj(one_video):
#     TIME_BUFF = []
#     for i in range(0, len(one_video)):
#         vehicle_list = pickle.load(open(one_video[i], "rb"))
#         TIME_BUFF.append(vehicle_list)
#     print("Trajectory length: {0} s".format(len(TIME_BUFF) * 0.4))
#     return TIME_BUFF

# # Load all frames of a episode
# one_video = glob.glob(
#     os.path.join(r'./results/inference/AA_rdbt_inference/example/traj_data', '*.pickle'))
# one_sim_TIME_B
# %%

path = './data/inference/rounD/simulation_initialization/gen_veh_states/rounD/'
datafolder_dirs = glob.glob(os.path.join(path, '*.pickle'))
vehicle_list = pickle.load(open(datafolder_dirs[0], "rb"))
vehicle_list.keys()
vehicle_list['n_in1']


# %%

pprint(vars(v))

# %%
print('location', vars(v.location))
print('gt_size', vars(v.gt_size))
print('pixel_bottom_center', vars(v.pixel_bottom_center))
print('poly_box', vars(v.poly_box))
print('rotation', vars(v.rotation))
print('safe_poly_box', vars(v.safe_poly_box))
print('safe_size', vars(v.safe_size))
print('size', vars(v.size))

# %%
path = '../data/inference/rounD/simulation_initialization/gen_veh_states/rounD/'
datafolder_dirs = glob.glob(os.path.join(path, '*.pickle'))
vehicle_list = pickle.load(open(datafolder_dirs[0], "rb"))
vehicle_list

# %%
map_file_dir = '../data/inference/rounD/basemap/rounD-official-map.png'
basemap = cv2.imread(map_file_dir, cv2.IMREAD_COLOR)
print(basemap.shape)

# %%
map_file_dir = '../data/inference/rounD/drivablemap/rounD-drivablemap.jpg'
basemap = cv2.imread(map_file_dir, cv2.IMREAD_COLOR)
print(basemap.shape, basemap.min(), basemap.max())
# %%
map_file_dir = '../data/sumo/ring-drivablemap.jpg'
basemap = cv2.imread(map_file_dir, cv2.IMREAD_COLOR)
print(basemap.shape, basemap.min(), basemap.max())
plt.imshow(basemap)

# %%

