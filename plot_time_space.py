# %%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
# %matplotlib inline

experiment = 'groundtruth'
experiment = 'ring_behavior_net_training_0.4_no_jump2'
experiment = 'ring_speed_acc'
# experiment = 'ring_xy_out'
# experiment = 'ring_xy_out__moveToXY'
experiment = 'ring_speed_acc__setAcceleration'
experiment = 'ring_xy_out2'
experiment = 'ring_speed'
experiment = 'ring_speed_acc_fixed_08-10-24'
experiment = 'ring_position_fixed_08-09-24'
experiment = 'ring_speed_close_firstNoMatch_08-22-24'
experiment = 'ring_position_test'
experiment = 'ring_position_close_loop_8-20'
experiment = 'ring_position_fixed_08-09-24__dxdy'
experiment = 'ring_speed_close_8-30'
experiment = 'ring_position_close_multiConf_libsumo_9-16__1_speedMode0'
# experiment = 'ring_position_close_multi_configs_9-05__3'
# experiment = 'ring_position_close_multiConf_libsumo_9-16__3'
# experiment = 'ring_position_close_multiConf_libsumo_9-16__1'
# experiment = 'ring_speed_close_firstNoMatch_08-22-24'
# experiment = 'ring_speed_open_8-20'
experiment = 'ring_position_close_setSpeedMode_9-18'
experiment = 'ring_position_close_setSpeedMode_9-18__refine_angle_9-22'
# experiment = 'ring_position_close_setSpeedMode_9-18__refine_fix_angle_9-22'
# experiment = 'ring_speed_close_setSpeedMode_9-18'

save_path = f'./results/inference/behavior_net/{experiment}'

### log vs model close-loop
path = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_gt_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_close_1000.csv'
# path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_1000.csv'
# path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_1000.csv'
title_list = ['env_no_set (gt log)', 'env_set_by_model (pred)']
exp_name = 'envSUMO_gt_vs_envSUMO_close'

# model-env_set_by_model vs sumo control output
path = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_close_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_1000.csv'
title_list = ['env_set_by_model (pred)', 'model_control_output']
exp_name = 'envSUMO_close_vs_model_pred'

#  sumo pred (gt) vs model pred
path = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_sumoPRED_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_1000.csv'
title_list = ['SUMO_control_output (oracle)', 'model_control_output']
exp_name = 'sumo_pred_gt_vs_model_pred'

# #  sumo pred (gt) vs model pred
# path = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_gt_1000.csv'
# path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_sumoPRED_1000.csv'
# title_list = ['env_no_set (gt log)', 'SUMO_control_output (oracle)']
# exp_name = 'envSUMO_gt_vs_sumo_pred_gt'

# model-env_set_by_model vs sumo pred (gt)
path = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_close_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_sumoPRED_1000.csv'
title_list = ['SUMO sim env', 'Perfect model']
exp_name = 'envSUMO_close_vs_sumo_pred_gt'


def get_traj(path):
    trajectory = pd.read_csv(path)
    print(trajectory)
    trajectory.drop('Unnamed: 0', axis=1, inplace=True)
    trajectory['Simulation No'] = trajectory['Simulation No'].fillna(-1)
    trajectory['Simulation No'] = trajectory['Simulation No'].astype(int)
    trajectory['Car'] = trajectory['Car'].fillna(-1)
    trajectory['Car'] = trajectory['Car'].astype(int)
    # Ensure the trajectory position starts from zeroes in your plot
    for i in range(trajectory['Simulation No'].max()):
        if np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Position'])<0:
            trajectory.loc[trajectory['Simulation No'] == i+1,'Position'] += np.abs(np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Position']))
        elif np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Position'])>0:
            trajectory.loc[trajectory['Simulation No'] == i+1,'Position'] -= np.abs(np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Position']))
    return trajectory

traj_gt = get_traj(path)
traj_pred = get_traj(path_pred)

time_gt = 300
# time_gt = 0
traj_gt = traj_gt[traj_gt['Time'] >= time_gt]
traj_pred = traj_pred[traj_pred['Time'] >= time_gt]
# time_gt = 'all'

traj_gt = traj_gt[traj_gt['Time'] <= time_gt + 100]
traj_pred = traj_pred[traj_pred['Time'] <= time_gt + 100]

traj_list = [traj_gt, traj_pred]

#####################
# %%
#TODO: Plot the time space diagram of the Simulation No 4 and 10. Note that the cars repeatedly appears. Becareful that you need to justify the whether the cars starts a repeated cycle to make the right plot.

# fig,axis = plt.subplots(1,1,figsize=(6,5))
# colors = cm.rainbow(np.linspace(0, 1, trajectory.loc[trajectory['Simulation No'] == 0]['Car'].max()+1)) # The colors you should use to plot each car
# colors = cm.rainbow(np.linspace(0, 1, int(trajectory.loc[trajectory['Simulation No'] == 0]['Speed'].max()+1)))
# colors = cm.rainbow(np.linspace(0, trajectory.loc[trajectory['Simulation No'] == 0]['Speed'].max()+1), 100)

# data_speed = traj_gt.loc[traj_gt['Simulation No'] == 0]['Speed']
# norm = Normalize(vmin=min(data_speed), vmax=max(data_speed))
# norm = Normalize(vmin=0., vmax=9.0)
# print('data_speed min max', data_speed.min(), data_speed.max())
# assert data_speed.min() >= 0.0, data_speed.min()
# assert data_speed.max() < 9.0, data_speed.max()

# data_speed_pred = traj_pred.loc[traj_gt['Simulation No'] == 0]['Speed']
# data_speed = data_speed_pred
# colors = cm.rainbow(norm(data_speed))
# print('colors', len(colors))

# print('norm(data_speed)', norm(data_speed))
# assert False

###################################################################


# # %%
# ##############################
# # Create a figure with two subplots
# fig, axis = plt.subplots(1, 2, figsize=(11, 5))  # axis is now an array of subplots

# # #########################################################
# # Set up color normalization based on the full range of calculated speeds
# all_speeds = []

# # Loop to gather all speeds for normalization, but only for increasing positions
# for trajectory in traj_list:
#     sim_num = 0
#     trajs = trajectory.loc[trajectory['Simulation No'] == sim_num]

#     # Loop through each car's trajectory
#     for car in trajs['Car'][trajs['Car'] != -1].unique():
#         car_data = trajs[trajectory['Car'] == car]
        
#         for i in range(len(car_data) - 1):
#             current_point = car_data.iloc[i]
#             next_point = car_data.iloc[i + 1]
            
#             # Only calculate speed if the position is increasing
#             if next_point['Position'] > current_point['Position']:
#                 # Calculate speed (delta_position / delta_time)
#                 delta_position = next_point['Position'] - current_point['Position']
#                 delta_time = next_point['Time'] - current_point['Time']
#                 speed = delta_position / delta_time if delta_time != 0 else 0  # Avoid division by zero
#                 all_speeds.append(speed)
#             # If position decreases or resets, do nothing (skip this point)

# # Normalize speeds
# data_speed = np.array(all_speeds)
# if len(data_speed) > 0:  # Ensure there are speeds to normalize
#     speed_min, speed_max = np.min(data_speed), np.max(data_speed)
# else:
#     speed_min, speed_max = 0, 1  # Defaults if no valid speeds
# norm = mcolors.Normalize(vmin=speed_min, vmax=speed_max)
# colors = cm.rainbow(norm(data_speed))
# print('colors', len(colors))
# ##########################################################



# iplot = 0
# for trajectory in traj_list:
#     sim_num = 0
#     trajs = trajectory.loc[trajectory['Simulation No'] == sim_num]

#     # Plotting the trajectories
#     for car in trajs['Car'][trajs['Car'] != -1].unique():

#         car_data = trajs[trajectory['Car'] == car]
#         print('car', car, 'len', len(car_data))
#         prev_position = -np.inf  # Initialize with a value that's always smaller
        
#         for i in range(len(car_data) - 1):
#             current_point = car_data.iloc[i]
#             next_point = car_data.iloc[i + 1]

#             # Only process if the position is increasing
#             if next_point['Position'] > current_point['Position']:
#                 # Calculate speed as the difference in position divided by time interval
#                 delta_position = next_point['Position'] - current_point['Position']
#                 delta_time = next_point['Time'] - current_point['Time']
#                 speed = delta_position / delta_time if delta_time != 0 else 0  # Avoid division by zero
              
#                 # Normalize the speed to find its corresponding color
#                 index = (np.abs(data_speed - speed)).argmin()
#                 color = colors[index]
                
#                 # Draw the line for increasing position
#                 axis[iplot].plot([current_point['Time'], next_point['Time']], 
#                                  [current_point['Position'], next_point['Position']], 
#                                  color=color)
#             else:
#                 # Reset or skip this point if position decreases
#                 continue
        
#         # Plot the last point in the series (if position has been increasing)
#         if car_data.iloc[-1]['Position'] > car_data.iloc[-2]['Position']:
#             last_speed = car_data.iloc[-1]['Speed']
#             last_color = colors[(np.abs(data_speed - last_speed)).argmin()]
#             axis[iplot].scatter([car_data.iloc[-1]['Time']], [car_data.iloc[-1]['Position']], color=last_color, s=1)

#     axis[iplot].legend(loc=1, bbox_to_anchor=(1.15, 1), borderaxespad=0.)
#     axis[iplot].set_xlabel('Time')
#     axis[iplot].set_ylabel('Position')
#     axis[iplot].set_title(f'{title_list[iplot]}')
#     axis[iplot].grid(True)

#     iplot += 1

# # Add a colorbar to the figure
# sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=norm)
# sm.set_array([])  # Only needed for the colorbar

# # For multiple axes, you need to pass all the axes to the colorbar
# fig.colorbar(sm, ax=axis.ravel().tolist(), label='Speed')

# # Save the figure
# save_png = os.path.join(save_path, f'time_space_diagram_{time_gt}.png')
# fig.savefig(save_png)
# print('Saved to ', save_png)



# %%


all_speeds = []

# Loop to gather all speeds for normalization
for trajectory in traj_list:
    sim_num = 0
    trajs = trajectory.loc[trajectory['Simulation No'] == sim_num]

    xs = trajs['x']
    ys = trajs['y']
    headings = trajs['Heading']

    print(xs)
    print(ys)
    
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    headings_min, headings_max = np.min(headings), np.max(headings)
    print(x_min, x_max)
    print(y_min, y_max)
    print(headings_min, headings_max)

# %%                
# Create a figure with two subplots (or more if needed)
fig, axis = plt.subplots(1, 2, figsize=(11, 5))  # axis is now an array of subplots

# #########################################################
# Set up color normalization based on the full range of calculated speeds
all_speeds = []

# Loop to gather all speeds for normalization
for trajectory in traj_list:
    sim_num = 0
    trajs = trajectory.loc[trajectory['Simulation No'] == sim_num]

    # Loop through each car's trajectory
    for car in trajs['Car'][trajs['Car'] != -1].unique():
        car_data = trajs[trajectory['Car'] == car]
        
        for i in range(len(car_data) - 1):
            current_point = car_data.iloc[i]
            next_point = car_data.iloc[i + 1]
            
            # Calculate speed (delta_position / delta_time)
            if next_point['Position'] >= current_point['Position']:
                delta_position = next_point['Position'] - current_point['Position']
                delta_time = next_point['Time'] - current_point['Time']
                speed = delta_position / delta_time if delta_time != 0 else 0  # Avoid division by zero
                all_speeds.append(speed)

# Normalize speeds
data_speed = np.array(all_speeds)
speed_min, speed_max = np.min(all_speeds), np.max(all_speeds)
assert speed_min >= 0, speed_min
assert speed_max <= 3.75, speed_max
print('speed_min', speed_min, 'speed_max', speed_max)
speed_min, speed_max = 0.0, 3.75
norm = mcolors.Normalize(vmin=speed_min, vmax=speed_max)
colors = cm.rainbow(norm(data_speed))
print('colors', len(colors))

###########################################################

iplot = 0
for trajectory in traj_list:
    sim_num = 0
    trajs = trajectory.loc[trajectory['Simulation No'] == sim_num]

    # Plotting the trajectories
    for car in trajs['Car'][trajs['Car'] != -1].unique():

        car_data = trajs[trajectory['Car'] == car]
        print('car', car, 'len', len(car_data))
        prev_position = -np.inf  # Initialize with a value that's always smaller
        
        for i in range(len(car_data) - 1):
            current_point = car_data.iloc[i]
            next_point = car_data.iloc[i + 1]

            if next_point['Position'] >= current_point['Position']:
                # Calculate speed as the difference in position divided by time interval
                delta_position = next_point['Position'] - current_point['Position']
                delta_time = next_point['Time'] - current_point['Time']
                speed = delta_position / delta_time if delta_time != 0 else 0  # Avoid division by zero
            
                # Normalize the speed to find its corresponding color
                index = (np.abs(data_speed - speed)).argmin()
                color = colors[index]
            
            # Only draw a line to the next point if the position is greater than or equal to the previous one
            if next_point['Position'] >= current_point['Position']:
                axis[iplot].plot([current_point['Time'], next_point['Time']], 
                                 [current_point['Position'], next_point['Position']], 
                                 color=color)
            else:
                axis[iplot].scatter([current_point['Time']], [current_point['Position']], color=color, s=1)  # Plot only current point
            
        # Plotting the last point in the series
        last_point = car_data.iloc[-1]
        axis[iplot].scatter([last_point['Time']], [last_point['Position']], color=colors[(np.abs(data_speed - speed)).argmin()], s=1)

    axis[iplot].legend(loc=1, bbox_to_anchor=(1.15, 1), borderaxespad=0.)
    axis[iplot].set_xlabel('Time')
    axis[iplot].set_ylabel('Position')
    axis[iplot].set_title(f'{title_list[iplot]}')
    axis[iplot].grid(True)

    iplot += 1

# Add a colorbar to the figure
sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=norm)
sm.set_array([])  # Only needed for the colorbar

# For multiple axes, you need to pass all the axes to the colorbar
fig.colorbar(sm, ax=axis.ravel().tolist(), label='Speed')

# Save the figure
save_png = os.path.join(save_path, f'time_space_diagram_{time_gt}_{exp_name}.png')
fig.savefig(save_png)
print('Saved to ', save_png)