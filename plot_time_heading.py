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
exp_name = 'envSUMO_gt_vs_close'

# model-env_set_by_model vs sumo control output
path = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_close_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_1000.csv'
title_list = ['env_set_by_model (closed)', 'model_control_output']
exp_name = 'envSUMO_close_vs_model_pred'

#  sumo pred (gt) vs model pred
path = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_sumoPRED_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_1000.csv'
title_list = ['Perfect model', 'model_control_output']
exp_name = 'sumo_pred_gt_vs_model_pred'

#  sumo pred (gt) vs model pred
path = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_gt_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_sumoPRED_1000.csv'
title_list = ['env_no_set (gt log)', 'Perfect model']
exp_name = 'envSUMO_gt_vs_sumo_pred_gt'

# model-env_set_by_model vs sumo pred (gt)
path = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_close_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_sumoPRED_1000.csv'
title_list = ['env_set_by_model (closed)', 'Perfect model']
exp_name = 'SUMO_sim_vs_perfect_model'

def get_traj(path):
    trajectory = pd.read_csv(path)
    print(trajectory)
    trajectory.drop('Unnamed: 0', axis=1, inplace=True)
    trajectory['Simulation No'] = trajectory['Simulation No'].fillna(-1)
    trajectory['Simulation No'] = trajectory['Simulation No'].astype(int)
    trajectory['Car'] = trajectory['Car'].fillna(-1)
    trajectory['Car'] = trajectory['Car'].astype(int)
    # Ensure the trajectory Heading starts from zeroes in your plot
    for i in range(trajectory['Simulation No'].max()):
        if np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Heading'])<0:
            trajectory.loc[trajectory['Simulation No'] == i+1,'Heading'] += np.abs(np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Heading']))
        elif np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Heading'])>0:
            trajectory.loc[trajectory['Simulation No'] == i+1,'Heading'] -= np.abs(np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Heading']))
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

# Create a figure with two subplots (or more if needed)
fig, axis = plt.subplots(1, 2, figsize=(11, 5))  # axis is now an array of subplots

# #########################################################
# Set up color normalization based on the full range of calculated headings
all_headings = []

# Loop to gather all headings for normalization
for trajectory in traj_list:
    sim_num = 0
    trajs = trajectory.loc[trajectory['Simulation No'] == sim_num]

    # Loop through each car's trajectory
    for car in trajs['Car'][trajs['Car'] != -1].unique():
        car_data = trajs[trajectory['Car'] == car]
        
        for i in range(len(car_data) - 1):
            current_point = car_data.iloc[i]
            next_point = car_data.iloc[i + 1]
            
            # Calculate heading (delta_Heading / delta_time)
        if next_point['Heading'] >= current_point['Heading']:
            delta_Heading = next_point['Heading'] - current_point['Heading']
            delta_time = next_point['Time'] - current_point['Time']
            heading = delta_Heading / delta_time if delta_time != 0 else 0  # Avoid division by zero
            all_headings.append(heading)

# Normalize headings
data_heading = np.array(all_headings)
heading_min, heading_max = np.min(all_headings), np.max(all_headings)
# assert heading_min >= 0, heading_min
# assert heading_max <= 3.75, heading_max
print('heading_min', heading_min, 'heading_max', heading_max)
# heading_min, heading_max = 0.0, 3.75
norm = mcolors.Normalize(vmin=heading_min, vmax=heading_max)
colors = cm.rainbow(norm(data_heading))
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
        prev_Heading = -np.inf  # Initialize with a value that's always smaller
        
        for i in range(len(car_data) - 1):
            current_point = car_data.iloc[i]
            next_point = car_data.iloc[i + 1]

            if next_point['Heading'] >= current_point['Heading']:
                # Calculate heading as the difference in Heading divided by time interval
                delta_Heading = next_point['Heading'] - current_point['Heading']
                delta_time = next_point['Time'] - current_point['Time']
                heading = delta_Heading / delta_time if delta_time != 0 else 0  # Avoid division by zero
            
                # Normalize the heading to find its corresponding color
                index = (np.abs(data_heading - heading)).argmin()
                color = colors[index]
            
            # Only draw a line to the next point if the Heading is greater than or equal to the previous one
            if next_point['Heading'] >= current_point['Heading']:
                axis[iplot].plot([current_point['Time'], next_point['Time']], 
                                 [current_point['Heading'], next_point['Heading']], 
                                 color=color)
            else:
                color = colors[0]
                axis[iplot].scatter([current_point['Time']], [current_point['Heading']], color=color, s=1)  # Plot only current point
            
        # Plotting the last point in the series
        last_point = car_data.iloc[-1]
        axis[iplot].scatter([last_point['Time']], [last_point['Heading']], color=colors[(np.abs(data_heading - heading)).argmin()], s=1)

    axis[iplot].legend(loc=1, bbox_to_anchor=(1.15, 1), borderaxespad=0.)
    axis[iplot].set_xlabel('Time')
    axis[iplot].set_ylabel('Heading')
    axis[iplot].set_title(f'{title_list[iplot]}')
    axis[iplot].grid(True)

    iplot += 1

# Add a colorbar to the figure
sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=norm)
sm.set_array([])  # Only needed for the colorbar

# For multiple axes, you need to pass all the axes to the colorbar
fig.colorbar(sm, ax=axis.ravel().tolist(), label='Heading')

# Save the figure
save_png = os.path.join(save_path, f'time_heading_{time_gt}_{exp_name}.png')
fig.savefig(save_png)
print('Saved to ', save_png)