import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
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
# experiment = 'ring_speed_close_firstNoMatch_08-22-24'
# experiment = 'ring_speed_open_8-20'

save_path = f'./results/inference/behavior_net/{experiment}'

### log vs model close-loop
path = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_gt_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_close_1000.csv'
# path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_1000.csv'
# path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_1000.csv'
exp_type = 'close-loop'

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
traj_gt = traj_gt[traj_gt['Time'] >= time_gt]
traj_pred = traj_pred[traj_pred['Time'] >= time_gt]

traj_list = [traj_gt, traj_pred]
title_list = ['groundtruth', 'model']
#####################

#TODO: Plot the time space diagram of the Simulation No 4 and 10. Note that the cars repeatedly appears. Becareful that you need to justify the whether the cars starts a repeated cycle to make the right plot.
fig,axis = plt.subplots(1,2,figsize=(11,5))  # use axis[iplot] instead of just 'axis'
# fig,axis = plt.subplots(1,1,figsize=(6,5))
# colors = cm.rainbow(np.linspace(0, 1, trajectory.loc[trajectory['Simulation No'] == 0]['Car'].max()+1)) # The colors you should use to plot each car
# colors = cm.rainbow(np.linspace(0, 1, int(trajectory.loc[trajectory['Simulation No'] == 0]['Speed'].max()+1)))
# colors = cm.rainbow(np.linspace(0, trajectory.loc[trajectory['Simulation No'] == 0]['Speed'].max()+1), 100)
data_speed = traj_gt.loc[traj_gt['Simulation No'] == 0]['Speed']
norm = Normalize(vmin=min(data_speed), vmax=max(data_speed))
# norm = Normalize(vmin=0., vmax=9.0)
print('data_speed min max', data_speed.min(), data_speed.max())
assert data_speed.min() > 0.0, data_speed.min()
assert data_speed.max() < 9.0, data_speed.max()
colors = cm.rainbow(norm(data_speed))
print('colors', len(colors))

iplot = 0
# for sim_num in [0]:
for trajectory in traj_list:
    sim_num = 0
    trajs = trajectory.loc[trajectory['Simulation No'] == sim_num]

    # Plotting the trajectories
    for car in trajs['Car'][trajs['Car']!=-1].unique():

        car_data = trajs[trajectory['Car'] == car]
        print('car', car, 'len', len(car_data), )
        print(car_data)
        prev_position = -np.inf  # Initialize with a value that's always smaller
        
        for i in range(len(car_data) - 1):
            current_point = car_data.iloc[i]
            next_point = car_data.iloc[i + 1]
            speed = car_data.iloc[i]['Speed']

            # Normalize the speed to find its corresponding color
            index = (np.abs(data_speed - speed)).argmin()
            color = colors[index]
            
            # Only draw a line to the next point if the position is greater than or equal to the previous one
            if next_point['Position'] >= current_point['Position']:
                axis[iplot].plot([current_point['Time'], next_point['Time']], 
                        [current_point['Position'], next_point['Position']], 
                        color=color
                        )
            else:
                axis[iplot].scatter([current_point['Time']], [current_point['Position']], color=color, s=1)  # Plot only current point
            
        # Plotting the last point in the series
        # color = colors[car]
        last_speed = car_data.iloc[-1]['Speed']
        last_color = colors[(np.abs(data_speed - last_speed)).argmin()]
        axis[iplot].scatter([car_data.iloc[-1]['Time']], [car_data.iloc[-1]['Position']], color=last_color, s=1)

    axis[iplot].legend(loc=1,bbox_to_anchor=(1.15, 1),borderaxespad=0.)
    axis[iplot].set_xlabel('Time')
    axis[iplot].set_ylabel('Position')
    axis[iplot].set_title(f'{title_list[iplot]} Time-Space Diagram')
    axis[iplot].grid(True)
    # axis.show()

    iplot += 1

# Add a colorbar to the figure
sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=norm)
sm.set_array([])  # Only needed for the colorbar
fig.colorbar(sm, ax=axis, label='Speed')
# fig.colorbar(sm, ax=axis.ravel().tolist(), label='Speed') # for multiple axis

save_pgn = os.path.join(save_path, f'time_space_diagram_{time_gt}.png')
fig.savefig(save_pgn)
print('Saved to ', save_pgn)