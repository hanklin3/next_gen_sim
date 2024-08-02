import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
# %matplotlib inline

experiment = 'groundtruth'
experiment = 'ring_behavior_net_training_0.4_no_jump2'
experiment = 'ring_speed_acc'
# experiment = 'ring_xy_out'
# experiment = 'ring_xy_out__moveToXY'
experiment = 'ring_speed_acc__setAcceleration'
experiment = 'ring_speed_test'

path = f'./results/inference/behavior_net/{experiment}/df_traj_1000.csv'
save_path = f'./results/inference/behavior_net/{experiment}'

trajectory = pd.read_csv(path)
trajectory.drop('Unnamed: 0', axis=1, inplace=True)
trajectory['Simulation No'] = trajectory['Simulation No'].astype(int)
trajectory['Car'] = trajectory['Car'].astype(int)
# Ensure the trajectory position starts from zeroes in your plot
for i in range(trajectory['Simulation No'].max()):
    if np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Position'])<0:
        trajectory.loc[trajectory['Simulation No'] == i+1,'Position'] += np.abs(np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Position']))
    elif np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Position'])>0:
        trajectory.loc[trajectory['Simulation No'] == i+1,'Position'] -= np.abs(np.min(trajectory.loc[trajectory['Simulation No'] == i+1,'Position']))
trajectory

trajectory = trajectory[trajectory['Time'] < 100]

#####################

#TODO: Plot the time space diagram of the Simulation No 4 and 10. Note that the cars repeatedly appears. Becareful that you need to justify the whether the cars starts a repeated cycle to make the right plot.
fig,axis = plt.subplots(1,2,figsize=(10,5))
colors = cm.rainbow(np.linspace(0, 1, trajectory.loc[trajectory['Simulation No'] == 0]['Car'].max()+1)) # The colors you should use to plot each car

iplot = 0
for sim_num in [0]:
    trajs = trajectory.loc[trajectory['Simulation No'] == sim_num]

    # Plotting the trajectories
    for car in trajs['Car'].unique():
        car_data = trajs[trajectory['Car'] == car]
        print('car', car, 'len', len(car_data))
        prev_position = -np.inf  # Initialize with a value that's always smaller
        for i in range(len(car_data) - 1):
            current_point = car_data.iloc[i]
            next_point = car_data.iloc[i + 1]
            
            # Only draw a line to the next point if the position is greater than or equal to the previous one
            if next_point['Position'] >= current_point['Position']:
                axis[iplot].plot([current_point['Time'], next_point['Time']], 
                        [current_point['Position'], next_point['Position']], 
                        color=colors[car])
            else:
                axis[iplot].scatter([current_point['Time']], [current_point['Position']], color=colors[car], s=1)  # Plot only current point
            
        # Plotting the last point in the series
        axis[iplot].scatter([car_data.iloc[-1]['Time']], [car_data.iloc[-1]['Position']], color=colors[car], s=1)

    axis[iplot].legend(loc=1,bbox_to_anchor=(1.15, 1),borderaxespad=0.)
    axis[iplot].set_xlabel('Time')
    axis[iplot].set_ylabel('Position')
    axis[iplot].set_title(f'Sim. No. {sim_num} Time-Space Diagram')
    axis[iplot].grid(True)
    # axis[iplot].show()

    iplot += 1

fig.savefig(os.path.join(save_path, 'time_space_diagram.png'))
print('Saved to ', os.path.join(save_path, 'time_space_diagram.png'))