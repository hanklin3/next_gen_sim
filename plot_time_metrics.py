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
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_1000.csv'
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
trajectory_pred = pd.read_csv(path_pred)
trajectory_pred = trajectory_pred[trajectory_pred['Time'] < 100]

######################

#TODO: Plot the time space diagram of the Simulation No 4 and 10. Note that the cars repeatedly appears. Becareful that you need to justify the whether the cars starts a repeated cycle to make the right plot.
fig,axis = plt.subplots(1,2,figsize=(10,5))
colors = cm.rainbow(np.linspace(0, 1, trajectory.loc[trajectory['Simulation No'] == 0]['Car'].max()+1)) # The colors you should use to plot each car

displacement_errors = 0
num_data = 0

iplot = 0
for sim_num in [0]:
    trajs = trajectory.loc[trajectory['Simulation No'] == sim_num]
    trajs_pred = trajectory_pred.loc[trajectory_pred['Simulation No'] == sim_num]

    # Plotting the trajectories
    for car in trajs['Car'].unique():
        car_data = trajs[trajectory['Car'] == car]
        print('car', car, 'len', len(car_data))
        car_data_pred = trajs_pred[trajs_pred['Car'] == car]

        print(car_data['Time'], car_data['Time'].max())
        print(car_data_pred['Time'], car_data_pred['Time'].max())
        
        for i in range(len(car_data) - 1):
            current_point = car_data.iloc[i]
            next_point = car_data.iloc[i + 1]

            time_gt = car_data.iloc[i]['Time']
            time_pred = car_data_pred.iloc[i]['Time']
            x_gt = car_data.iloc[i]['x']
            x_pred = car_data_pred.iloc[i]['x']
            y_gt = car_data.iloc[i]['y']
            y_pred = car_data_pred.iloc[i]['y']

            assert time_gt == time_pred, (time_gt, time_pred)

            displacement_errors = np.sqrt((x_gt-x_pred)**2 - (y_gt-y_pred)**2)
            num_data += 1
            
    # axis[iplot].show()

    iplot += 1

    displacement_errors_norm = displacement_errors/num_data
    print('displacement_errors_norm', displacement_errors_norm)

# fig.savefig(os.path.join(save_path, 'time_space_diagram.png'))
# print('Saved to ', os.path.join(save_path, 'time_space_diagram.png'))