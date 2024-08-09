import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
# %matplotlib inline

from collections import defaultdict


experiment = 'groundtruth'
experiment = 'ring_behavior_net_training_0.4_no_jump2'
experiment = 'ring_speed_acc'
# experiment = 'ring_xy_out'
# experiment = 'ring_xy_out__moveToXY'
# experiment = 'ring_speed_acc__setAcceleration'
experiment = 'ring_xy_out2'
experiment = 'ring_speed'

### model position vs sumo position (model xy->speed->sumo xy)
path = f'./results/inference/behavior_net/{experiment}/df_traj_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_1000.csv'
exp_type = 'model_vs_controller_xy'

### log vs model close-loop
path = f'./results/inference/behavior_net/{experiment}/df_traj_gt_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_1000.csv'
exp_type = 'close-loop'

### log vs model open-loop
# path = f'./results/inference/behavior_net/{experiment}/df_traj_gt_1000.csv'
# path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_open_loop_1000.csv'
# exp_type = 'open-loop'

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
bar_time_vs_disp_error = defaultdict(list)

iplot = 0
for sim_num in [0]:
    trajs = trajectory.loc[trajectory['Simulation No'] == sim_num]
    trajs_pred = trajectory_pred.loc[trajectory_pred['Simulation No'] == sim_num]

    print('Total cars:', len(trajs['Car'].unique()), trajs['Car'].unique())

    # Plotting the trajectories
    for car in trajs['Car'].unique():
        car_data = trajs[trajectory['Car'] == car]
        print('car', car, 'exist timesteps:', len(car_data))
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
            assert not np.isnan(x_gt), x_gt
            assert not np.isnan(x_pred), x_pred
            displacement_error = np.sqrt((x_gt-x_pred)**2 + (y_gt-y_pred)**2)
            bar_time_vs_disp_error[i].append(displacement_error)
            displacement_errors += displacement_error
            num_data += 1
            
    # axis[iplot].show()

    iplot += 1

    displacement_errors_norm = displacement_errors/num_data

    print('Total cars:', len(trajs['Car'].unique()), trajs['Car'].unique())
    print('displacement_errors', displacement_errors)
    print('displacement_errors_norm', displacement_errors_norm)

    print(i, bar_time_vs_disp_error[i])

    disp_error_avg_over_cars_per_time = np.zeros((len(bar_time_vs_disp_error)))
    for row, disp_errs in bar_time_vs_disp_error.items():
        assert len(disp_errs) > 1, len(disp_errs)
        disp_error_avg_over_cars_per_time[row] = np.mean(disp_errs)

    fig = plt.figure()
    plt.bar(bar_time_vs_disp_error.keys(), disp_error_avg_over_cars_per_time)
    
    plt.xlabel('Timestep')
    plt.ylabel('Avg. Displace Error over all Cars')

    if exp_type == 'model_vs_controller_xy':
        plt.title('Time t+1: Model predicted xy VS Sumo xy from model xy-->speed')
        save_png = os.path.join(save_path, 'avg_disp_err_controller.png')
    elif exp_type == 'close-loop':
        plt.title('Log vs Model close-loop')
        save_png = os.path.join(save_path, 'avg_disp_err_log_vs_close-loop.png')
    elif exp_type == 'open-loop':
        plt.title('Log vs Model open-loop')
        save_png = os.path.join(save_path, 'avg_disp_err_log_vs_open-loop.png')
    fig.savefig(save_png)
    print('Saved to ', save_png)

# fig.savefig(os.path.join(save_path, 'time_space_diagram.png'))
# print('Saved to ', os.path.join(save_path, 'time_space_diagram.png'))