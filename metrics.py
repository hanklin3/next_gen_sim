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
experiment = 'ring_position_close_loop_8-20'   # pos_rmse 0.0, minADE = 0.0
experiment = 'ring_position_fixed_08-09-24__dxdy'  # pos_rmse 41.437289445872075, minADE 0.10204540416577479
# experiment = 'ring_speed_close_firstNoMatch_08-22-24'
# experiment = 'ring_speed_open_8-20'
experiment = 'ring_position_close_multi_configs_9-05__1' # pos_rmse 0.0 minADE 0.0
experiment = 'ring_position_close_multi_configs_9-05__2' # pos_rmse 0.0 minADE 0.0
experiment = 'ring_position_close_multi_configs_9-05__3' # pos_rmse 0.0 minADE 0.0
experiment = 'ring_position_close_setSpeedMode_9-18__refine_angle_9-22' # pos_rmse 5.49, minADE 0.184
                                                                        # pos_rmse 6.9795 , minADE 0.1738 
# experiment = 'ring_position_close_setSpeedMode_9-18__refine_fix_angle_9-22' # pos_rmse 9.8909, minADE 0.1294
# experiment = 'ring_speed_close_setSpeedMode_9-18'

save_path = f'./results/inference/behavior_net/{experiment}'

### log vs model close-loop
path = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_gt_1000.csv'
path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_sumo_close_1000.csv'
# path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_pred_close_loop_1000.csv'
# path_pred = f'./results/inference/behavior_net/{experiment}/df_traj_1000.csv'
# exp_type = 'close-loop'

def get_traj(path):
    trajectory = pd.read_csv(path)
    # print(trajectory)
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

time_gt = 0
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

def get_xy(df):
    num_samples = df.shape[0]
    xy = np.zeros((num_samples, 2))
    xy[:, 0] = df['x']
    xy[:, 1] = df['y']

    return xy

# for sim_num in [0]:

def get_rmse(traj_gt, traj_pred):
    xy_se_all = None

    # Plotting the trajectories
    for car in traj_gt['Car'][traj_gt['Car']!=-1].unique():

        df_gt = traj_gt[traj_gt['Car'] == car]
        # print('car', car, 'len', len(df_gt), )
        # print(df_gt)

        df_pred = traj_pred[traj_pred['Car'] == car]
        # print('car', car, 'len', len(df_pred), )
        # print(df_pred)

        
        pred_xy = get_xy(df_pred)  # [T, 2]
        gt_xy = get_xy(df_gt)      # [T, 2]
        # (x1-x2)**2 + (y1-y2)**2
        xy_se = np.square(gt_xy - pred_xy)  # [T, 2]
        xy_se = xy_se.sum(-1) # [T]
        # sqrt((x1-x2)**2 + (y1-y2)**2)**2 == ((x1-x2)**2 + (y1-y2)**2)
        # xy_se = np.square(np.sqrt(xy_se))
        xy_se = xy_se[..., np.newaxis]  # [T, M]
        # print('xy_se', xy_se.shape, xy_se)

        xy_se_all = np.concatenate((xy_se_all, xy_se), axis=-1) if (xy_se_all is not None) else xy_se
    # print('xy_se_all', xy_se_all.shape, xy_se_all)    
    num_sim, num_agents = xy_se_all.shape

    # 1/M * sum_M ((x1-x2)**2 + (y1-y2)**2))
    position_rmse = xy_se_all.sum(-1)/num_agents
    # 1/T * sum_T{ sqrt(1/M * sum_M ((x1-x2)**2 + (y1-y2)**2)) }
    position_rmse = np.sqrt(position_rmse).mean()

    # print('position_rmse', position_rmse)

    return position_rmse



assert get_rmse(traj_gt, traj_gt) == 0.0

pos_rmse = get_rmse(traj_gt, traj_pred)

print('pos_rmse', pos_rmse)



def get_minADE(traj_gt, traj_pred):
    total_sim_time_s = 20
    interval_s = 0.4
    N_rollout = total_sim_time_s / interval_s # 50
    N_rollout = 20

    min_ade_list = []

    # Plotting the trajectories
    for car in traj_gt['Car'][traj_gt['Car']!=-1].unique():

        df_gt = traj_gt[traj_gt['Car'] == car]
        # print('car', car, 'len', len(df_gt), )
        # print(df_gt)

        df_pred = traj_pred[traj_pred['Car'] == car]
        # print('car', car, 'len', len(df_pred), )
        # print(df_pred)

        rows = df_pred.shape[0]
        # print('df_pred.shape', df_pred.shape) # [995, 7] # [rows, cols]
        xy_se_all = []
        for i_start in range(0, rows-N_rollout):
            pred_xy = get_xy(df_pred.iloc[i_start:i_start+N_rollout])  # [N_rollout, 2] #[20, 2]
            gt_xy = get_xy(df_gt.iloc[i_start:i_start+N_rollout])      # [N_rollout, 2]
            # print('pred_xy', pred_xy.shape) #[20, 2]
            # (x1-x2)**2 + (y1-y2)**2
            xy_se = np.square(gt_xy - pred_xy)  # [N_rollout, 2]  # (20,)
            xy_se = xy_se.sum(-1) # [N_rollout]
            # sqrt((x1-x2)**2 + (y1-y2)**2)**2
            # xy_se = np.square(np.sqrt(xy_se))
            # print('xy_se.shape', xy_se.shape)  # (20,)

            # why? become displacement error = L2
            # sqrt( sqrt((x1-x2)**2 + (y1-y2)**2)**2 )
            # L2: sqrt((x1-x2)**2 + (y1-y2)**2)
            xy_se = np.sqrt(xy_se)
            xy_se_all.append(xy_se)

        ade = np.stack(xy_se_all, axis=0) # [num_traj, N_rollout] # (975, 20)
        # print('ade stack', ade.shape) # (975, 20)
        ade = np.mean(ade, axis=1) # [num_traj] # (975,)
        # print('ade mean', ade.shape) # (975,)
        min_ade = np.amin(ade,axis=0) # ()
        # print('min_ade', min_ade.shape, min_ade) # ()
        assert min_ade in ade
        assert np.all( (ade - min_ade) >= 0)

        min_ade_list.append(min_ade)

    # minADE = np.concatenate(min_ade_list, axis=0).mean()
    # print(np.asarray(min_ade_list))
    print('min_ade_list == num_cars', len(min_ade_list))
    assert len(min_ade_list) == len(traj_gt['Car'][traj_gt['Car']!=-1].unique())
    minADE = np.asarray(min_ade_list).mean()
    return minADE

minADE = get_minADE(traj_gt, traj_pred)
print('minADE', minADE)
