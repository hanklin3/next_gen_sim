import numpy as np
import pickle
import os
import glob
import random
import torch

import logging
logging.basicConfig(level=logging.WARNING)
import traci

from torch.utils.data import Dataset, DataLoader
# from trajectory_pool import TrajectoryPool

# import simulation_modeling.utils as utils
from . import utils
from utils import time_buff_to_traj_pool, to_vehicle


class MTLTrajectoryPredictionDataset(Dataset):
    """
    Pytorch Dataset Loader...
    """

    def __init__(self, history_length, pred_length, max_num_vehicles, is_train, dataset='ring',
                 sumo_cmd = ['sumo', '-c', 'data/sumo/ring/circles.sumocfg']):
        self.history_length = history_length
        self.pred_length = pred_length
        self.max_num_vehicles = max_num_vehicles
        self.is_train = is_train
        self.dataset = dataset
        self.max_length = 1000
        self.TIME_BUFF = []
        self.sumo_cmd = sumo_cmd
        
        print('sumo_cmd', sumo_cmd)


        if self.dataset == 'rounD' or self.dataset == 'AA_rdbt' or self.dataset == 'ring':
            split = 'train' if is_train else 'val'
            # traci.start(sumo_cmd)
        else:
            raise NotImplementedError( 'Wrong dataset name %s (choose one from [AA_rdbt, rounD,...])' % self.dataset)

    def __len__(self):
        if self.dataset == 'rounD' or self.dataset == 'AA_rdbt' or self.dataset == 'ring':
            return self.max_length - (self.history_length+self.pred_length)
        else:
            raise NotImplementedError( 'Wrong dataset name %s (choose one from [AA_rdbt, rounD,...])' % self.dataset)

    def __getitem__(self, idx):

        # if self.dataset == 'rounD' or self.dataset == 'AA_rdbt' or self.dataset == 'ring':
        #     subfolder_id = random.choices(range(len(self.each_subfolder_size)), weights=self.subfolder_data_proportion)[0]
        #     subsubfolder_id = random.choices(range(len(self.traj_dirs[subfolder_id])), weights=self.subsubfolder_data_proportion[subfolder_id])[0]
        #     datafolder_dirs = self.traj_dirs[subfolder_id][subsubfolder_id]

        #     idx_start = self.history_length + 1
        #     idx_end = len(datafolder_dirs) - self.pred_length - 1
        #     idx = random.randint(idx_start, idx_end)
        # else:
        #     raise NotImplementedError( 'Wrong dataset name %s (choose one from [AA_rdbt, rounD,...])' % self.dataset)

        # traj_pool = self.fill_in_traj_pool(t0=idx, datafolder_dirs=datafolder_dirs)
        # buff_lat, buff_lon, buff_cos_heading, buff_sin_heading = traj_pool.flatten_trajectory(
        #     max_num_vehicles=self.max_num_vehicles, time_length=self.history_length+self.pred_length)

        # input_matrix, gt_matrix = self.make_training_data_pair(buff_lat, buff_lon, buff_cos_heading, buff_sin_heading)

        # input_matrix = torch.tensor(input_matrix, dtype=torch.float32)
        # gt_matrix = torch.tensor(gt_matrix, dtype=torch.float32)
        # data = {'input': input_matrix, 'gt': gt_matrix}
        
        traci.start(self.sumo_cmd)

        step = 0
        while step < self.max_length:
            # print(step)

            traci.simulationStep()
            
            step += 1
            
            if step < idx:
                continue

            car_list = traci.vehicle.getIDList()
            vehicle_list = []
            
            for car_id in car_list:
                x,y = traci.vehicle.getPosition(car_id)
                angle_deg = traci.vehicle.getAngle(car_id)
                speed = traci.vehicle.getSpeed(car_id)
                acceleration = traci.vehicle.getAcceleration(car_id)
                road_id = traci.vehicle.getRoadID(car_id)
                lane_id = traci.vehicle.getLaneID(car_id)
                lane_index = traci.vehicle.getLaneIndex(car_id)
                # speed = traci.getSpeed(car_id)
                # print(car_id, '(', x, y, ')', angle_deg, road_id, lane_index)
                # print('road_id, lane_id', road_id, lane_index, type(road_id), type(lane_index))

                # lat, lon, cos_heading, sin_heading = converter(x, y, angle_deg)
                # print('lat, lon, cos_heading, sin_heading', 
                #       lat, lon, cos_heading, sin_heading)
                vehicle_list.append(to_vehicle(x, y, angle_deg, car_id, speed, road_id, 
                                               lane_id, lane_index, acceleration))
                
            self.TIME_BUFF.append(vehicle_list)
                
            if len(self.TIME_BUFF) < self.history_length+self.pred_length:
                continue
            
            assert len(self.TIME_BUFF) >= self.history_length + self.pred_length, len(self.TIME_BUFF)
            traj_pool = time_buff_to_traj_pool(self.TIME_BUFF)
            
            buff_lat, buff_lon, buff_cos_heading, buff_sin_heading, \
                buff_vid, buff_speed, buff_acc, buff_road_id, buff_lane_id, buff_lane_index = \
                traj_pool.flatten_trajectory(
                time_length=self.history_length+self.pred_length, max_num_vehicles=self.max_num_vehicles, output_vid=True)

            input_matrix, gt_matrix = self.make_training_data_pair(buff_lat, buff_lon, buff_cos_heading, buff_sin_heading, buff_speed, buff_acc)

            input_matrix = torch.tensor(input_matrix, dtype=torch.float32)
            gt_matrix = torch.tensor(gt_matrix, dtype=torch.float32)
            data = {'input': input_matrix, 'gt': gt_matrix}
            
            break
            
        traci.close()

        return data

    # def fill_in_traj_pool(self, t0, datafolder_dirs):
    #     # read frames within a time interval
    #     traj_pool = TrajectoryPool()
    #     for i in range(t0-self.history_length+1, t0+self.pred_length+1):
    #         vehicle_list = pickle.load(open(datafolder_dirs[i], "rb"))
    #         traj_pool.update(vehicle_list)
    #     return traj_pool

    def make_training_data_pair(self, buff_lat, buff_lon, buff_cos_heading, buff_sin_heading, buff_speed, buff_acc):

        buff_lat_in = buff_lat[:, 0:self.history_length]
        buff_lat_out = buff_lat[:, self.history_length:]
        buff_lon_in = buff_lon[:, 0:self.history_length]
        buff_lon_out = buff_lon[:, self.history_length:]
        buff_cos_heading_in = buff_cos_heading[:, 0:self.history_length]
        buff_cos_heading_out = buff_cos_heading[:, self.history_length:]
        buff_sin_heading_in = buff_sin_heading[:, 0:self.history_length]
        buff_sin_heading_out = buff_sin_heading[:, self.history_length:]
        buff_speed_out = buff_speed[:, self.history_length:]
        buff_acc_out = buff_acc[:, self.history_length:]

        buff_lat_in = utils.nan_intep_2d(buff_lat_in, axis=1)
        buff_lon_in = utils.nan_intep_2d(buff_lon_in, axis=1)
        buff_cos_heading_in = utils.nan_intep_2d(buff_cos_heading_in, axis=1)
        buff_sin_heading_in = utils.nan_intep_2d(buff_sin_heading_in, axis=1)

        input_matrix = np.concatenate([buff_lat_in, buff_lon_in, buff_cos_heading_in, buff_sin_heading_in], axis=1)
        gt_matrix = np.concatenate([buff_speed_out, buff_acc_out, buff_cos_heading_out, buff_sin_heading_out], axis=1)

        # # mask-out those output traj whose input is nan
        gt_matrix[np.isnan(input_matrix).sum(1) > 0, :] = np.nan

        # shuffle the order of input tokens
        input_matrix, gt_matrix = self._shuffle_tokens(input_matrix, gt_matrix)

        # data augmentation
        input_matrix = self._data_augmentation(input_matrix, pos_scale=0.05, heading_scale=0.001)

        return input_matrix, gt_matrix

    @staticmethod
    def _shuffle_tokens(input_matrix, gt_matrix):

        max_num_vehicles = input_matrix.shape[0]
        shuffle_id = list(range(0, max_num_vehicles))
        random.shuffle(shuffle_id)

        input_matrix = input_matrix[shuffle_id, :]
        gt_matrix = gt_matrix[shuffle_id, :]

        return input_matrix, gt_matrix

    def _data_augmentation(self, input_matrix, pos_scale=1.0, heading_scale=1.0):
        pos_mask, heading_mask = np.ones_like(input_matrix), np.ones_like(input_matrix)
        pos_mask[:, 2*self.history_length:] = 0
        heading_mask[:, :2*self.history_length] = 0

        pos_rand = pos_scale * utils.randn_like(input_matrix) * pos_mask
        heading_rand = heading_scale * utils.randn_like(input_matrix) * heading_mask

        augmented_input = input_matrix + pos_rand + heading_rand

        return augmented_input


def get_loaders(configs, sumo_cmd):

    if configs["dataset"] == 'AA_rdbt' or configs["dataset"] == 'rounD' or configs["dataset"] == 'ring':
        training_set = MTLTrajectoryPredictionDataset(history_length=configs["history_length"], pred_length=configs["pred_length"],
                                                      max_num_vehicles=configs["max_num_vehicles"], is_train=True, dataset=configs["dataset"], sumo_cmd=sumo_cmd)
        val_set = MTLTrajectoryPredictionDataset(history_length=configs["history_length"], pred_length=configs["pred_length"],
                                                 max_num_vehicles=configs["max_num_vehicles"], is_train=False, dataset=configs["dataset"], sumo_cmd=sumo_cmd)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [AA_rdbt, rounD,...])'
            % configs.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=configs["batch_size"],
                                 shuffle=True, num_workers=configs["dataloader_num_workers"])
                   for x in ['train', 'val']}

    return dataloaders

