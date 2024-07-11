import numpy as np
import os
import random
import torch
import copy
from itertools import combinations
import math
from . import utils


from behavior_net.networks import define_G, define_safety_mapper

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Predictor(object):
    """
    Model inference. This class provides fundamental methods for model training and checkpoint management.
    """
    
    def __init__(self, model, history_length, pred_length, m_tokens, checkpoint_dir,
                 safety_mapper_ckpt_dir=None, device=device,):

        self.model = model
        self.history_length =history_length
        self.pred_length = pred_length
        self.m_tokens = m_tokens
        self.checkpoint_dir = checkpoint_dir
        self.safety_mapper_ckpt_dir = safety_mapper_ckpt_dir
        self.device = device

        self.net_G = self.initialize_net_G()
        self.net_G.eval()

        if safety_mapper_ckpt_dir is not None:
            self.net_safety_mapper = self.initialize_net_safety_mapper()
            self.net_safety_mapper.eval()
    
    def initialize_net_G(self):

        net_G = define_G(
            model=self.model, input_dim=4*self.history_length,
            output_dim=4*self.pred_length, m_tokens=self.m_tokens).to(self.device)

        if self.checkpoint_dir is None:
            # Initialized traffic_sim during the start of training
            pass
        elif os.path.exists(self.checkpoint_dir):
            # initialize network
            print('initializing networks...from checkpoint_dir', self.checkpoint_dir)
            checkpoint = torch.load(self.checkpoint_dir, map_location=self.device)
            net_G.load_state_dict(checkpoint['model_G_state_dict'])
            # load pre-trained weights
            print('loading pretrained weights...')
        else:
            print('initializing networks...')
            raise NotImplementedError(
                'pre-trained weights %s does not exist...' % self.checkpoint_dir)

        return net_G

    def initialize_net_safety_mapper(self):

        print('initializing neural safety mapper...')
        net_safety_mapper = define_safety_mapper(self.safety_mapper_ckpt_dir, self.m_tokens, device=self.device).to(self.device)
        net_safety_mapper.eval()

        return net_safety_mapper
    
    def run_forwardpass(self, buff_lat, buff_lon, buff_cos_heading, buff_sin_heading):
        """
        Flatten a trajectory pool and run forward pass...
        """

        # buff_lat, buff_lon, buff_cos_heading, buff_sin_heading, buff_vid = traj_pool.flatten_trajectory(
        #     time_length=self.history_length, max_num_vehicles=self.m_tokens, output_vid=True)

        buff_lat = utils.nan_intep_2d(buff_lat, axis=1)
        buff_lon = utils.nan_intep_2d(buff_lon, axis=1)
        buff_cos_heading = utils.nan_intep_2d(buff_cos_heading, axis=1)
        buff_sin_heading = utils.nan_intep_2d(buff_sin_heading, axis=1)

        input_matrix = np.concatenate([buff_lat, buff_lon, buff_cos_heading, buff_sin_heading], axis=1)
        input_matrix = torch.tensor(input_matrix, dtype=torch.float32)

        # # sample an input state from testing data (e.g. 0th state)
        input_matrix = input_matrix.unsqueeze(dim=0) # make sure the input has a shape of N x D
        input_matrix = input_matrix.to(self.device)

        input_matrix[torch.isnan(input_matrix)] = 0.0

        # run prediction
        mean_pos, std_pos, cos_sin_heading = self.net_G(input_matrix)
        pred_mean_pos = mean_pos.detach().cpu().numpy()[0, :, :]
        pred_std_pos = std_pos.detach().cpu().numpy()[0, :, :]
        pred_cos_sin_heading = cos_sin_heading.detach().cpu().numpy()[0, :, :]
        # pred_vid = buff_vid

        pred_lat_mean = pred_mean_pos[:, 0:self.pred_length].astype(np.float64)
        pred_lon_mean = pred_mean_pos[:, self.pred_length:].astype(np.float64)
        pred_lat_std = pred_std_pos[:, 0:self.pred_length].astype(np.float64)
        pred_lon_std = pred_std_pos[:, self.pred_length:].astype(np.float64)
        pred_cos_heading = pred_cos_sin_heading[:, 0:self.pred_length].astype(np.float64)
        pred_sin_heading = pred_cos_sin_heading[:, self.pred_length:].astype(np.float64)

        pred_lat, pred_lon = self.sampling(pred_lat_mean, pred_lon_mean, pred_lat_std, pred_lon_std)
        # pred_lat = pred_lat_mean
        # pred_lon = pred_lon_mean
        
        # print('mean_pos', mean_pos.shape, mean_pos)
        # print('pred_lat_mean', pred_lat_mean.shape, pred_lat_mean)
        # print('pred_lat', pred_lat.shape, pred_lat)
        
        # import pdb; pdb.set_trace()

        # return pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, buff_vid, buff_lat, buff_lon
        return pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, buff_lat, buff_lon


    def do_safety_mapping(self, pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, buff_vid, output_delta_position_mask=False):
        # Neural safety mapping
        delta_position_mask_list = []

        for i in range(4):  # Four consecutive pass of safety mapping network to guarantee safety.
            pred_lat, pred_lon, delta_position_mask = self.net_safety_mapper(pred_lat=pred_lat, pred_lon=pred_lon, pred_cos_heading=pred_cos_heading, pred_sin_heading=pred_sin_heading,
                                                                             pred_vid=pred_vid, device=self.device)
            delta_position_mask_list.append(delta_position_mask)

        delta_position_mask = delta_position_mask_list[0] + delta_position_mask_list[1] + delta_position_mask_list[2] + delta_position_mask_list[3]
        # delta_position_mask = np.logical_or(delta_position_mask_list[0], delta_position_mask_list[1])

        if output_delta_position_mask:
            return pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, delta_position_mask

        return pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid
    
    def sampling(self, pred_lat_mean, pred_lon_mean, pred_lat_std, pred_lon_std):
        """
        Sample a trajectory from predicted mean and std.
        """

        epsilons_lat = np.reshape([random.gauss(0, 1) for _ in range(self.m_tokens)], [-1, 1])
        epsilons_lon = np.reshape([random.gauss(0, 1) for _ in range(self.m_tokens)], [-1, 1])

        pred_lat = pred_lat_mean + epsilons_lat * pred_lat_std
        pred_lon = pred_lon_mean + epsilons_lon * pred_lon_std

        return pred_lat, pred_lon