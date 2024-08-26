import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
import time
import traci

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from .networks import define_G, define_D, set_requires_grad, define_safety_mapper
from .loss import UncertaintyRegressionLoss, GANLoss
from .metric import RegressionAccuracy
from . import utils
from vehicle.utils_vehicle import (to_vehicle, time_buff_to_traj_pool, 
                                   traci_get_vehicle_data, traci_set_vehicle_state)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    """
    Model trainer. This class provides fundamental methods for model training and checkpoint management.
    """

    def __init__(self, configs, dataloaders):

        self.dataloaders = dataloaders
        self.sumo_cmd = configs["sumo_cmd"]
        self.traci_label = "sumo_training"
        print("model_trainer.py sumo_cmd: ", self.sumo_cmd)

        # input and output dimension
        self.input_dim = 4 * configs["history_length"]  # x, y, cos_heading, sin_heading
        self.output_dim = 4 * configs["pred_length"]  # x, y, cos_heading, sin_heading
        self.history_length = configs["history_length"]
        self.pred_length = configs["pred_length"]
        self.m_tokens = configs["max_num_vehicles"]
        self.rolling_step = 1
        self.sim_resol = configs['sim_resol']

        self.model_output = configs["model_output"]  # position or speed
        assert 'position' in self.model_output or 'speed' in self.model_output

        # initialize networks
        self.net_G = define_G(
            configs["model"], input_dim=self.input_dim, output_dim=self.output_dim,
            m_tokens=self.m_tokens).to(device)
        self.net_D = define_D(input_dim=self.output_dim).to(device)

        # initialize safety mapping networks to involve it in the training loop
        self.net_safety_mapper = define_safety_mapper(configs["safety_mapper_ckpt_dir"], self.m_tokens).to(device)
        self.net_safety_mapper.eval()
        set_requires_grad(self.net_safety_mapper, requires_grad=False)

        # Learning rate
        self.lr = configs["lr"]

        # define optimizers
        self.optimizer_G = optim.RMSprop(self.net_G.parameters(), lr=self.lr)
        self.optimizer_D = optim.RMSprop(self.net_D.parameters(), lr=self.lr)

        # define lr schedulers
        self.exp_lr_scheduler_G = lr_scheduler.StepLR(
            self.optimizer_G, step_size=configs["lr_decay_step_size"], gamma=configs["lr_decay_gamma"])
        self.exp_lr_scheduler_D = lr_scheduler.StepLR(
            self.optimizer_D, step_size=configs["lr_decay_step_size"], gamma=configs["lr_decay_gamma"])

        # define loss function and error metric
        self.regression_loss_func_pos = UncertaintyRegressionLoss(choice='mae')
        self.regression_loss_func_heading = UncertaintyRegressionLoss(choice='cos_sin_heading_mae')

        self.gan_loss = GANLoss(gan_mode='vanilla').to(device)

        self.accuracy_metric_pos = RegressionAccuracy(choice='neg_mae')
        self.accuracy_metric_heading = RegressionAccuracy(choice='neg_mae')

        # define some other vars to record the training states
        self.running_acc_pos = []
        self.running_acc_heading = []
        self.running_loss_G, self.running_loss_D = [], []
        self.running_loss_G_reg_pos, self.running_loss_G_reg_heading, self.running_loss_G_adv = [], [], []
        self.train_epoch_acc_pos, self.train_epoch_acc_heading = -1e9, -1e9
        self.train_epoch_loss = 0.0
        self.val_epoch_acc_pos, self.val_epoch_acc_heading = -1e9, -1e9
        self.val_epoch_loss = 0.0
        self.epoch_to_start = 0
        self.max_num_epochs = configs["max_num_epochs"]

        self.is_training = True
        self.batch = None
        self.batch_loss = 0.0
        self.batch_acc_pos = -1e9
        self.batch_acc_heading = -1e9
        self.batch_id = 0
        self.epoch_id = 0

        self.checkpoint_dir = configs["checkpoint_dir"]
        self.vis_dir = configs["vis_dir"]
        print('checkpoint_dir', self.checkpoint_dir)

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        if os.path.exists(self.vis_dir) is False:
            os.makedirs(self.vis_dir, exist_ok=True)

        # buffers to logfile
        self.logfile = {'val_acc_pos': [], 'train_acc_pos': [],
                        'val_acc_heading': [], 'train_acc_heading': [],
                        'val_loss_G': [], 'train_loss_G': [],
                        'val_loss_D': [], 'train_loss_D': [],
                        'epochs': []}

        # tensorboard writer
        self.writer = SummaryWriter(self.vis_dir)

    def _load_checkpoint(self):

        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            print('loading last checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_ckpt.pt'))

            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])
            self.net_G.to(device)

            # update net_D states
            self.net_D.load_state_dict(checkpoint['model_D_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.exp_lr_scheduler_D.load_state_dict(
                checkpoint['exp_lr_scheduler_D_state_dict'])
            self.net_D.to(device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1

            print('Epoch_to_start = %d' % self.epoch_to_start)
            print()

        else:
            print('training from scratch...')

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
            'model_D_state_dict': self.net_D.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'exp_lr_scheduler_D_state_dict': self.exp_lr_scheduler_D.state_dict()
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()
        self.exp_lr_scheduler_D.step()

    def _collect_running_batch_states(self):

        self.running_acc_pos.append(self.batch_acc_pos.item())
        self.running_acc_heading.append(self.batch_acc_heading.item())
        self.running_loss_G.append(self.batch_loss_G.item())
        self.running_loss_D.append(self.batch_loss_D.item())
        self.running_loss_G_reg_pos.append(self.reg_loss_position.item())
        self.running_loss_G_reg_heading.append(self.reg_loss_heading.item())
        self.running_loss_G_adv.append(self.G_adv_loss.item())

        if self.is_training:
            m_batches = len(self.dataloaders['train'])
        else:
            m_batches = len(self.dataloaders['val'])

        if np.mod(self.batch_id, 100) == 1:
            print('Is_training: %s. epoch [%d,%d], batch [%d,%d], reg_loss_pos: %.5f, reg_loss_heading: %.5f, '
                  'G_adv_loss: %.5f, D_adv_loss: %.5f, running_acc_pos: %.5f, running_acc_heading: %.5f'
                  % (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m_batches,
                     self.reg_loss_position.item(), self.reg_loss_heading.item(),
                     self.G_adv_loss.item(), self.D_adv_loss.item(), np.mean(self.running_acc_pos),
                     np.mean(self.running_acc_heading)))

    def _collect_epoch_states(self):

        if self.is_training:
            self.train_epoch_acc_pos = np.mean(self.running_acc_pos).item()
            self.train_epoch_acc_heading = np.mean(self.running_acc_heading).item()
            self.train_epoch_loss_G = np.mean(self.running_loss_G).item()
            self.train_epoch_loss_D = np.mean(self.running_loss_D).item()
            self.train_epoch_loss_G_reg_pos = np.mean(self.running_loss_G_reg_pos).item()
            self.train_epoch_loss_G_reg_heading = np.mean(self.running_loss_G_reg_heading).item()
            self.train_epoch_loss_G_adv = np.mean(self.running_loss_G_adv).item()
            print('Training, Epoch %d / %d, epoch_loss_G= %.5f, epoch_loss_D= %.5f, epoch_acc_pos= %.5f, epoch_acc_heading= %.5f' %
                  (self.epoch_id, self.max_num_epochs-1,
                   self.train_epoch_loss_G, self.train_epoch_loss_D, self.train_epoch_acc_pos, self.train_epoch_acc_heading))

            self.writer.add_scalar("Accuracy/train_pos", self.train_epoch_acc_pos, self.epoch_id)
            self.writer.add_scalar("Accuracy/train_heading", self.train_epoch_acc_heading, self.epoch_id)
            self.writer.add_scalar("Loss/train_G", self.train_epoch_loss_G, self.epoch_id)
            self.writer.add_scalar("Loss/train_D", self.train_epoch_loss_D, self.epoch_id)
            self.writer.add_scalar("Loss/train_G_reg_position", self.train_epoch_loss_G_reg_pos, self.epoch_id)
            self.writer.add_scalar("Loss/train_G_reg_heading", self.train_epoch_loss_G_reg_heading, self.epoch_id)
            self.writer.add_scalar("Loss/train_G_adv", self.train_epoch_loss_G_adv, self.epoch_id)
        else:
            self.val_epoch_acc_pos = np.mean(self.running_acc_pos).item()
            self.val_epoch_acc_heading = np.mean(self.running_acc_heading).item()
            self.val_epoch_loss_G = np.mean(self.running_loss_G).item()
            self.val_epoch_loss_D = np.mean(self.running_loss_D).item()
            self.val_epoch_loss_G_reg_pos = np.mean(self.running_loss_G_reg_pos).item()
            self.val_epoch_loss_G_reg_heading = np.mean(self.running_loss_G_reg_heading).item()
            self.val_epoch_loss_G_adv = np.mean(self.running_loss_G_adv).item()
            print('Validation, Epoch %d / %d, epoch_loss_G= %.5f, epoch_loss_D= %.5f, epoch_acc_pos= %.5f, epoch_acc_heading= %.5f' %
                  (self.epoch_id, self.max_num_epochs - 1,
                   self.val_epoch_loss_G, self.val_epoch_loss_D, self.val_epoch_acc_pos, self.val_epoch_acc_heading))

            self.writer.add_scalar("Accuracy/val_pos", self.val_epoch_acc_pos, self.epoch_id)
            self.writer.add_scalar("Accuracy/val_heading", self.val_epoch_acc_heading, self.epoch_id)
            self.writer.add_scalar("Loss/val_G", self.val_epoch_loss_G, self.epoch_id)
            self.writer.add_scalar("Loss/val_D", self.val_epoch_loss_D, self.epoch_id)
            self.writer.add_scalar("Loss/val_G_reg_position", self.val_epoch_loss_G_reg_pos, self.epoch_id)
            self.writer.add_scalar("Loss/val_G_reg_heading", self.val_epoch_loss_G_reg_heading, self.epoch_id)
            self.writer.add_scalar("Loss/val_G_adv", self.val_epoch_loss_G_adv, self.epoch_id)
        print()

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        sum_val_epoch_acc = self.val_epoch_acc_pos + self.val_epoch_acc_heading
        print('Latest model updated. Epoch_acc_pos=%.4f, Epoch_acc_heading=%.4f,'
              'Sum Epoch_acc=%.4f'
              % (self.val_epoch_acc_pos, self.val_epoch_acc_heading, sum_val_epoch_acc))
        print()

    def _update_logfile(self):

        logfile_path = os.path.join(self.checkpoint_dir, 'logfile.json')

        # read historical logfile and update
        if os.path.exists(logfile_path):
            with open(logfile_path) as json_file:
                self.logfile = json.load(json_file)

        self.logfile['train_acc_pos'].append(self.train_epoch_acc_pos)
        self.logfile['train_acc_heading'].append(self.train_epoch_acc_heading)
        self.logfile['val_acc_pos'].append(self.val_epoch_acc_pos)
        self.logfile['val_acc_heading'].append(self.val_epoch_acc_heading)
        self.logfile['train_loss_G'].append(self.train_epoch_loss_G)
        self.logfile['val_loss_G'].append(self.val_epoch_loss_G)
        self.logfile['train_loss_D'].append(self.train_epoch_loss_D)
        self.logfile['val_loss_D'].append(self.val_epoch_loss_D)
        self.logfile['epochs'].append(self.epoch_id)

        # save new logfile to disk
        with open(logfile_path, "w") as fp:
            json.dump(self.logfile, fp)

    def _visualize_logfile(self):

        plt.subplot(1, 2, 1)
        plt.plot(self.logfile['epochs'], self.logfile['train_loss_G'])
        plt.plot(self.logfile['epochs'], self.logfile['val_loss_G'])
        plt.plot(self.logfile['epochs'], self.logfile['train_loss_D'])
        plt.plot(self.logfile['epochs'], self.logfile['val_loss_D'])
        plt.xlabel('m epochs')
        plt.ylabel('loss')
        plt.legend(['train loss G', 'val loss G', 'train loss D', 'val loss D'])

        plt.subplot(1, 2, 2)
        plt.plot(self.logfile['epochs'], self.logfile['train_acc_pos'])
        plt.plot(self.logfile['epochs'], self.logfile['train_acc_heading'])
        plt.plot(self.logfile['epochs'], self.logfile['val_acc_pos'])
        plt.plot(self.logfile['epochs'], self.logfile['val_acc_heading'])
        plt.xlabel('m epochs')
        plt.ylabel('acc')
        plt.legend(['train acc', 'train heading', 'val acc', 'val heading'])

        logfile_path = os.path.join(self.vis_dir, 'logfile.png')
        plt.savefig(logfile_path)
        plt.close('all')

    def _visualize_prediction(self):

        print('visualizing prediction...')
        print()

        batch, idx_data = next(iter(self.dataloaders['val']))
        with torch.no_grad():
            self._forward_pass(batch)

        vis_path = os.path.join(
            self.vis_dir, 'epoch_'+str(self.epoch_id).zfill(5)+'_pred.png')

        x_pos, gt = self.x[:, :, :int(self.output_dim / 2)], self.gt[:, :, :int(self.output_dim / 2)]
        pred_pos_mean, pred_pos_std = self.G_pred_mean[0][:, :, :int(self.output_dim / 2)], self.G_pred_std[0][:, :, :int(self.output_dim / 2)]

        utils.visualize_training(
            x=x_pos, pred_true=gt,
            pred_mean=pred_pos_mean, pred_std=pred_pos_std, vis_path=vis_path)

    def _sampling_from_mu_and_std(self, mu, std, heading_mu):

        bs, _, _ = mu.shape
        epsilons_lat = torch.tensor([random.gauss(0, 1) for _ in range(bs * self.m_tokens)]).to(device).detach()
        epsilons_lon = torch.tensor([random.gauss(0, 1) for _ in range(bs * self.m_tokens)]).to(device).detach()
        epsilons_lat = torch.reshape(epsilons_lat, [bs, -1, 1])
        epsilons_lon = torch.reshape(epsilons_lon, [bs, -1, 1])

        lat_mean = mu[:, :, 0:self.pred_length]
        lon_mean = mu[:, :, self.pred_length:]
        lat_std = std[:, :, 0:self.pred_length]
        lon_std = std[:, :, self.pred_length:]

        lat = lat_mean + epsilons_lat * lat_std
        lon = lon_mean + epsilons_lon * lon_std

        return torch.cat([lat, lon, heading_mu], dim=-1)

    def _forward_pass(self, batch, rollout=5):
        self.batch = batch
        self.x = batch['input'].to(device)
        self.gt = self.batch['gt'].to(device)

        self.mask = torch.ones_like(self.gt)
        self.mask[torch.isnan(self.gt)] = 0.0
        self.rollout_mask = torch.ones_like(self.gt)
        self.rollout_mask[torch.isnan(self.gt)] = 0.0
        self.rollout_mask[torch.sum(self.mask, dim=-1) > 0, :] = 1.0

        self.x[torch.isnan(self.x)] = 0.0
        self.gt[torch.isnan(self.gt)] = 0.0

        x_input = self.x
        self.G_pred_mean, self.G_pred_std = [], []
        self.rollout_pos = []
        for _ in range(rollout):
            mu, std, cos_sin_heading = self.net_G(x_input)
            # HL: why do we need sampling?
            # x_input = self._sampling_from_mu_and_std(mu, std, cos_sin_heading)
            x_input = x_input * self.rollout_mask  # For future rollouts
            # x_input = self.net_safety_mapper.safety_mapper_in_the_training_loop_forward(x_input)
            self.rollout_pos.append(x_input)
            self.G_pred_mean.append(torch.cat([mu, cos_sin_heading], dim=-1))
            self.G_pred_std.append(std)

        print('x_input', x_input.shape) # [32, 32, 20]
        print('self.G_pred_mean', len(self.G_pred_mean), self.G_pred_mean[0].shape) #  1 torch.Size([32, 32, 20])
        print('self.G_pred_std', len(self.G_pred_std), self.G_pred_std[0].shape) # 1 torch.Size([32, 32, 10])
        print('self.rollout_mask', self.rollout_mask.shape) # torch.Size([32, 32, 20])
        # assert False


    def _compute_acc(self):
        G_pred_mean_at_step0 = self.G_pred_mean[0]

        G_pred_mean_pos_at_step0 = G_pred_mean_at_step0[:, :, :int(self.output_dim / 2)]
        G_pred_cos_sin_heading_at_step0 = G_pred_mean_at_step0[:, :, int(self.output_dim / 2):]

        gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
        gt_cos_sin_heading, mask_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]

        self.batch_acc_pos = self.accuracy_metric_pos(G_pred_mean_pos_at_step0.detach(), gt_pos.detach(), mask=mask_pos > 0)
        self.batch_acc_heading = self.accuracy_metric_heading(G_pred_cos_sin_heading_at_step0.detach(), gt_cos_sin_heading.detach(), mask=mask_heading > 0)

    def _compute_loss_G(self):

        # reg loss (between pred0 and gt)
        G_pred_mean_at_step0 = self.G_pred_mean[0]
        G_pred_std_at_step0 = self.G_pred_std[0]

        G_pred_pos_at_step0, G_pred_cos_sin_heading_at_step0 = G_pred_mean_at_step0[:, :, :int(self.output_dim / 2)], G_pred_mean_at_step0[:, :, int(self.output_dim / 2):]
        G_pred_std_pos_at_step0 = G_pred_std_at_step0

        gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
        gt_cos_sin_heading, mask_cos_sin_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]

        self.reg_loss_position = self.regression_loss_func_pos(G_pred_pos_at_step0, G_pred_std_pos_at_step0, gt_pos, weight=mask_pos)
        self.reg_loss_heading = 20 * self.regression_loss_func_heading(y_pred_mean=G_pred_cos_sin_heading_at_step0, y_pred_std=None, y_true=gt_cos_sin_heading, weight=mask_cos_sin_heading)

        D_pred_fake = self.net_D(self.G_pred_mean[0])
        # Filter out ghost vehicles
        # Reformat the size into num x n, where num = bs * m_token - ghost vehicles (also for those have missing values in gt)
        ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 20).flatten()  # bs * m_token.
        D_pred_fake_filtered = (D_pred_fake.flatten())[ghost_vehicles_mask]
        self.G_adv_loss = 0.1*self.gan_loss(D_pred_fake_filtered, True)

        self.batch_loss_G = self.reg_loss_position + self.reg_loss_heading + self.G_adv_loss
        print('self.batch_loss_G', self.batch_loss_G)

    def _compute_loss_D(self):

        D_pred_fake = self.net_D(self.G_pred_mean[0].detach())
        D_pred_real = self.net_D(self.gt.detach())

        # Filter out ghost vehicles
        pred_fake_ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 20).flatten()  # bs * m_token.
        D_pred_fake_filtered = (D_pred_fake.flatten())[pred_fake_ghost_vehicles_mask]

        pred_real_ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 20).flatten()  # bs * m_token.
        D_pred_real_filtered = (D_pred_real.flatten())[pred_real_ghost_vehicles_mask]

        D_adv_loss_fake = self.gan_loss(D_pred_fake_filtered, False)
        D_adv_loss_real = self.gan_loss(D_pred_real_filtered, True)
        self.D_adv_loss = 0.5 * (D_adv_loss_fake + D_adv_loss_real)

        self.batch_loss_D = self.D_adv_loss

    def _backward_G(self):
        self.batch_loss_G.backward()

    def _backward_D(self):
        self.batch_loss_D.backward()

    def _clear_cache(self):
        self.running_acc_pos, self.running_acc_heading = [], []
        self.running_loss_G, self.running_loss_D = [], []
        self.running_loss_G_reg_pos, self.running_loss_G_reg_heading, self.running_loss_G_adv = [], [], []

    def _forward_pass_sim(self, batch, rollout=1):
        self.batch = batch
        self.x = batch['input'].to(device)   #[32,32,20] history_length*(lat, lon, cos, sin),5*4
        self.gt = self.batch['gt'].to(device) #[32,32,20]

        self.mask = torch.ones_like(self.gt)
        self.mask[torch.isnan(self.gt)] = 0.0

        self.x[torch.isnan(self.x)] = 0.0
        self.gt[torch.isnan(self.gt)] = 0.0

        idx = self.batch['idx'].to(device) #[32]
        veh_ids = self.batch['vehicle_ids'].to(device)

        batch_size = len(self.x)

        self.G_pred_mean, self.G_pred_std = [], []
        mean_pos_cos_sin_heading = torch.zeros((batch_size, self.m_tokens, self.output_dim)).to(device)
        std_pos = torch.zeros((batch_size, self.m_tokens, 2 * self.pred_length)).to(device)

        for i_batch in range(batch_size):
            outputs = self._forward_pass_sim_one_batch(self.x[i_batch], idx[i_batch], veh_ids[i_batch])
            mean_pos_cos_sin_heading[i_batch,:,:] = outputs['mean_pos_cos_sin_heading'].to(device)
            std_pos[i_batch,:,:] = outputs['std_pos'].to(device)
            # self.x[i_batch,:,:] = outputs['x_history'].to(device)


        self.G_pred_mean.append(mean_pos_cos_sin_heading)
        self.G_pred_std.append(std_pos)

    def _forward_pass_sim_one_batch(self, one_input, idx_history, veh_ids, debug=True):
    
        outputs = {}
        outputs['mean_pos_cos_sin_heading'] = torch.zeros((self.m_tokens, self.output_dim)).to(device)
        outputs['std_pos'] = torch.zeros((self.m_tokens, 2 * self.pred_length)).to(device)

        pred_lat_mean_loop = torch.zeros((self.m_tokens, self.pred_length)).to(device)
        pred_lon_mean_loop = torch.zeros((self.m_tokens, self.pred_length)).to(device)
        pred_lat_std_loop = torch.zeros((self.m_tokens, self.pred_length)).to(device)
        pred_lon_std_loop = torch.zeros((self.m_tokens, self.pred_length)).to(device)
        pred_cos_heading_loop = torch.zeros((self.m_tokens, self.pred_length)).to(device)
        pred_sin_heading_loop = torch.zeros((self.m_tokens, self.pred_length)).to(device)

        traci.start(self.sumo_cmd, label=self.traci_label)
        # time.sleep(1)
        step = 0
        while step < idx_history:
            traci.simulationStep()
            step += 1

        idx_output = 0
        TIME_BUFF = []
        is_first_match = False
        while step < idx_history + self.history_length + self.pred_length - 1:
            traci.simulationStep()

            assert step >= idx_history
            vehicle_list = traci_get_vehicle_data()
            TIME_BUFF.append(vehicle_list)

            if debug:
                print('step added', step)

            if step < idx_history + self.history_length - 1:
                step += 1
                continue

            assert len(TIME_BUFF) == self.history_length, len(TIME_BUFF)

            traj_pool = time_buff_to_traj_pool(TIME_BUFF)

            buff_lat, buff_lon, buff_cos_heading, buff_sin_heading, \
            buff_vid, buff_speed, buff_acc, buff_road_id, buff_lane_id, buff_lane_index = \
            traj_pool.flatten_trajectory(
                time_length=self.history_length, max_num_vehicles=self.m_tokens, output_vid=True)
            
            # Check if vehicle id is the same order
            buff_vid = torch.tensor(buff_vid, dtype=torch.float32).to(device)
            # print('veh_ids', veh_ids.shape, veh_ids) # [32, 10]
            # print('buff_vid', buff_vid.shape, buff_vid) # [32, 5]
            assert torch.allclose(veh_ids[:, :self.history_length], buff_vid, equal_nan=True), (
                veh_ids[:, :self.history_length], buff_vid)
            
            TIME_BUFF = TIME_BUFF[self.rolling_step:]

            if self.model_output == 'position' or 'position' in self.model_output:
                buff_lat, buff_lon = buff_lat, buff_lon
            elif self.model_output == 'speed':
                buff_lat, buff_lon = buff_speed, buff_acc

            input_matrix = np.concatenate([buff_lat, buff_lon, buff_cos_heading, buff_sin_heading], axis=-1)
            input_matrix = torch.tensor(input_matrix, dtype=torch.float32).to(device)

            input_matrix = input_matrix.unsqueeze(dim=0).type(torch.float32) # make sure the input has a shape of N x D. HL: 1 x N x D?
            one_input = one_input.unsqueeze(dim=0).type(torch.float32)

            input_matrix[torch.isnan(input_matrix)] = 0.0

            # run prediction
            mean_pos, std_pos, cos_sin_heading = self.net_G(input_matrix)

            if 'position' in self.model_output:
                if step - self.history_length + 1 == idx_history :
                    assert torch.allclose(one_input, input_matrix, equal_nan=True), (one_input, input_matrix)
                    is_first_match = True
                # assert False, "ITs true in step"
            else:
                is_first_match = True

            # if torch.allclose(one_input, input_matrix, equal_nan=True):
            #     print('step,', step)
            #     print('idx_history', idx_history)
            #     assert False, "ITs true"

            #####
            # remove batch
            # print('mean_pos', mean_pos.shape) # [1, 32, 10]
            assert mean_pos.shape[0] == 1
            pred_lat_mean = mean_pos[:, :, 0:self.pred_length]
            pred_lon_mean = mean_pos[:, :, self.pred_length:]
            pred_lat_std = std_pos[:, :, 0:self.pred_length]
            pred_lon_std = std_pos[:, :, self.pred_length:]
            pred_cos_heading_mean = cos_sin_heading[:, :, 0:self.pred_length]
            pred_sin_heading_mean = cos_sin_heading[:, :, self.pred_length:]

            # print('pred_lat_mean_loop', pred_lat_mean_loop.shape) # [32, 5]
            # print('pred_lat_mean', pred_lat_mean.shape) # [1, 32, 5]
            assert pred_lat_mean.shape[0] == 1
            pred_lat_mean_loop[:, idx_output] = pred_lat_mean[0, :, 0]
            pred_lon_mean_loop[:, idx_output] = pred_lon_mean[0, :, 0]
            pred_lat_std_loop[:, idx_output] = pred_lat_std[0, :, 0]
            pred_lon_std_loop[:, idx_output] = pred_lon_std[0, :, 0]
            pred_cos_heading_loop[:, idx_output] = pred_cos_heading_mean[0, :, 0]
            pred_sin_heading_loop[:, idx_output] = pred_sin_heading_mean[0, :, 0]

            #### Feedback to next iteration ####
            pred_lat = pred_lat_mean[0, :, :]
            pred_lon = pred_lon_mean[0, :, :]
            pred_cos_heading = pred_cos_heading_mean[0, :, :]
            pred_sin_heading = pred_sin_heading_mean[0, :, :]

            pred_speed = pred_lat
            pred_acceleration = pred_lon
            traci_set_vehicle_state(self.model_output, buff_vid.cpu().detach().numpy(),
                pred_lat.cpu().detach().numpy(), pred_lon.cpu().detach().numpy(), 
                pred_cos_heading.cpu().detach().numpy(), pred_sin_heading.cpu().detach().numpy(),
                pred_speed.cpu().detach().numpy(), pred_acceleration.cpu().detach().numpy(), self.sim_resol)


            idx_output += 1
            step += 1
            if debug:
                print("Next iteration")

        traci.close()

        assert idx_output == 5, idx_output
        assert is_first_match, "Training: Input history didn't match the dataloader"

        outputs['mean_pos_cos_sin_heading'][:, :] = torch.cat(
            [pred_lat_mean_loop, pred_lon_mean_loop, pred_cos_heading_loop, 
             pred_cos_heading_loop], axis=-1).to(device)
        outputs['std_pos'][:, :] = torch.cat([pred_lat_std_loop, pred_lon_std_loop], axis=-1).to(device)

        return outputs

        
    def train_models(self):
        """
        Main training loop. We loop over the dataset multiple times.
        In each minibatch, we perform gradient descent on the network parameters.
        """

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            # ################## train #################
            # ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            
            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass_sim(batch, rollout=1)
                # self._forward_pass(batch, rollout=1)
                print("batch['input']", batch['input'].shape)
                print("batch['gt']", batch['gt'].shape)
                print("self.batch['idx']", batch['idx'].shape, batch['idx'])
                # assert False

                # update D
                set_requires_grad(self.net_D, True)
                self.optimizer_D.zero_grad()
                self._compute_loss_D()
                self._backward_D()
                self.optimizer_D.step()
                # update G
                set_requires_grad(self.net_D, False)
                self.optimizer_G.zero_grad()
                self._compute_loss_G()
                self._backward_G()
                self.optimizer_G.step()
                # evaluate acc
                self._compute_acc()

                self._collect_running_batch_states()
            self._collect_epoch_states()

            ################## Eval ##################
            ##########################################
            print('Begin evaluation...')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()  # Set model to eval mode

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass_sim(batch, rollout=1)
                    # self._forward_pass(batch, rollout=1)
                    self._compute_loss_G()
                    self._compute_loss_D()
                    self._compute_acc()
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_checkpoints()

            ########### Update_LR Scheduler ##########
            ##########################################
            self._update_lr_schedulers()

            ############## Update logfile ############
            ##########################################
            self._update_logfile()
            self._visualize_logfile()

            ########### Visualize Prediction #########
            ##########################################
            # self._visualize_prediction()



