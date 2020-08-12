#!/usr/bin/env python
import os, sys
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
import numpy as np
import pprint
import time
import torch
from torch.utils.data import Dataset, DataLoader
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D
from diff_gpmp2.gpmp2.diff_gpmp2_planner import DiffGPMP2Planner
from diff_gpmp2.utils.helpers import load_params
from diff_gpmp2.datasets import PlanningDataset


use_cuda = False
np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
pp = pprint.PrettyPrinter()
torch.set_default_tensor_type(torch.DoubleTensor)
use_cuda = torch.cuda.is_available() if use_cuda else False
device = torch.device('cuda') if use_cuda else torch.device('cpu')

dataset_folder   = os.path.abspath('../diff_gpmp2/datasets/dataset_files/dataset_2d_1/')
plan_param_file  = os.path.abspath('configs/gpmp2_2d_params.yaml')
robot_param_file = os.path.abspath('configs/robot_2d.yaml')
env_param_file   = os.path.abspath('configs/env_2d_params.yaml')

np.random.seed(0)
torch.manual_seed(0)

#Get sample problem from dataset
batch_size = 4
dataset    = PlanningDataset(dataset_folder, mode='train', label_subdir='opt_trajs_gpmp2')
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

#Load parameters
env_data, planner_params, gp_params, obs_params, optim_params, robot_data = load_params(plan_param_file, robot_param_file, env_param_file, device)

#Get a batch of data
sample_batch = {}
for i, sample in enumerate(dataloader):
  sample_batch = sample
  if i == 1:
    break

im_b = sample_batch['im']
sdf_b = sample_batch['sdf']
start_b = sample_batch['start']
goal_b = sample_batch['goal']
th_opt_b = sample_batch['th_opt']    
# env_params_b = sample_batch['env_params']
env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
#2D Point robot model
robot = PointRobot2D(robot_data['sphere_radius'][0], use_cuda=use_cuda)

#Initial trajectories are just straight lines from start to goal
total_time_step = planner_params['total_time_step']
total_time_sec = planner_params['total_time_sec']
dof = planner_params['dof']
th_init_tmp = torch.zeros((batch_size, int(total_time_step)+1, planner_params['state_dim']), device=device) #Straight   line at constant velocity

for j in range(batch_size):
  avg_vel = (goal_b[j][0, 0:dof] - start_b[j][0, 0:dof])/total_time_sec
  for i in range(int(total_time_step)+1):
    th_init_tmp[j][i, 0:2] = start_b[j][0, 0:dof]*(total_time_step - i)*1./total_time_step*1. + goal_b[j][0, 0:dof] * i*1./total_time_step*1. #+ np.array([0., 5.0])
    th_init_tmp[j][i, 2:4] = avg_vel

th_init_b = th_init_tmp
th_init_b.requires_grad_(True)

planner = DiffGPMP2Planner(gp_params, obs_params, planner_params, optim_params, env_params, robot, use_cuda=use_cuda)
th_finalb, err_initb, err_finalb, err_per_iterb, jb, timeb = planner.forward(th_init_b, start_b, goal_b, im_b, sdf_b)
# print('Num iterations = {}, Time taken = {}'.format(k, time_taken))



plt.ion()
for i in range(batch_size):
  im = im_b[i, 0, :, :]
  sdf = sdf_b[i, 0, :, : ]
  start = start_b[i]
  goal = goal_b[i]
  th_f = th_finalb[i]
  th_opt = th_opt_b[i]
  env = Env2D(env_params)
  env.initialize_from_image(im, sdf)
  path_f = []
  path_opt = []
  for i in range(th_opt.shape[0]):
    path_f.append(th_f[i,0:2])
    path_opt.append(th_opt[i,0:2])
  env.initialize_plot(start[0][0:2], goal[0][0:2])
  env.plot_edge(path_f)
  # env.plot_edge(path_opt)
  env.plot_signed_distance_transform()
  plt.show()
  input('Press enter to view next data point')
  env.close_plot()
  plt.close()