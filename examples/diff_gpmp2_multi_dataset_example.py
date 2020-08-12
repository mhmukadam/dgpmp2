#!/usr/bin/env python
import os, sys
sys.path.insert(0, "..")
import matplotlib
matplotlib.use('Agg')
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
from diff_gpmp2.utils.planner_utils import check_convergence, check_convergence_batch
from datasets import PlanningDatasetMulti


use_cuda = False
np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
pp = pprint.PrettyPrinter()
torch.set_default_tensor_type(torch.DoubleTensor)
use_cuda = torch.cuda.is_available() if use_cuda else False
device = torch.device('cuda') if use_cuda else torch.device('cpu')

dataset_folder1   = os.path.abspath('../datasets/dataset_files/dataset_2d_1/')
dataset_folder2   = os.path.abspath('../datasets/dataset_files/dataset_2d_7/')
plan_param_file   = os.path.abspath(dataset_folder1 + '/gpmp2_params.yaml')
robot_param_file  = os.path.abspath(dataset_folder1 +'/robot.yaml')
env_param_file    = os.path.abspath(dataset_folder1 +'/env_params.yaml')

np.random.seed(0)
torch.manual_seed(0)

#Get sample problem from dataset
batch_size = 10
dataset    = PlanningDatasetMulti([dataset_folder1, dataset_folder2], mode='train', label_subdir='opt_trajs_gpmp2')
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

#Load parameters
env_data, planner_params, gp_params, obs_params, optim_params, robot_data = load_params(plan_param_file, robot_param_file, env_param_file, device)

#Get a batch of data
sample_batch = {}
for i, sample in enumerate(dataloader):
  sample_batch = sample
  if i == 1:
    break
env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
imb     = sample_batch['im']
res     = (env_params['x_lims'][1] - env_params['x_lims'][0])/(imb[0].shape[-1]*1.)
sdfb    = sample_batch['sdf'] * res
startb  = sample_batch['start']
goalb   = sample_batch['goal']
th_optb = sample_batch['th_opt']    
# env_params_b = sample_batch['env_params']
#2D Point robot model
total_time_step = planner_params['total_time_step']
robot = PointRobot2D(robot_data['sphere_radius'][0], batch_size, total_time_step+1, use_cuda=use_cuda)

#Initial trajectories are just straight lines from start to goal
total_time_sec = planner_params['total_time_sec']
dof = planner_params['dof']
th_init_tmp = torch.zeros((batch_size, int(total_time_step)+1, planner_params['state_dim']), device=device) #Straight   line at constant velocity

for j in xrange(batch_size):
  avg_vel = (goalb[j][0, 0:dof] - startb[j][0, 0:dof])/total_time_sec
  for i in range(int(total_time_step)+1):
    th_init_tmp[j][i, 0:2] = startb[j][0, 0:dof]*(total_time_step - i)*1./total_time_step*1. + goalb[j][0, 0:dof] * i*1./total_time_step*1. #+ np.array([0., 5.0])
    th_init_tmp[j][i, 2:4] = avg_vel

th_initb = th_init_tmp
th_initb.requires_grad_(True)

planner = DiffGPMP2Planner(gp_params, obs_params, planner_params, optim_params, env_params, robot, batch_size=batch_size, use_cuda=use_cuda)

itr = 0
th_currb = th_initb
conv_vec = torch.zeros(batch_size,1,1).byte() 
stp = time.time()
while True:
  print "Current iteration, %d"%itr
  dthetab, _, _, _, _, _, _ = planner.step(th_currb, startb, goalb, imb, sdfb)
  dthetab = torch.where(conv_vec < 1, dthetab, torch.zeros(batch_size, total_time_step+1, planner_params['state_dim'], device=device))
  # if itr == 0: err_init = err_old
  err_old  = planner.error_batch(th_currb, sdfb)
  th_currb = th_currb + dthetab #torch.where(conv_vec < 1, th_currb + dthetab, th_currb)
  err_new  = planner.error_batch(th_currb, sdfb)
  err_delta = err_new - err_old
  # print torch.norm(dthetab.view(batch_size,-1), p=2, dim=1) 
  # if render:
  #   th_curr_np = th_curr.cpu().detach().numpy()
  #   path_curr = [th_curr_np[0, i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
  #   env.plot_edge(path_curr, color='gray', linestyle='--')#, linewidth=0.1, alpha=1.0-(1.0/(j+0.0001)) )
  #   plt.show(block=False)
  #   if step:
  #     raw_input('Press enter for next step')

  itr = itr + 1
  with torch.no_grad():
    conv_vec = check_convergence_batch(dthetab, itr, err_delta, optim_params['tol_err'], optim_params['tol_delta'], optim_params['max_iters'], device=device)
    if torch.sum(conv_vec) == batch_size:
      print('All converged')
      break

print('Planning time = %f'%(time.time()-stp))
th_finalb = th_currb
stb=  time.time()
th_finalb.backward(torch.randn(th_finalb.shape, device=device))
print('Backprop time = %f'%(time.time()-stb))


plt.ion()
for i in xrange(batch_size):
  im = imb[i, 0, :, :]
  sdf = sdfb[i, 0, :, : ]
  start = startb[i]
  goal = goalb[i]
  th_f = th_finalb[i]
  th_opt = th_optb[i]
  env = Env2D(env_params)
  env.initialize_from_image(im, sdf)
  path_f = []
  path_opt = []
  for i in xrange(th_opt.shape[0]):
    path_f.append(th_f[i,0:2])
    path_opt.append(th_opt[i,0:2])
  env.initialize_plot(start[0][0:2], goal[0][0:2])
  env.plot_edge(path_f)
  env.plot_edge(path_opt, color='red')
  env.plot_signed_distance_transform()
  plt.show()
  raw_input('Press enter to view next data point')
  env.close_plot()
  plt.close()