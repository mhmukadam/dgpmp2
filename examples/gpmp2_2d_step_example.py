#!/usr/bin/env python
import os, sys
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
import numpy as np
import pprint
import time
import torch
import yaml
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D
from diff_gpmp2.gpmp2 import GPMP2Planner
from diff_gpmp2.utils.helpers import load_params
from diff_gpmp2.utils.planner_utils import check_convergence

use_cuda = False
render = True
step = True
np.set_printoptions(threshold=np.nan, linewidth=np.inf)
pp = pprint.PrettyPrinter()
torch.set_default_tensor_type(torch.DoubleTensor)
use_cuda = torch.cuda.is_available() if use_cuda else False
device = torch.device('cuda') if use_cuda else torch.device('cpu')


plan_param_file = os.path.abspath('gpmp2_2d_params.yaml')
robot_param_file = os.path.abspath('robot_2d.yaml')
env_param_file = os.path.abspath('env_2d_params.yaml')
ENV_FILE = os.path.abspath("../diff_gpmp2/env/simple_2d/1.png")

#Load the environment and planning parameters
env_data, planner_params, gp_params, obs_params, optim_params, robot_data = load_params(plan_param_file, robot_param_file, env_param_file, device)
env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
env = Env2D(env_params, use_cuda=use_cuda)
env.initialize_from_file(ENV_FILE)

#2D Point robot model
robot = PointRobot2D(robot_data['sphere_radius'][0], use_cuda=use_cuda)

start_conf = torch.tensor([[-4., -4.]], device=device)
start_vel = torch.tensor([[0., 0.]], device=device)
goal_conf = torch.tensor([[4., 4.]], device=device)#[17, 14])
goal_vel = torch.tensor([[0., 0.]], device=device)
avg_vel = (goal_conf - start_conf)/planner_params['total_time_sec']
start = torch.cat((start_conf, start_vel), dim=1)
goal = torch.cat((goal_conf, goal_vel), dim=1)
total_time_step = planner_params['total_time_step']
th_init_tmp = torch.zeros((int(total_time_step)+1, planner_params['state_dim']), device=device) #Straight   line at constant velocity

for i in range(int(total_time_step)+1):
  th_init_tmp[i, 0:2] = start_conf*(total_time_step - i)*1./total_time_step*1. + goal_conf * i*1./total_time_step*1. #+ np.array([0., 5.0])
  th_init_tmp[i, 2:4] = avg_vel

th_init = th_init_tmp
th_init.requires_grad_(True)
planner = GPMP2Planner(gp_params, obs_params, planner_params, env, robot, use_cuda)


j = 0
th_curr = th_init
th_init_np = th_init.cpu().detach().numpy()
path_init = [th_init_np[i, 0:planner_params['dof']] for i in xrange(total_time_step+1)]

if render:
  env.initialize_plot(start_conf.cpu().numpy()[0], goal_conf.cpu().numpy()[0])
  env.plot_signed_distance_transform()
  env.plot_edge(path_init, color='red')
  plt.show(block=False)


while True:
  print "Current iteration, %d"%j
  dtheta, err_old = planner.step(th_curr, start, goal, optim_params)
  if j == 0: err_init = err_old
  th_curr = th_curr + dtheta
  err_new = planner.error(th_curr)
  err_delta = err_new - err_old
  if render:
    th_curr_np = th_curr.cpu().detach().numpy()
    path_curr = [th_curr_np[i, 0:planner_params['dof']] for i in xrange(total_time_step+1)]
    env.plot_edge(path_curr, color='gray', linestyle='-', linewidth=0.1*j, alpha=1.0-(1.0/(j+0.0001)) )
    plt.show(block=False)
    if step:
      raw_input('Press enter for next step')
  j = j + 1
  if check_convergence(dtheta, j, err_delta, optim_params['tol_err'], optim_params['tol_delta'], optim_params['max_iters']):
    print('Converged')
    break
  
th_final = th_curr
if render:
  th_final_np = th_final.cpu().detach().numpy()
  path_final = [th_final_np[i, 0:planner_params['dof']] for i in xrange(total_time_step+1)]
  env.plot_edge(path_final, linewidth=0.1*j)

plt.show()