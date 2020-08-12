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
from diff_gpmp2.utils.planner_utils import straight_line_traj


use_cuda = False
np.set_printoptions(threshold=np.nan, linewidth=np.inf)
pp = pprint.PrettyPrinter()
torch.set_default_tensor_type(torch.DoubleTensor)
use_cuda = torch.cuda.is_available() if use_cuda else False
device = torch.device('cuda') if use_cuda else torch.device('cpu')
#Path to all parameter files
env_file = os.path.abspath("../diff_gpmp2/env/simple_2d/7.png")
plan_param_file = os.path.abspath('gpmp2_2d_params.yaml')
robot_param_file = os.path.abspath('robot_2d.yaml')
env_param_file = os.path.abspath('env_2d_params.yaml')

#Load the environment and planning parameters
env_data, planner_params, gp_params, obs_params, optim_params, robot_data = load_params(plan_param_file, robot_param_file, env_param_file, device)
env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
env = Env2D(env_params, use_cuda=use_cuda)
env.initialize_from_file(env_file)
im = env.image
sdf = env.sedt

#2D Point robot model
robot = PointRobot2D(robot_data['sphere_radius'][0], use_cuda=use_cuda)

start_conf = torch.tensor([[-19., -19.]], device=device)
start_vel = torch.tensor([[0., 0.]], device=device)
goal_conf = torch.tensor([[19., 19.]], device=device)#[17, 14])
goal_vel = torch.tensor([[0., 0.]], device=device)
start = torch.cat((start_conf, start_vel), dim=1)
goal = torch.cat((goal_conf, goal_vel), dim=1)

th_init = straight_line_traj(start_conf, goal_conf, planner_params['total_time_sec'], planner_params['total_time_step'], planner_params['dof'], device)
th_init.requires_grad_(True)
planner = GPMP2Planner(gp_params, obs_params, planner_params, env_params, robot, use_cuda)
th_final, err_init, err_final, err_per_iter, k, time_taken = planner.plan(start, goal, th_init, im, sdf, optim_params)

path_init = []
path_final = []
th_init_np = th_init.cpu().detach().numpy()
th_final_np = th_final.cpu().detach().numpy()

for i in range(planner_params['total_time_step']+1):
  path_init.append(th_init_np[i, 0:planner_params['dof']])
  path_final.append(th_final_np[i, 0:planner_params['dof']])


pp.pprint('Initial cost = %f'%(err_init))
pp.pprint('Final cost = %f'%(err_final))
pp.pprint('Iterations taken = %d'%(k))
pp.pprint('Time taken = %f (seconds)'%(time_taken))

#Plot the final results
env.initialize_plot(start_conf.cpu().numpy()[0], goal_conf.cpu().numpy()[0])
env.plot_edge(path_init, color='red')
env.plot_edge(path_final)
env.plot_signed_distance_transform()
fig, ax = plt.subplots()
ax.plot(err_per_iter)
plt.show()