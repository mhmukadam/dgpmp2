#!/usr/bin/env python
import os, sys
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
import numpy as np
import pprint
import time
import torch
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobotXYH
from diff_gpmp2.gpmp2.diff_gpmp2_planner import DiffGPMP2Planner
from diff_gpmp2.utils.helpers import rgb2gray, load_params
from diff_gpmp2.utils.sdf_utils import sdf_2d
from diff_gpmp2.utils.planner_utils import straight_line_traj
from datasets import PlanningDataset


use_cuda = False
np.set_printoptions(threshold=np.nan, linewidth=np.inf)
pp = pprint.PrettyPrinter()
torch.set_default_tensor_type(torch.DoubleTensor)
use_cuda = torch.cuda.is_available() if use_cuda else False
device = torch.device('cuda') if use_cuda else torch.device('cpu')

env_file = os.path.abspath("../diff_gpmp2/env/simple_2d/4.png")
plan_param_file  = os.path.abspath('gpmp2_xyh_params.yaml')
robot_param_file = os.path.abspath('robot_2d.yaml')
env_param_file   = os.path.abspath('env_2d_params.yaml')

np.random.seed(0)
torch.manual_seed(0)

#Load parameters
env_data, planner_params, gp_params, obs_params, optim_params, robot_data = load_params(plan_param_file, robot_param_file, env_param_file, device)
env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
env_image = plt.imread(env_file)
if len(env_image.shape) > 2:
  env_image = rgb2gray(env_image)
cell_size = (env_params['x_lims'][1] - env_params['x_lims'][0])/env_image.shape[0]
env_sdf = sdf_2d(env_image, res = cell_size)
#2D Point robot model
robot = PointRobotXYH(robot_data['sphere_radius'][0], use_cuda=use_cuda)

start_conf = torch.tensor([[env_params['x_lims'][0]+1., env_params['y_lims'][0]+ 1., 0.0]], device=device)
start_vel = torch.tensor([[0., 0., 0.]], device=device)
goal_conf = torch.tensor([[env_params['x_lims'][1]-1., env_params['y_lims'][1]-1., np.pi/2.0]], device=device)#[17, 14])
goal_vel = torch.tensor([[0., 0., 0.]], device=device)
start = torch.cat((start_conf, start_vel), dim=1)
goal = torch.cat((goal_conf, goal_vel), dim=1)
th_init = straight_line_traj(start_conf, goal_conf, planner_params['total_time_sec'], planner_params['total_time_step'], planner_params['dof'], device)
th_init.requires_grad_(True)
im = torch.tensor(env_image, device=device)
sdf = torch.tensor(env_sdf, device=device)
planner = DiffGPMP2Planner(gp_params, obs_params, planner_params, optim_params, env_params, robot, use_cuda=use_cuda)
start_t = time.time()
print th_init
th_final, _, err_init, err_final, err_per_iter, err_ext_per_iter, k, time_taken = planner.forward(th_init.unsqueeze(0), start.unsqueeze(0), goal.unsqueeze(0), im.unsqueeze(0).unsqueeze(0), sdf.unsqueeze(0).unsqueeze(0))
print th_final
pp.pprint('Initial cost = %f'%(err_init[0]))
pp.pprint('Final cost = %f'%(err_final[0]))
pp.pprint('Iterations taken = %d'%(k[0]))
pp.pprint('Time taken = %f (seconds)'%(time_taken[0]))


path_init = []
path_final = []

th_init_np = th_init.cpu().detach().numpy()
th_final_np = th_final[0].cpu().detach().numpy()
for i in range(planner_params['total_time_step']+1):
  path_init.append(th_init_np[i, 0:planner_params['dof']])
  path_final.append(th_final_np[i, 0:planner_params['dof']])
  
#Plot the final results
env = Env2D(env_params)
env.initialize_from_file(env_file)
env.initialize_plot(start_conf.cpu().numpy()[0], goal_conf.cpu().numpy()[0])
env.plot_edge(path_init, color='red')
env.plot_edge(path_final)
env.plot_signed_distance_transform()
fig=plt.figure()
plt.plot(err_per_iter[0])
plt.show()
