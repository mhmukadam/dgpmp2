#!/usr/bin/env python
import torch
import os, sys
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
import numpy as np
from ompl import base as ob
from ompl import geometric as og
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D
from diff_gpmp2.ompl_rrtstar import RRTStar
from diff_gpmp2.utils.helpers import load_params
from diff_gpmp2.utils.planner_utils import path_to_traj_avg_vel
from diff_gpmp2.gpmp2 import GPMP2Planner


use_cuda = False
np.set_printoptions(threshold=np.nan, linewidth=np.inf)
torch.set_default_tensor_type(torch.DoubleTensor)
use_cuda = torch.cuda.is_available() if use_cuda else False
device = torch.device('cuda') if use_cuda else torch.device('cpu')

#GPMP2 and env setup
env_file = os.path.abspath('../diff_gpmp2/env/simple_2d/7.png')
plan_param_file = os.path.abspath('gpmp2_2d_params.yaml')
robot_param_file = os.path.abspath('robot_2d.yaml')
env_param_file = os.path.abspath('env_2d_params.yaml')

env_data, planner_params, gp_params, obs_params, optim_params, robot_data = load_params(plan_param_file, robot_param_file, env_param_file, device)
env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
env = Env2D(env_params)
env.initialize_from_file(env_file)
robot = PointRobot2D(robot_data['sphere_radius'], use_cuda=use_cuda)



#RRTstar setup
space = ob.RealVectorStateSpace(2)
bounds = ob.RealVectorBounds(2)
bounds.setLow(env_params['x_lims'][0])
bounds.setHigh(env_params['x_lims'][1])
space.setBounds(bounds)

init_planner = RRTStar(space, bounds, env, robot, planner_params, obs_params)
smooth_planner = GPMP2Planner(gp_params, obs_params, planner_params, env, robot, use_cuda)

start_conf = torch.tensor([[env_params['x_lims'][0]+0.1, env_params['y_lims'][0] + 0.1]], device=device)
start_vel = torch.tensor([[0., 0.]], device=device)
goal_conf = torch.tensor([[env_params['x_lims'][1]-0.1, env_params['y_lims'][1]- 0.1]], device=device)#[17, 14])
goal_vel = torch.tensor([[0., 0.]], device=device)
start = torch.cat((start_conf, start_vel), dim=1)
goal = torch.cat((goal_conf, goal_vel), dim=1)


#Get initial RRTstar path
init_path = init_planner.plan(start_conf, goal_conf, 1.5)
#Run gpmp2 to get smooth path on top
th_init = path_to_traj_avg_vel(init_path, planner_params['total_time_sec'], planner_params['dof'], device)
th_final, err_init, err_final, err_per_iter, k, time_taken = smooth_planner.plan(start, goal, th_init, optim_params)

th_final_np = th_final.cpu().detach().numpy()
path_final = [th_final_np[i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]

print('Initial cost = %f'%(err_init))
print('Final cost = %f'%(err_final))
print('Iterations taken = %d'%(k))
print('Time taken = %f (seconds)'%(time_taken))

#Plot the final results



env.initialize_plot(start_conf.numpy()[0], goal_conf.numpy()[0])
env.plot_edge(init_path, color='red', linestyle='--')
env.plot_edge(path_final)
env.plot_signed_distance_transform()
plt.show()