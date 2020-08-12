#!/usr/bin/env python
import os, sys
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
import numpy as np
import pprint
import time
import torch
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D
from diff_gpmp2.gpmp2.diff_gpmp2_planner import DiffGPMP2Planner
from diff_gpmp2.utils.helpers import load_params
from diff_gpmp2.utils.sdf_utils import sdf_2d
from diff_gpmp2.utils.planner_utils import check_convergence, straight_line_traj


use_cuda = False
step = False
np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
pp = pprint.PrettyPrinter()
torch.set_default_tensor_type(torch.DoubleTensor)
use_cuda = torch.cuda.is_available() if use_cuda else False
device = torch.device('cuda') if use_cuda else torch.device('cpu')

env_file = os.path.abspath("../diff_gpmp2/env/simple_2d/12.png")
plan_param_file  = os.path.abspath('configs/gpmp2_2d_params.yaml')
robot_param_file = os.path.abspath('configs/robot_2d.yaml')
env_param_file   = os.path.abspath('configs/env_2d_params.yaml')
render = True

np.random.seed(0)
torch.manual_seed(0)

#Load parameters
env_data, planner_params, gp_params, obs_params, optim_params, robot_data = load_params(plan_param_file, robot_param_file, env_param_file, device)
env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
env_image = plt.imread(env_file)
res = (env_params['x_lims'][1] - env_params['x_lims'][0])/((env_image.shape[1])*1.)
env_sdf = sdf_2d(env_image, res=res) 
#2D Point robot model
robot = PointRobot2D(robot_data['sphere_radius'][0], use_cuda=use_cuda)

start_conf = torch.tensor([[-4., -4.]], device=device)
start_vel = torch.tensor([[0., 0.]], device=device)
goal_conf = torch.tensor([[4., 4.]], device=device)#[17, 14])
goal_vel = torch.tensor([[0., 0.]], device=device)
startb = torch.cat((start_conf, start_vel), dim=1).unsqueeze(0)
goalb = torch.cat((goal_conf, goal_vel), dim=1).unsqueeze(0)
# th_init.requires_grad_(True)
imb = torch.tensor(env_image, device=device).unsqueeze(0).unsqueeze(0)
sdfb = torch.tensor(env_sdf, device=device).unsqueeze(0).unsqueeze(0)
th_init = straight_line_traj(start_conf, goal_conf, planner_params['total_time_sec'], planner_params['total_time_step'], planner_params['dof'], device)

planner = DiffGPMP2Planner(gp_params, obs_params, planner_params, optim_params, env_params, robot, batch_size=1, use_cuda=use_cuda)

j = 0
th_curr = th_init.unsqueeze(0)
th_init.requires_grad_(True)

th_init_np = th_init.cpu().detach().numpy()

env = Env2D(env_params)
env.initialize_from_file(env_file)
path_init = [th_init_np[i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]

if render:
  env.initialize_plot(start_conf.cpu().numpy()[0], goal_conf.cpu().numpy()[0])
  env.plot_signed_distance_transform()
  env.plot_edge(path_init, color='red')
  plt.show(block=False)

stp = time.time()
while True:
  print "Current iteration, %d"%j
  dthetab, _, err_old, _, _, _, _ = planner.step(th_curr, startb, goalb, imb, sdfb)
  if j == 0: err_init = err_old
  th_curr = th_curr + dthetab
  err_new = planner.error_batch(th_curr, sdfb)
  err_delta = err_new - err_old
  
  # if render:
  #   th_curr_np = th_curr.cpu().detach().numpy()
  #   path_curr = [th_curr_np[0, i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
  #   env.plot_edge(path_curr, color='gray', linestyle='--')#, linewidth=0.1, alpha=1.0-(1.0/(j+0.0001)) )
  #   plt.show(block=False)
  #   if step:
  #     raw_input('Press enter for next step')
  
  j = j + 1
  if check_convergence(dthetab[0], j, err_delta[0], optim_params['tol_err'], optim_params['tol_delta'], optim_params['max_iters']):
    print('Converged')
    break

print('Planning time = %f'%(time.time()-stp))
th_final = th_curr

if render:
  th_final_np = th_final.cpu().detach().numpy()
  path_final = [th_final_np[0, i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
  env.plot_edge(path_final, color='blue')
# stb=  time.time()
# th_final.backward(torch.randn(th_final.shape, device=device))
# print('Backprop time = %f'%(time.time()-stb))

plt.show()
