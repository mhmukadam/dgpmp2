#!/usr/bin/env python
import os, sys
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import yaml
from ompl import base as ob
from ompl import geometric as og
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D
from diff_gpmp2.gpmp2 import GPMP2Planner, DiffGPMP2Planner
from diff_gpmp2.ompl_rrtstar import RRTStar
from diff_gpmp2.utils.planner_utils import path_to_traj_avg_vel


use_cuda = False
np.set_printoptions(threshold=np.nan, linewidth=np.inf)
torch.set_default_tensor_type(torch.DoubleTensor)
use_cuda = torch.cuda.is_available() if use_cuda else False
device = torch.device('cuda') if use_cuda else torch.device('cpu')


def load_prob_params(param_file, robot_file, env_file):
  with open(param_file, 'r') as fp:
    planner_data = yaml.load(fp)

  planner_params = planner_data['gpmp2']['planner_params']
  gp_params = planner_data['gpmp2']['gp_params']
  obs_params = planner_data['gpmp2']['obs_params']
  optim_params = planner_data['gpmp2']['optim_params']
  with open(env_file, 'r') as fp:
    env_data = yaml.load(fp)
   
  with open(robot_file, 'r') as fp:
    robot_data = yaml.load(fp)


  gp_params['Q_c_inv'] = torch.tensor(gp_params['Q_c_inv'])
  gp_params['K_s'] = torch.tensor(gp_params['K_s'])
  gp_params['K_g'] = torch.tensor(gp_params['K_g'])
  obs_params['cost_sigma'] = torch.tensor(obs_params['cost_sigma'])
  obs_params['epsilon_dist'] = torch.tensor(obs_params['epsilon_dist'])
  robot_data['sphere_radius'] = torch.tensor(robot_data['sphere_radius'])
  print planner_params
  print gp_params['Q_c_inv']
  print env_data
  print optim_params
  print obs_params

  return env_data, planner_params, gp_params, obs_params, optim_params, robot_data  

def get_random_2d_confs(x_lims, y_lims, env, eps):
  is_feas_start = False
  is_feas_goal = False
  min_dist_achieved = False
  lbx = x_lims[0] + 0.5
  lby = y_lims[0] + 0.5
  ubx = x_lims[1] - 0.5
  uby = y_lims[1] - 0.5
  max_d = np.linalg.norm(np.array([ubx, uby]) - np.array([lbx, lby]))
  while not is_feas_start:
    start_x = lbx + torch.rand(1).item() * (ubx - lbx) 
    start_y =  lby + torch.rand(1).item() * (uby - lby) 
    start_conf = torch.tensor([[start_x, start_y]], device=device)
    is_feas_start = env.is_feasible(start_conf[0], eps)
  
  num_tries = 0
  while (not is_feas_goal) or (not min_dist_achieved):
    goal_x = lbx + torch.rand(1).item() * (ubx - lbx)
    goal_y = lby + torch.rand(1).item() * (uby - lby)
    goal_conf = torch.tensor([[goal_x, goal_y ]], device=device)
    is_feas_goal = env.is_feasible(goal_conf[0], eps)
    if is_feas_goal:
      dist = torch.norm(goal_conf - start_conf)
      if dist.item() >= 0.6*max_d or num_tries > 15:
        min_dist_achieved = True
      else:
        num_tries += 1
  return start_conf, goal_conf

def straight_line_traj(start_conf, goal_conf, num_inter_states, state_dim):
  start_vel = torch.tensor([[0., 0.]], device=device)
  goal_vel = torch.tensor([[0., 0.]], device=device)
  avg_vel = (goal_conf - start_conf)/num_inter_states
  start = torch.cat((start_conf, start_vel), dim=1)
  goal = torch.cat((goal_conf, goal_vel), dim=1)
  th_init = torch.zeros((int(num_inter_states)+1, state_dim), device=device) #Straight   line at constant velocity
  for i in range(int(num_inter_states)+1):
    th_init[i, 0:2] = start_conf*(num_inter_states - i)*1./num_inter_states*1. + goal_conf * i*1./num_inter_states*1. #+ np.array([0., 5.0])
    th_init[i, 2:4] = avg_vel
  return start, goal, th_init

def rrt_star_traj(start_conf, goal_conf, env_params, env, robot, planner_params, obs_params):
  #RRTstar setup
  space = ob.RealVectorStateSpace(2)
  bounds = ob.RealVectorBounds(2)
  bounds.setLow(env_params['x_lims'][0])
  bounds.setHigh(env_params['x_lims'][1])
  space.setBounds(bounds)
  
  init_planner = RRTStar(space, bounds, env, robot, planner_params, obs_params)
  init_path = init_planner.plan(start_conf, goal_conf, 4.0)
  th_init = path_to_traj_avg_vel(init_path, planner_params['total_time_sec'], planner_params['dof'], device)
  start_vel = torch.tensor([[0., 0.]], device=device)
  goal_vel = torch.tensor([[0., 0.]], device=device)
  start = torch.cat((start_conf, start_vel), dim=1)
  goal = torch.cat((goal_conf, goal_vel), dim=1)
  return start, goal, th_init

def is_feas_traj(th, env, eps):
  feas_traj = True
  for i in range(th.shape[0]):
    if not env.is_feasible(th[i, 0:2], eps):
      feas_traj=False
      break
  return feas_traj 

def generate_start_goal(env_params, planner_params, ndims, num, env, robot, obs_params, rrt_star_init=False, fix_start_goal=False):
  if ndims == 2:
    x_lims = env_params['x_lims']
    y_lims = env_params['y_lims']
    eps = robot.get_sphere_radii()[0] + obs_params['epsilon_dist'] + 0.1
    if num == 0 or fix_start_goal:
      #Sample random diagonal problem
      if fix_start_goal: rand_diag = 0
      else: rand_diag = np.random.randint(0,4)
      
      print rand_diag
      start_noise = [0.0,0.0]#np.abs(np.random.normal(size=2))
      goal_noise  = [0.0,0.0]#np.abs(np.random.normal(size=2))
      
      if rand_diag == 0:
        start_conf = torch.tensor([[x_lims[0] + 0.2 + start_noise[0], y_lims[0] + 0.2+start_noise[1]]], device=device) 
        goal_conf = torch.tensor([[x_lims[1] - 0.2 - goal_noise[0], y_lims[1] - 0.2 - goal_noise[1]]], device=device)
      elif rand_diag == 1:
        start_conf = torch.tensor([[x_lims[1] - 0.2 - start_noise[0], y_lims[1] - 0.2 -  start_noise[1]]], device=device)
        goal_conf = torch.tensor([[x_lims[0] + 0.2 + goal_noise[0], y_lims[0] + 0.2 + goal_noise[1]]], device=device)
      elif rand_diag == 2:
        start_conf = torch.tensor([[x_lims[1] - 0.2 - start_noise[0], y_lims[0] + 0.2 + start_noise[1]]], device=device)
        goal_conf = torch.tensor([[x_lims[0] + 0.2 + goal_noise[0], y_lims[1] - 0.2 - goal_noise[1]]], device=device)
      elif rand_diag == 3:
        start_conf = torch.tensor([[x_lims[0] + 0.2 + start_noise[0], y_lims[1] - 0.2 - start_noise[1]]], device=device)
        goal_conf = torch.tensor([[x_lims[1] - 0.2 - goal_noise[0], y_lims[0] + 0.2 + goal_noise[1]]], device=device)
      
      if not env.is_feasible(start_conf[0], eps) or not env.is_feasible(goal_conf[0], eps):
        start_conf, goal_conf = get_random_2d_confs(x_lims, y_lims, env, eps.item())

    else:
      #Choose randomly
      start_conf, goal_conf = get_random_2d_confs(x_lims, y_lims, env, eps.item())
    

    #Generate initial trajectory once you have configuration
    if not rrt_star_init:
      start, goal, th_init = straight_line_traj(start_conf, goal_conf, planner_params['total_time_step'], planner_params['state_dim'])
    else:
      start, goal, th_init = rrt_star_traj(start_conf, goal_conf, env_params, env, robot, planner_params, obs_params)
    

  return start, goal, th_init



def generate_trajs_and_save(folder, num_envs, probs_per_env, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, out_folder_name, rrt_star_init=False, fix_start_goal=False):
  for i in xrange(num_envs):
    if env_data['dim'] == 2:
      env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
      env = Env2D(env_params)
      if robot_data['type'] == 'point_robot':
        robot = PointRobot2D(robot_data['sphere_radius'])
      # print   robot.get_sphere_radii()
      im  = plt.imread(folder + "/im_sdf/" +  str(i) + "_im.png")
      sdf = np.load(folder + "/im_sdf/" +  str(i) + "_sdf.npy")
      env.initialize_from_image(im, sdf)
      imp = torch.tensor(im, device=device)
      sdfp = torch.tensor(sdf, device=device)
    
    for j in xrange(probs_per_env):
      planner = DiffGPMP2Planner(gp_params, obs_params, planner_params, optim_params, env_params, robot, use_cuda=use_cuda)
      start, goal, th_init = generate_start_goal(env_params, planner_params, env_data['dim'], j, env, robot, obs_params, rrt_star_init, fix_start_goal)
      th_final,_, err_init, err_final, err_per_iter, err_ext_per_iter, k, time_taken = \
                                                                      planner.forward(th_init.unsqueeze(0), start.unsqueeze(0), goal.unsqueeze(0), imp.unsqueeze(0).unsqueeze(0), sdfp.unsqueeze(0).unsqueeze(0))
      print('Num iterations = %d, Time taken %f'%(k[0], time_taken[0]))

      path_init = []
      path_final = []

      start_np = start.cpu().detach().numpy()[0]
      goal_np = goal.cpu().detach().numpy()[0]
      th_init_np = th_init.cpu().detach().numpy()
      th_final_np = th_final[0].cpu().detach().numpy()
      out_folder = os.path.join(folder, out_folder_name)
      if not os.path.exists(out_folder):
        os.makedirs(out_folder)
      out_path = out_folder + "/" + "env_" + str(i) + "_prob_" + str(j)
      np.savez(out_path, start=start_np, goal=goal_np, th_opt=th_final_np)

  print('Saving meta data')
  with open(os.path.join(folder, "meta.yaml"), 'w') as fp:
    d = {'num_envs': num_envs,
         'probs_per_env': probs_per_env,
         'env_params': env_params,
         'im_size': args.im_size}
    yaml.dump(d, fp)


def generate_opt_trajs(args):
  np.random.seed(args.seed_val)
  torch.manual_seed(args.seed_val)
  data_folder = os.path.abspath(args.data_folder)
  param_file   = os.path.join(data_folder, 'gpmp2_params.yaml')
  robot_file   = os.path.join(data_folder, "robot.yaml")
  env_file     = os.path.join(data_folder, "env_params.yaml")
  train_folder = os.path.join(data_folder, "train")
  test_folder  = os.path.join(data_folder, "test")
  out_folder_name = "opt_trajs_gpmp2"
  env_data, planner_params, gp_params, obs_params, optim_params, robot_data = load_prob_params(param_file, robot_file, env_file)

  #generate training environments
  if args.train:
    generate_trajs_and_save(train_folder, args.num_train_envs, args.probs_per_env, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, out_folder_name, args.rrt_star_init, args.fix_start_goal)
  if args.test:
    generate_trajs_and_save(test_folder, args.num_test_envs, args.probs_per_env, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, out_folder_name, args.rrt_star_init, args.fix_start_goal)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_folder', type=str, required=True, help="Relative path of output folder", default='.')
  parser.add_argument('--num_train_envs', type=int, help='Number of environments.')
  parser.add_argument('--num_test_envs', type=int, help='Number of environments.')
  parser.add_argument('--im_size', type=int, required=True, help='Size of dataset images')
  parser.add_argument('--probs_per_env', type=int, required=True, help='Number of planning problems per environment')
  parser.add_argument('--seed_val', type=int, default=0, help='Random seed for generating dataset')
  parser.add_argument('--rrt_star_init', action='store_true', help='Generate initial trajectory using rrtstar')
  parser.add_argument('--train', action='store_true', help='Generate training data')
  parser.add_argument('--test', action='store_true', help='Generate test data')
  parser.add_argument('--fix_start_goal', action='store_true', help='Fix start and goal for all problems')
  args = parser.parse_args()
  generate_opt_trajs(args)