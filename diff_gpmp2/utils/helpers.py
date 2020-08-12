import torch
import yaml
import numpy  as np


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_params(param_file, robot_file, env_file, device):
  with open(param_file, 'r') as fp:
    planner_data = yaml.load(fp)
  with open(env_file, 'r') as fp:
    env_data = yaml.load(fp)
  with open(robot_file, 'r') as fp:
    robot_data = yaml.load(fp)
  
  planner_params = planner_data['gpmp2']['planner_params']
  gp_params = planner_data['gpmp2']['gp_params']
  obs_params = planner_data['gpmp2']['obs_params']
  optim_params = planner_data['gpmp2']['optim_params']
  gp_params['Q_c_inv'] = torch.tensor(gp_params['Q_c_inv'], device=device)
  gp_params['K_s'] = torch.tensor(gp_params['K_s'], device=device)
  gp_params['K_g'] = torch.tensor(gp_params['K_g'], device=device)
  nonhol = planner_params['non_holonomic'] if 'non_holonomic' in planner_params else False
  use_vel = planner_params['use_vel_limits'] if 'use_vel_limits' in planner_params else False
  if nonhol: gp_params['K_d'] = torch.tensor(gp_params['K_d'], device=device)  
  if use_vel: gp_params['K_v'] = torch.tensor(gp_params['K_v'], device=device)  
  obs_params['cost_sigma'] = torch.tensor(obs_params['cost_sigma'], device=device)
  obs_params['epsilon_dist'] = torch.tensor(obs_params['epsilon_dist'], device=device)
  robot_data['sphere_radius'] = torch.tensor(robot_data['sphere_radius'], device=device)


  return env_data, planner_params, gp_params, obs_params, optim_params, robot_data

def load_params_learn(param_file, robot_file, env_file, learn_params_file, device):
  with open(param_file, 'r') as fp:
    planner_data = yaml.load(fp)
  with open(env_file, 'r') as fp:
    env_data = yaml.load(fp)
  with open(robot_file, 'r') as fp:
    robot_data = yaml.load(fp)
  with open(learn_params_file, 'r') as fp:
    learn_params = yaml.load(fp)

  planner_params = planner_data['gpmp2']['planner_params']
  gp_params = planner_data['gpmp2']['gp_params']
  obs_params = planner_data['gpmp2']['obs_params']
  optim_params = planner_data['gpmp2']['optim_params']  
  gp_params['Q_c_inv'] = torch.tensor(gp_params['Q_c_inv'], device=device)
  gp_params['K_s'] = torch.tensor(gp_params['K_s'], device=device)
  gp_params['K_g'] = torch.tensor(gp_params['K_g'], device=device)
  nonhol = planner_params['non_holonomic'] if 'non_holonomic' in planner_params else False
  use_vel = planner_params['use_vel_limits'] if 'use_vel_limits' in planner_params else False
  if nonhol: gp_params['K_d'] = torch.tensor(gp_params['K_d'], device=device)  
  if use_vel: gp_params['K_v'] = torch.tensor(gp_params['K_v'], device=device)  
  obs_params['cost_sigma'] = torch.tensor(obs_params['cost_sigma'], device=device)
  obs_params['epsilon_dist'] = torch.tensor(obs_params['epsilon_dist'], device=device)
  robot_data['sphere_radius'] = torch.tensor(robot_data['sphere_radius'], device=device)

  return env_data, planner_params, gp_params, obs_params, optim_params, robot_data, learn_params