#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('../..'))
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
import torch
import yaml
import random
try:
  from ompl import base as ob
  from diff_gpmp2.ompl_rrtstar import RRTStar

except ModuleNotFoundError:
  print("ompl not found. rrt_star data generation will not work")
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D
from diff_gpmp2.gpmp2 import GPMP2Planner, DiffGPMP2Planner
from diff_gpmp2.utils.planner_utils import straight_line_traj, path_to_traj_avg_vel
from diff_gpmp2.utils.helpers import load_params
from diff_gpmp2.utils.sdf_utils import bilinear_interpolate
from diff_gpmp2.datasets.utils import sdf_2d, costmap_2d
from obst_generator import generate_rect_obstacle_map, generate_wall_obstacle_map, save_map_image

datasets = {'tar_pit': 0,'forest': 1,'multi_obs': 2,'passage': 3, 'mixed_clutter': 4}


def get_tarpit(im_size, start_goal_dist, obstacle_sep, seed_val):
  n_obs = np.random.randint(5,8)  
  w_min = int(im_size/10); h_min = int(im_size/10)
  w_max = w_min + 1; h_max = h_min + 1
  fact = 0.15#np.random.uniform(0, 0.2)
  # facty = np.random.uniform(0, 0.5)
  startx = int(fact*im_size); starty=int(fact*im_size)
  endx   =  int(startx + 0.5*im_size); endy=int(starty + 0.5*im_size)
  # endx =  int((fact+0.3)*im_size); endy=int((fact+0.3)*im_size)
  obs_map = generate_rect_obstacle_map(map_dim,n_obs,start_pts,goal_pts, w_min, w_max, h_min, h_max, startx, starty, endx,endy, patch_size=start_goal_dist, patch_size_obs=obstacle_sep, seed=seed_val)
  return obs_map

def get_forest(im_size, start_goal_dist, obstacle_sep, seed_val):
  #Many small obstacles
  #n_obs = min(int(np.random.normal(30, 15)), 50)#np.random.randint(25, 45)  
  n_obs = np.random.randint(23, 45)
  w_min = int(im_size/30); h_min = int(im_size/30)
  w_max = w_min+1; h_max = h_min+1
  startx = 0.0; starty=0.0
  endx =  im_size-1; endy=im_size-1
  # obs_map = generate_rect_obstacle_map(map_dim,n_obs,start_pts,goal_pts, w_min, w_max, h_min, h_max, startx, starty, endx,endy, patch_size=1.5*patch_size, patch_size_obs=2.0*patch_size, seed=seed_val)
  obs_map = generate_rect_obstacle_map(map_dim,n_obs,start_pts,goal_pts, w_min, w_max, h_min, h_max, startx, starty, endx,endy, patch_size=start_goal_dist, patch_size_obs=obstacle_sep, seed=seed_val)
  return obs_map



def get_multi_obs(im_size, start_goal_dist, obstacle_sep, seed_val):
   #Vert few large obstacles
  n_obs = np.random.randint(2,5)
  w_min = int(args.im_size/8); h_min = int(args.im_size/8)
  w_max = w_min + 10; h_max = h_min + 10
  fact = 0.1#np.random.uniform(0, 0.2)
  # facty = np.random.uniform(0, 0.5)
  startx = int(fact*im_size); starty=int(fact*im_size)
  endx =  int((1.0-fact)*im_size); endy=int((1.0-fact)*im_size) 
  # obs_map = generate_rect_obstacle_map(map_dim,n_obs,start_pts,goal_pts, w_min, w_max, h_min, h_max, startx, starty, endx,endy, patch_size=1.5*patch_size, patch_size_obs=3.0*patch_size, seed=args.seed_val)
  obs_map = generate_rect_obstacle_map(map_dim,n_obs,start_pts,goal_pts, w_min, w_max, h_min, h_max, startx, starty, endx,endy, patch_size=start_goal_dist, patch_size_obs=obstacle_sep, seed=seed_val)
  return obs_map


def get_passage(im_size, start_goal_dist, passage_size, seed_val):
  n_obs = 1
  w_min = int(im_size/5); gw_min = int(passage_size)
  w_max = w_min + 10; gw_max = gw_min + 1
  startx = int(0.15*im_size); gap_y = 0    
  obs_map = generate_wall_obstacle_map(map_dim,n_obs,start_pts,goal_pts, w_min, w_max, gw_min, gw_max, startx, gap_y, patch_size=start_goal_dist, seed=seed_val)
  return obs_map

def get_mixed_clutter(im_size, patch_size, seed_val):
  d = np.random.choice([0, 1, 2])
  if d == 0:
    print('Sample tarpit')
    obs_map = get_tarpit(im_size, patch_size, seed_val)
  elif d == 1:
    print('Sample forest')
    obs_map = get_forest(im_size, patch_size, seed_val)
  elif d == 2:
    print('Sample multi obs')
    obs_map = get_multi_obs(im_size, patch_size, seed_val)
  return obs_map

def rrt_star_traj(start_conf, goal_conf, env_params, env, robot, planner_params, obs_params):
  #RRTstar setup
  space = ob.RealVectorStateSpace(2)
  bounds = ob.RealVectorBounds(2)
  bounds.setLow(env_params['x_lims'][0])
  bounds.setHigh(env_params['x_lims'][1])
  space.setBounds(bounds)
  init_planner = RRTStar(space, bounds, env, robot, planner_params, 0.0)
  init_path = init_planner.plan(start_conf, goal_conf, 4.0)
  th_init = path_to_traj_avg_vel(init_path, planner_params['total_time_sec'], planner_params['dof'])
  return th_init

parser = argparse.ArgumentParser()
parser.add_argument('--out_folder', type=str, default='./dataset_2d_',  help="Relative path of output folder")
parser.add_argument('--dataset_type', type=str, help='Type of dataset. See README.md for types available.')
parser.add_argument('--im_size', type=int, default=128, help='Size of dataset images')
parser.add_argument('--num_train', type=int, default=500, help='Number of training samples to generate.')
parser.add_argument('--num_test', type=int, default=200, help='Number of training samples to generate.')
parser.add_argument('--probs_per_env', type=int, default=1, help='Number of problems per environment')
parser.add_argument('--seed_val', type=int, default=0, help='Random seed for generating dataset')
parser.add_argument('--rrtstar_init', action='store_true', help='Initialize path using RRTstar')
parser.add_argument('--render', action='store_true', help='Render while generating data')
parser.add_argument('--step', action='store_true', help='Step through each data point')
args = parser.parse_args()
np.random.seed(args.seed_val)
random.seed(args.seed_val)
torch.manual_seed(args.seed_val)
torch.set_default_tensor_type(torch.DoubleTensor)

out_folder   = os.path.abspath(args.out_folder)
param_file   = os.path.join(out_folder, 'gpmp2_params.yaml')
robot_file   = os.path.join(out_folder, "robot.yaml")
env_file     = os.path.join(out_folder, "env_params.yaml")
train_folder = os.path.join(out_folder, "train")
test_folder  = os.path.join(out_folder, "test")
out_folder_name = "opt_trajs_gpmp2"

out_folder_train = os.path.join(train_folder, out_folder_name)
out_folder_test = os.path.join(train_folder, out_folder_name)
if not os.path.exists(out_folder_train):
  os.makedirs(out_folder_train)
if not os.path.exists(out_folder_test):
  os.makedirs(out_folder_test)
  
env_data, planner_params, gp_params, obs_params, optim_params, robot_data = load_params(param_file, robot_file, env_file, torch.device('cpu'))
print(env_data, planner_params, obs_params, gp_params)

#parameters
dist_factor = 0.6 #Distance between start goal must be atleast this factor of max diagonal distance 
safety_distance = (obs_params['epsilon_dist'] + robot_data['sphere_radius'][0]).double() #start and goal must be safety distance away from obstacles
robot_radius = robot_data['sphere_radius'][0].double()
x_min = env_data['x_lims'][0]
y_min = env_data['y_lims'][0]
x_max = env_data['x_lims'][1]
y_max = env_data['y_lims'][1]

sgmax_x = x_max - 1.0
sgmin_x = x_min + 1.0
sgmax_y = y_max - 1.0
sgmin_y = y_min + 1.0

cell_size         = (x_max - x_min)/args.im_size*1.0
start_goal_dist   = dist_factor* torch.sqrt(torch.tensor((x_max - x_min) ** 2 + (y_max - y_min) ** 2))
patch_size_safety = int(np.ceil(safety_distance.item()/cell_size*1.0)) #Padding size for sdf and min distance of start goal from obstacles
patch_size_robot  = int(np.ceil(robot_radius.item()/cell_size*1.0))

#Generate and save training data
dataset_number = datasets[args.dataset_type]
map_dim=(args.im_size, args.im_size)

env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims'], 'padlen': patch_size_safety}
env = Env2D(env_params)
robot = PointRobot2D(robot_data['sphere_radius'])


env_number = 0
while env_number < args.num_train:
  print('Creating env number %d'%env_number)
  #Sample start and goal in meters
  far_enough=False
  while not far_enough:
    print("Sampling valid start and goal")
    start_xs = (sgmax_x - sgmin_x) * torch.rand(args.probs_per_env, 1) + sgmin_x 
    start_ys = (sgmax_y - sgmin_y) * torch.rand(args.probs_per_env, 1) + sgmin_y 
    goal_xs  = (sgmax_x - sgmin_x) * torch.rand(args.probs_per_env, 1) + sgmin_x 
    goal_ys  = (sgmax_y - sgmin_y) * torch.rand(args.probs_per_env, 1) + sgmin_y
    start_confs = torch.cat((start_xs, start_ys), dim=-1) 
    goal_confs = torch.cat((goal_xs, goal_ys), dim=-1)
    far_enough_vec = torch.where(torch.norm(goal_confs-start_confs,2) >= start_goal_dist, torch.tensor(1.0), torch.tensor(0.0))
    far_enough = torch.sum(far_enough_vec) == args.probs_per_env

  start_vels = torch.zeros_like(start_confs)
  goal_vels  = torch.zeros_like(goal_confs)
  start_states = torch.cat((start_confs, start_vels), dim=-1)
  goal_states  = torch.cat((goal_confs, goal_vels), dim=-1)
  #convert to pixel coordinates
  orig_pix  = torch.tensor([-x_min*1.0/cell_size, -y_min*1.0/cell_size])
  start_pts = torch.div(start_confs,cell_size)
  goal_pts  = torch.div(goal_confs,cell_size)
  start_pts[:,0] = orig_pix[0] + start_pts[:,0]
  goal_pts[:,0]  = orig_pix[0] + goal_pts[:,0]
  start_pts[:,1] = orig_pix[1] - start_pts[:,1]
  goal_pts[:,1]  = orig_pix[1] - goal_pts[:,1]

  print("Creating obstacle map")
  if dataset_number == 0:
    obs_map = get_tarpit(args.im_size, patch_size_robot + 2.0*patch_size_safety, 0.0, args.seed_val)

  elif dataset_number == 1:
    obs_map = get_forest(args.im_size, 3.0*patch_size_robot, 3*patch_size_robot, args.seed_val)

  elif dataset_number == 2:
    obs_map = get_multi_obs(args.im_size, patch_size_safety + patch_size_robot, 2.0*(patch_size_robot + patch_size_safety), args.seed_val)

  elif dataset_number == 3:
    obs_map = get_passage(args.im_size, 3.0*patch_size_robot, 4.0*patch_size_robot, args.seed_val)

  elif dataset_number == 4:
    obs_map = get_mixed_clutter(args.im_size, 0.8*patch_size_safety, 2.0*(patch_size_robot + patch_size_safety), args.seed_val)

  obs_map = torch.tensor(obs_map)
  obs_sdf = torch.tensor(sdf_2d(obs_map, padlen=0, res=cell_size))
  obs_costmap = costmap_2d(obs_sdf, safety_distance)
  env.initialize_from_image(obs_map, obs_sdf)
  if args.render:
    env.close_plot()
    env.initialize_plot(start_confs[0], goal_confs[0])
    # env.plot_signed_distance_transform()
    # env.plot_costmap(safety_distance)
    plt.show(block=False)
  planner = DiffGPMP2Planner(gp_params, obs_params, planner_params, optim_params, env_params, robot, batch_size=1)
  
  try:
    for j in range(args.probs_per_env):
      start = start_states[j]; goal = goal_states[j]
      if args.rrtstar_init:
        th_init = rrt_star_traj(start_confs[j].unsqueeze(0), goal_confs[j].unsqueeze(0), env_params, env, robot, planner_params, obs_params)
      else:
        th_init = straight_line_traj(start_confs[j], goal_confs[j], planner_params['total_time_sec'], planner_params['total_time_step'], planner_params['dof']) 
      #Run GPMP2 on top
      print('Running GPMP2')
      th_final, _, _, _, _, _, _, _ = planner.forward(th_init.unsqueeze(0), start.unsqueeze(0), goal.unsqueeze(0), obs_map.unsqueeze(0).unsqueeze(0), obs_sdf.unsqueeze(0).unsqueeze(0))
      #th_final = th_init.unsqueeze(0)
      start_np = start.cpu().detach().numpy()
      goal_np  = goal.cpu().detach().numpy()
      th_init_np = th_init.cpu().detach().numpy()
      th_final_np = th_final[0].cpu().detach().numpy()

      if args.render:
        path_init = [th_init_np[k, 0:planner_params['dof']] for k in range(planner_params['total_time_step']+1)]
        path_final = [th_final_np[k, 0:planner_params['dof']] for k in range(planner_params['total_time_step']+1)]
        env.plot_edge(path_init, color='red', linestyle='--', markersize=robot_data['sphere_radius'][0]/cell_size)
        env.plot_edge(path_final, markersize=robot_data['sphere_radius'][0]/cell_size)
        plt.show(block=False)
      out_path = out_folder_train + "/" + "env_" + str(env_number) + "_prob_" + str(j)
      np.savez(out_path, start=start_np, goal=goal_np, th_opt=th_final_np, th_init=th_init_np)  
      
      for state in th_final[0]:
        state_pos = state[0:2]
        state_pos = state_pos.reshape(1,1,state_pos.shape[0])
        d_obs, _ = bilinear_interpolate(obs_sdf.unsqueeze(0).unsqueeze(0), state_pos, cell_size, env_params['x_lims'], env_params['y_lims'], False)
        if d_obs <= robot_radius:
          raise ValueError("Trajectory is in collision")
          
      print('Trajectory not in collision')

      if args.step: input('Enter to generate next data point')

    save_map_image(obs_map, start_pts, goal_pts, dir='{}/{}'.format(train_folder,'im_sdf'), name=str(env_number)+'_im')
    out_sdf = '{}/{}/{}'.format(train_folder, 'im_sdf', str(env_number)+'_sdf')
    np.save(out_sdf, obs_sdf)
    env_number = env_number + 1

  except Exception as err:
    print(err)
    if args.step: input('Enter to generate next data point')
           

#Save training metadata
meta = {'num_envs': args.num_train,
        'probs_per_env': args.probs_per_env,
        'im_size': args.im_size,
        'env_params': env_params
        }
with open(os.path.join(train_folder, "meta.yaml"), 'w') as fp:
  yaml.dump(meta, fp)

#Generate test data

#Save test data
meta = {}
plt.show()

