import argparse
import os, sys
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
plt.rcParams['axes.facecolor'] = 'white'
import numpy as np
import pprint
import time
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.utils import make_grid
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D
from diff_gpmp2.gpmp2 import DiffGPMP2Planner
from diff_gpmp2.gpmp2.gp import GPFactor
from diff_gpmp2.gpmp2.obstacle import ObstacleFactor
from diff_gpmp2.utils.helpers import load_params 
from diff_gpmp2.utils import sdf_utils 
from diff_gpmp2.utils.planner_utils import straight_line_traj, check_convergence, check_convergence_batch, smoothness_metrics, collision_metrics 
from diff_gpmp2.utils.learn_utils import torch_optimizer, torch_loss 
from datasets import PlanningDataset, PlanningDatasetMulti

np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
pp = pprint.PrettyPrinter()
torch.set_default_tensor_type(torch.DoubleTensor)
pfiletype = '.yaml'

  
def run_validation(args, sigma_obs_arr):
  use_cuda = torch.cuda.is_available() if args.use_cuda else False
  device = torch.device('cuda') if use_cuda else torch.device('cpu')
  np.random.seed(args.seed_val)
  torch.manual_seed(args.seed_val)

  dataset_folders   = [os.path.abspath(folder) for folder in args.dataset_folders]
  plan_param_file   = os.path.join(dataset_folders[0], args.plan_param_file +pfiletype)
  robot_param_file  = os.path.join(dataset_folders[0], args.robot_param_file+pfiletype)
  env_param_file    = os.path.join(dataset_folders[0], args.env_param_file  +pfiletype)  

  env_data, planner_params, gp_params, obs_params,\
  optim_params, robot_data = load_params(plan_param_file, robot_param_file, env_param_file, device)
  
  dataset = PlanningDatasetMulti(dataset_folders, mode='train', num_envs=1000,
                                   num_env_probs=1, 
                                   label_subdir='opt_trajs_gpmp2')
  
    
  # idxs = np.random.choice(len(dataset), args.num_envs, replace=False) if args.num_envs < len(dataset) else xrange(0, len(dataset))
  idxs = xrange(args.num_envs)
  print idxs
  env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
  if robot_data['type'] == 'point_robot':
    robot = PointRobot2D(robot_data['sphere_radius'], use_cuda=args.use_cuda)
  batch_size = 1#learn_params['optim']['batch_size']
  
  #To be used for calculating metrics later
  dt = planner_params['total_time_sec']*1.0/planner_params['total_time_step']*1.0
  gpfactor = GPFactor(planner_params['dof'], dt, planner_params['total_time_step'])
  obsfactor = ObstacleFactor(planner_params['state_dim'], planner_params['total_time_step'], 0.0, env_params, robot)
  dof = planner_params['dof']
  use_vel_limits = planner_params['use_vel_limits'] if 'use_vel_limits' in planner_params else False

  results_dict = {}
  for sigma_obs in sigma_obs_arr:
    print('Curr Sigma = ', sigma_obs)
    obs_params['cost_sigma'] = torch.tensor(sigma_obs, device=device)
    planner = DiffGPMP2Planner(gp_params, obs_params, planner_params, optim_params, env_params, robot, learn_params=None, batch_size=batch_size, use_cuda=args.use_cuda)
    planner.to(device)
    planner.eval()

    # criterion = torch_loss(learn_params['optim']['criterion'], reduction=learn_params['optim']['loss_reduction'])
    
    valid_task_loss_per_iter = []
    valid_cost_per_iter  = []
    valid_num_iters = []
    valid_gp_error = []
    valid_avg_vel = []
    valid_avg_acc = []
    valid_avg_jerk = []
    valid_in_coll = []
    valid_avg_penetration = []
    valid_max_penetration = []
    valid_coll_intensity = []
    valid_constraint_violation = []
    

    with torch.no_grad():
      for i in idxs:
        sample = dataset[i]
        # print('Environment idx = %d'%i)
        im     = sample['im'].to(device)
        sdf    = sample['sdf'].to(device)
        start  = sample['start'].to(device)
        goal   = sample['goal'].to(device)
        th_opt = sample['th_opt'].to(device) 
        start_conf = start[0, 0:dof]
        goal_conf  = goal[0, 0:dof]
        th_init =  straight_line_traj(start_conf, goal_conf, planner_params['total_time_sec'], planner_params['total_time_step'], dof, device)
        j = 0
        
        th_curr = th_init.unsqueeze(0)
        dtheta = torch.zeros_like(th_curr)
        eps_traj = torch.zeros(planner_params['total_time_step']+1, robot.nlinks, 1)
        eps_traj = eps_traj.unsqueeze(0).repeat(th_curr.shape[0],1,1,1)
        obsfactor.set_eps(eps_traj.unsqueeze(0))
        curr_hidden=None

        cost_per_iter  = []
        task_loss_per_iter = []
        th_best = None
        best_task_loss = np.inf
        if args.render:
          th_init_np = th_init.cpu().detach().numpy()
          th_opt_np = th_opt.cpu().detach().numpy()
          env = Env2D(env_params)
          env.initialize_from_image(im[0], sdf[0])
          path_init = [th_init_np[i, 0:dof] for i in xrange(planner_params['total_time_step']+1)]
          path_opt  = [th_opt_np[i, 0:dof] for i in xrange(planner_params['total_time_step']+1)]
          env.initialize_plot(start_conf.cpu().numpy(), goal_conf.cpu().numpy())
          env.plot_signed_distance_transform()
          raw_input('Enter to start ...')
          plt.show(block=False)
        
        while True:
          # print("Current iteration = %d"%j)
          if args.render:
            th_curr_np = th_curr.cpu().detach().numpy()
            path_curr = [th_curr_np[0, i, 0:dof] for i in xrange(planner_params['total_time_step']+1)]            
            if j > 0: env.clear_edges()
            env.plot_edge(path_curr, color='blue')#, linestyle='-', linewidth=0.01*j , alpha=1.0-(1.0/(j+0.0001)) )
            plt.show(block=False)
            time.sleep(0.002)
            if args.step:
              raw_input('Press enter for next step')
          
          dtheta, curr_hidden, err_old, err_ext_old, qc_inv_traj, obscov_inv_traj, eps_traj = planner.step(th_curr, start.unsqueeze(0), goal.unsqueeze(0), im.unsqueeze(0), sdf.unsqueeze(0), dtheta, curr_hidden)
          err_sg, err_gp, err_obs = planner.unweighted_errors_batch(th_curr, sdf.unsqueeze(0))
          task_loss = err_gp + args.obs_lambda*err_obs
          
          #We only keep the best trajectory so far
          if task_loss.item() < best_task_loss:
            th_best = th_curr
            best_task_loss = task_loss.item()

          task_loss_per_iter.append(task_loss.item())
          cost_per_iter.append(err_old.item())
          
          th_old  = th_curr
          th_curr = th_curr + dtheta
          th_new  = th_curr
          err_new = planner.error_batch(th_curr, sdf.unsqueeze(0)).item()
          err_ext_new = planner.error_ext_batch(th_curr, sdf.unsqueeze(0)).item()

          err_delta = err_new - err_old[0]
          err_ext_delta = err_ext_new - err_ext_old[0]
          # print('|dtheta| = %f, err = %f, err_ext = %f, err_delta = %f,\
          #        |qc_inv| = %f, |obscov_inv| = %f'%(torch.norm(dtheta), err_old[0], err_delta, err_ext_delta,\
          #                                           torch.norm(qc_inv_traj, p='fro', dim=(2,3)).mean(),\
          #                                           torch.norm(obscov_inv_traj, p='fro', dim=(2,3)).mean()))
          
          j = j + 1
          if check_convergence(dtheta, j, torch.tensor(err_delta), optim_params['tol_err'], optim_params['tol_delta'], optim_params['max_iters']):
            # print('Converged')
            break
        th_final = th_best
        #########################METRICS##########################################
        # print th_final
        avg_vel, avg_acc, avg_jerk = smoothness_metrics(th_final[0], planner_params['total_time_sec'], planner_params['total_time_step'])
        gp_error, _ , _ = gpfactor.get_error(th_final) 
        obs_error, _    = obsfactor.get_error(th_final, sdf.unsqueeze(0))
        mse_gp = torch.mean(torch.sum(gp_error**2, dim=-1))
        in_coll, avg_penetration, max_penetration, coll_int = collision_metrics(th_final[0], obs_error[0], planner_params['total_time_sec'], planner_params['total_time_step'])
        print('Trajectory in collision = ', in_coll)
        # print('MSE GP = {}, Average velocity = {}, average acc = {}, avg jerk=  {}'.format(mse_gp, avg_vel, avg_acc, avg_jerk))
        # print('In coll = {}, average penetration = {}, max penetration = {}, collision intensity =  {}'.format(in_coll, 
                                                                                                               # avg_penetration, 
                                                                                                               # max_penetration, 
                                                                                                               # coll_int))
        constraint_violation = 0.0
        if use_vel_limits: #planner_params['use_vel_limits']:
          v_x_lim = gp_params['v_x']
          v_y_lim = gp_params['v_y']
          for i in xrange(th_final.shape[1]):
            s = th_final[0][i]
            v_x = s[2]; v_y = s[3]
            if torch.abs(v_x) <= v_x_lim and torch.abs(v_y) <= v_y_lim:
              continue
            else:
              constraint_violation = constraint_violation + 1.0
              print ('Constraint violatrion!!!!!')
        constraint_violation = constraint_violation/(th_final.shape[1]*1.0)
          


        valid_gp_error.append(mse_gp.item())
        valid_avg_vel.append(avg_vel.item())
        valid_avg_acc.append(avg_acc.item())
        valid_avg_jerk.append(avg_jerk.item())
        valid_in_coll.append(in_coll)
        valid_avg_penetration.append(avg_penetration.item())
        valid_max_penetration.append(max_penetration.item())
        valid_coll_intensity.append(coll_int)
        valid_constraint_violation.append(constraint_violation)

        err_sg, err_gp, err_obs = planner.unweighted_errors_batch(th_final, sdf.unsqueeze(0))
        task_loss = err_sg + err_gp + args.obs_lambda*err_obs
        
        task_loss_per_iter.append(task_loss.item())
        cost_per_iter.append(planner.error_batch(th_final, sdf.unsqueeze(0)).item())
        
        valid_task_loss_per_iter.append(task_loss_per_iter)
        valid_cost_per_iter.append(cost_per_iter)
        valid_num_iters.append(j)

        if args.render:
          th_final_np = th_final.cpu().detach().numpy()
          path_final = [th_final_np[0][i, 0:dof] for i in xrange(planner_params['total_time_step']+1)]
          env.clear_edges()
          env.plot_edge(path_final)#, linewidth=0.1*j)
          plt.show(block=False)
          raw_input('Press enter for next env')
          env.close_plot()      

    results_dict_sig = {}  
    results_dict_sig['num_iters']            = valid_num_iters
    # results_dict_sig['cost_per_iter']        = valid_cost_per_iter
    results_dict_sig['gp_mse']               = valid_gp_error
    results_dict_sig['avg_vel']              = valid_avg_vel
    results_dict_sig['avg_acc']              = valid_avg_acc
    results_dict_sig['avg_jerk']             = valid_avg_jerk
    results_dict_sig['in_collision']         = valid_in_coll
    results_dict_sig['avg_penetration']      = valid_avg_penetration
    results_dict_sig['max_penetration']      = valid_max_penetration
    results_dict_sig['coll_intensity']       = valid_coll_intensity
    # results_dict_sig['task_loss_per_iter']   = valid_task_loss_per_iter
    results_dict_sig['constraint_violation'] = valid_constraint_violation
    results_dict[str(sigma_obs)] = results_dict_sig

    print('Avg unsolved = ', np.mean(valid_in_coll))

  print('Dumping results')
  filename = 'sensitivity_results.yaml'
  # else: filename = args.model_file+"_valid_results.yaml"
  with open(dataset_folders[0] + '/' + filename, 'w') as fp:
    yaml.dump(results_dict, fp)
  # plot_results(results_dict)



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='PyTorch Example')
  parser.add_argument('--dataset_folders', type=str, nargs='+', required=True, help='Folder with training files')
  parser.add_argument('--plan_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--robot_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--env_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--render', action='store_true', help='Plots the test results')
  parser.add_argument('--step', action='store_true', help='Render intermediate trajectories')
  parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA')
  parser.add_argument('--seed_val', type=int, default=1234, help='Seed value shared by numpy as pytorch')
  parser.add_argument('--num_envs', type=int, default=100, help='Number of files to use for testing dataset')
  parser.add_argument('--obs_lambda', type=float, default=1.0, help='Weight for obstacle cost in task loss')
  args = parser.parse_args()
  sigma_obs_arr = [0.01,0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 5.0, 10.0]#
  run_validation(args, sigma_obs_arr)



