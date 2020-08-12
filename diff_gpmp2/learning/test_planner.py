import argparse
import os, sys
sys.path.insert(0, "../..")
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
from diff_gpmp2.utils.helpers import load_params_learn 
from diff_gpmp2.utils import sdf_utils 
from diff_gpmp2.utils.planner_utils import straight_line_traj, check_convergence, check_convergence_batch, smoothness_metrics, collision_metrics 
from diff_gpmp2.utils.learn_utils import torch_optimizer, torch_loss 
from datasets import PlanningDataset, PlanningDatasetMulti

np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
pp = pprint.PrettyPrinter()
torch.set_default_tensor_type(torch.DoubleTensor)
pfiletype = '.yaml'

  
def run_validation(args):
  use_cuda = torch.cuda.is_available() if args.use_cuda else False
  device = torch.device('cuda') if use_cuda else torch.device('cpu')
  np.random.seed(args.seed_val)
  torch.manual_seed(args.seed_val)
  
  dataset_folders   = [os.path.abspath(folder) for folder in args.dataset_folders]
  input_folder      = os.path.abspath(args.in_folder)
  plan_param_file   = os.path.join(input_folder, args.plan_param_file +pfiletype)
  robot_param_file  = os.path.join(input_folder, args.robot_param_file+pfiletype)
  env_param_file    = os.path.join(input_folder, args.env_param_file  +pfiletype)  
  learn_param_file  = os.path.join(input_folder, args.learn_param_file+pfiletype)
  model_folder      = os.path.join(input_folder, "models") #folder to store learnt models in 
  results_folder    = os.path.join(input_folder, "results")
#  if args.valid_idxs_file is not None:
#    idxs_file        = os.path.join(results_folder, 'train_val_split_'+args.valid_idxs_file+pfiletype) 
#  else: 
  idxs_file        = os.path.join(results_folder, 'train_val_split'+pfiletype) 

  if not os.path.exists(model_folder): os.makedirs(model_folder)
  if not os.path.exists(results_folder): os.makedirs(results_folder)
  env_data, planner_params, gp_params, obs_params,\
  optim_params, robot_data, learn_params = load_params_learn(plan_param_file, robot_param_file, env_param_file, 
                                                             learn_param_file, device)
  
  if args.validation:
    print learn_params['data']['num_train_envs']
    dataset = PlanningDatasetMulti(dataset_folders, mode='train', num_envs= learn_params['data']['num_train_envs'],
                                   num_env_probs=learn_params['data']['num_train_env_probs'], 
                                   label_subdir='opt_trajs_'+learn_params['data']['expert'])
  
    with open(idxs_file , 'r') as fp:
      train_val_split = yaml.load(fp)
    if args.test_overfit:
      print 'Testing overfitting using training data'
      idxs = train_val_split['train_idxs']
    else: idxs = train_val_split['valid_idxs']
  else:
    dataset = PlanningDatasetMulti(dataset_folders, mode='test', label_subdir='opt_trajs_'+learn_params['data']['expert'])
    idxs = range(len(dataset))
  if isinstance(idxs[0], basestring):
    idxs = [int(x) for x in idxs[0].split(" ")]
  print idxs
  idx_f = []
  if args.valid_idxs_file == 'tarpit':
    for i in idxs:
      if i < 2500:
        idx_f.append(i)
  elif args.valid_idxs_file == 'forest':
    for i in idxs:
      if i >= 2500:
        idx_f.append(i)
  else:
    for i in idxs:
      idx_f.append(i)
  print idx_f
#  nt=0
#  for i in idxs:
#    if i < 750:
#      nt += 1
#  print(nt)
  #print(len(idxs[idxs<750]))
  #raw_input('..')
  env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
  if robot_data['type'] == 'point_robot_2d':
    robot = PointRobot2D(robot_data['sphere_radius'], use_cuda=args.use_cuda)
  learn_params['im_size'] = dataset.meta_data[0]['im_size']
  if args.use_static_covs:
    planner = DiffGPMP2Planner(gp_params, obs_params, planner_params, optim_params, env_params, robot, None, use_cuda=args.use_cuda)
  else:  
    planner = DiffGPMP2Planner(gp_params, obs_params, planner_params, optim_params, env_params, robot, learn_params, use_cuda=args.use_cuda)
    if not args.model_file:
      raise ValueError, 'Model file not specified'
    planner.load_state_dict(torch.load(model_folder+"/"+args.model_file))
    planner.to(device)
    planner.eval()

  criterion = torch_loss(learn_params['optim']['criterion'], reduction=learn_params['optim']['loss_reduction'])
  
  if learn_params['model']['type'] == 'rnn_gru' or learn_params['model']['type'] == 'rnn_lstm':
    model_type = 'recurrent'
  else:
    model_type = 'feed_forward'

  valid_task_loss_per_iter = []  
  valid_cost_per_iter = []
  valid_ext_cost_per_iter = []
  valid_pos_loss_per_iter = []
  valid_vel_loss_per_iter = []
  valid_loss_per_iter = []
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
  
  #To be used for calculating metrics later
  dt = planner_params['total_time_sec']*1.0/planner_params['total_time_step']*1.0
  gpfactor = GPFactor(planner_params['dof'], dt, planner_params['total_time_step'])
  obsfactor = ObstacleFactor(planner_params['state_dim'], planner_params['total_time_step'], 0.0, env_params, robot)
  dof = planner_params['dof']
  use_dtheta = learn_params['dgpmp2']['dtheta_predict'] if 'dtheta_predict' in learn_params['dgpmp2'] else False
  use_vel_limits = planner_params['use_vel_limits'] if 'use_vel_limits' in planner_params else False
  batch_size = 1#learn_params['optim']['batch_size']
  fixed_conv = learn_params['dgpmp2']['fixed_conv'] if 'fixed_conv' in learn_params['dgpmp2'] else False
  with torch.no_grad():
    #Visualize the learned weights 
    if args.viz_ftrs and not args.use_static_covs:
      conv_weights = [p.weight.data for n, p in planner.learn_module._modules.items() if isinstance(p, nn.modules.conv.Conv2d)]
      num_subplots = len(conv_weights)
      for ip, vp in enumerate(xrange(num_subplots)):
        vp = vp + 1
        ax_curr = plt.subplot(num_subplots, 1, vp)
        if conv_weights[ip].shape[1] > 2:
          grid = make_grid(conv_weights[ip][:, 0:3, :, :], nrow=16, padding=1, pad_value=1)
        else:
          grid = make_grid(conv_weights[ip][:, 0:1, :, :], nrow=16, padding=1, pad_value=1)
        ax_curr.imshow(grid.numpy().transpose((1,2,0)))
        ax_curr.set_title('Conv layer %d weights'%ip)
      plt.show(block=False)    
    
    # print idxs    # for i in [idxs[0]]:
    for i in idx_f:
      sample = dataset[i]
      print('Environment idx = %d'%i)
      im     = sample['im'].to(device)
      sdf    = sample['sdf'].to(device)
      start  = sample['start'].to(device)
      goal   = sample['goal'].to(device)
      th_opt = sample['th_opt'].to(device) 
      start_conf = start[0, 0:planner_params['dof']]
      goal_conf  = goal[0, 0:planner_params['dof']]
      th_init =  straight_line_traj(start_conf, goal_conf, planner_params['total_time_sec'], planner_params['total_time_step'], planner_params['dof'], device)
      j = 0
      #Predict conv features
      conv_out = None
      if fixed_conv and not args.use_static_covs:
        data = torch.cat((im.unsqueeze(0), sdf.unsqueeze(0)), dim=1)
        conv_out, _ = planner.learn_module_conv(data)
      th_curr       = th_init.unsqueeze(0)
      dtheta        = torch.zeros_like(th_curr)
      eps_traj      = torch.zeros(planner_params['total_time_step']+1, robot.nlinks, 1)
      eps_traj      = eps_traj.unsqueeze(0).repeat(th_curr.shape[0],1,1,1)
      obsfactor.set_eps(eps_traj)
      if model_type == 'recurrent':
        curr_hidden = planner.learn_module.init_hidden(batch_size)
      else: curr_hidden = None       


      task_loss_per_iter = []
      cost_per_iter      = []
      ext_cost_per_iter  = []
      pos_loss_per_iter  = []
      vel_loss_per_iter  = []
      loss_per_iter      = []
      th_best = th_init.unsqueeze(0)

      best_gp_err = np.inf
      if args.render:
        # th_init_np = th_init.cpu().detach().numpy()
        # th_opt_np = th_opt.cpu().detach().numpy()
        env = Env2D(env_params)
        env.initialize_from_image(im[0], sdf[0])
        # path_init = [th_init_np[i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
        # path_opt  = [th_opt_np[i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
        env.initialize_plot(start_conf.cpu().numpy(), goal_conf.cpu().numpy())
        env.plot_signed_distance_transform()
        raw_input('Enter to start ...')
        plt.show(block=False)
      found_non_coll = False
      while True:
        print("Current iteration = %d"%j)
        if args.render:
          th_curr_np = th_curr.cpu().detach().numpy()
          th_opt_np = th_opt.cpu().detach().numpy()
          path_curr = [th_curr_np[0, i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
          path_opt  = [th_opt_np[i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
          
          if not args.use_static_covs and args.viz_ftrs:
            # Plot activations
            if use_dtheta: th_in = torch.cat((th_curr, dtheta), dim=-1)
            else: th_in = th_curr 
            costmap = sdf_utils.costmap_2d(sdf, obs_params['epsilon_dist'])
            _, activations = planner.learn_module_conv(sdf.unsqueeze(0))
            num_subplots = len(activations)
            for ip,vp in enumerate(xrange(num_subplots)):
              vp = vp + 1
              ax_curr = plt.subplot(num_subplots,1,vp)
              im_list = []
              for jp in xrange(activations[ip].shape[1]):
                curr_im = activations[ip][0, jp, :, :]
                curr_im = curr_im.unsqueeze(0)
                im_list.append(curr_im)
              grid_curr = make_grid(im_list, nrow=16, padding=1, pad_value=1)
              ax_curr.imshow(grid_curr.numpy().transpose((1,2,0)))
              ax_curr.set_title('Conv layer %d activations'%(ip))
          
          if j > 0:
            env.clear_edges()
          
          env.plot_edge(path_curr, color='blue')#, linestyle='-', linewidth=0.01*j , alpha=1.0-(1.0/(j+0.0001)) )
          # env.plot_edge(path_opt, color='red', linestyle='--')#, linewidth=0.01*j , alpha=1.0-(1.0/(j+0.0001)) )
          plt.show(block=False)
          time.sleep(0.002)
          if args.step:
            raw_input('Press enter for next step')
        

        dtheta, curr_hidden, err_old, err_ext_old, qc_inv_traj, obscov_inv_traj, eps_traj = planner.step(th_curr, start.unsqueeze(0), goal.unsqueeze(0), im.unsqueeze(0), sdf.unsqueeze(0), conv_out, dtheta, curr_hidden)
        err_sg, err_gp, err_obs = planner.unweighted_errors_batch(th_curr, sdf.unsqueeze(0))
        task_loss = err_sg + err_gp + learn_params['optim']['ext_obs_lambda']*err_obs
        
        #We will return the smoothest collision free trajectory 
        gp_error, _ , _ = gpfactor.get_error(th_curr) 
        obs_error, _ = obsfactor.get_error(th_curr, sdf.unsqueeze(0))
        mse_gp = torch.mean(torch.sum(gp_error**2, dim=-1))        
        in_coll, _, _, _ = collision_metrics(th_curr[0], obs_error[0], planner_params['total_time_sec'], planner_params['total_time_step'])
        if not in_coll:
          found_non_coll = True
          if mse_gp.item() < best_gp_err:
            th_best = th_curr
            best_gp_err = mse_gp.item()

        task_loss_per_iter.append(task_loss.item())
        cost_per_iter.append(err_old.item())
        ext_cost_per_iter.append(err_ext_old.item())
        pos_loss = criterion(th_curr[0][:,0:dof], th_opt[:,0:dof]).item()
        vel_loss = criterion(th_curr[0][:,dof:], th_opt[:,dof:]).item()
        mse_loss = criterion(th_curr[0], th_opt).item()
        pos_loss_per_iter.append(pos_loss)
        vel_loss_per_iter.append(vel_loss)
        loss_per_iter.append(mse_loss)
        
        print('Pos Loss = %f, Vel Loss = %f, MSE loss = %f'%(pos_loss, vel_loss, mse_loss))
        th_old  = th_curr
        th_curr = th_curr + dtheta
        th_new  = th_curr
        err_new = planner.error_batch(th_curr, sdf.unsqueeze(0)).item()
        err_ext_new = planner.error_ext_batch(th_curr, sdf.unsqueeze(0)).item()

        err_delta = err_new - err_old[0]
        err_ext_delta = err_ext_new - err_ext_old[0]
        print('|dtheta| = %f, err = %f, err_ext = %f, err_delta = %f, err_ext_delta = %f,\
               |qc_inv| = %f, |obscov_inv| = %f, |eps_traj| = %f'%(torch.norm(dtheta), err_old[0],\
                                                  err_ext_old[0], err_delta, err_ext_delta,\
                                                  torch.norm(qc_inv_traj, p='fro', dim=(2,3)).mean(),\
                                                  torch.norm(obscov_inv_traj, p='fro', dim=(2,3)).mean(),\
                                                  torch.norm(eps_traj, p='fro', dim=(2,3)).mean()))
        
        j = j + 1
        if check_convergence(dtheta, j, torch.tensor(err_delta), optim_params['tol_err'], optim_params['tol_delta'], optim_params['max_iters']):
          print('Converged')
          break

      #if found_non_coll: th_final = th_best
      #else: th_final = th_curr 
      th_final = th_curr

      #########################METRICS##########################################
      avg_vel, avg_acc, avg_jerk = smoothness_metrics(th_final[0], planner_params['total_time_sec'], planner_params['total_time_step'])
      gp_error, _ , _ = gpfactor.get_error(th_final) 
      obs_error, _ = obsfactor.get_error(th_final, sdf.unsqueeze(0))
      mse_gp = torch.mean(torch.sum(gp_error**2, dim=-1))
      in_coll, avg_penetration, max_penetration, coll_int = collision_metrics(th_final[0], obs_error[0], planner_params['total_time_sec'], planner_params['total_time_step'])
      print('MSE GP = {}, Average velocity = {}, average acc = {}, avg jerk=  {}'.format(mse_gp, avg_vel, avg_acc, avg_jerk))
      print('In coll = {}, average penetration = {}, max penetration = {}, collision intensity =  {}'.format(in_coll, 
                                                                                                             avg_penetration, 
                                                                                                             max_penetration, 
                                                                                                             coll_int))
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
            print ('Constraint violation!!!!!')
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
      task_loss = err_sg + err_gp + learn_params['optim']['ext_obs_lambda']*err_obs
      
      task_loss_per_iter.append(task_loss.item())
      cost_per_iter.append(planner.error_batch(th_final, sdf.unsqueeze(0)).item())
      ext_cost_per_iter.append(planner.error_ext_batch(th_final, sdf.unsqueeze(0)).item())
      loss_per_iter.append(criterion(th_final[0], th_opt).item())
      pos_loss_per_iter.append(criterion(th_final[0,:,0:dof], th_opt[:,0:dof]).item())
      vel_loss_per_iter.append(criterion(th_final[0,:,dof:], th_opt[:,dof:]).item())      
      
      valid_task_loss_per_iter.append(task_loss_per_iter)
      valid_cost_per_iter.append(cost_per_iter)
      valid_ext_cost_per_iter.append(ext_cost_per_iter)
      valid_pos_loss_per_iter.append(pos_loss_per_iter)
      valid_vel_loss_per_iter.append(vel_loss_per_iter)
      valid_loss_per_iter.append(loss_per_iter)
      valid_num_iters.append(j)

      if args.render:
        th_init_np = th_init.cpu().detach().numpy()
        th_final_np = th_final.cpu().detach().numpy()
        th_opt_np = th_opt.cpu().detach().numpy()
        path_init  = [th_init_np[i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
        path_opt  = [th_opt_np[i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
        path_final = [th_final_np[0][i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
        env.clear_edges()
        env.plot_edge(path_final, color='blue')#, linewidth=0.1*j)
        env.plot_edge(path_opt, color='red')#, linewidth=0.1*j)
        env.plot_edge(path_init, color='black')#, linewidth=0.1*j)
        plt.show(block=False)
        raw_input('Press enter for next env')
        env.close_plot()      

  print('Dumping results')  
  results_dict = {}
  results_dict['num_iters']            = valid_num_iters
  results_dict['cost_per_iter']        = valid_cost_per_iter
  results_dict['ext_cost_per_iter']    = valid_ext_cost_per_iter
  results_dict['pos_loss_per_iter']    = valid_pos_loss_per_iter
  results_dict['vel_loss_per_iter']    = valid_vel_loss_per_iter
  results_dict['loss_per_iter']        = valid_loss_per_iter
  results_dict['gp_mse']               = valid_gp_error
  results_dict['avg_vel']              = valid_avg_vel
  results_dict['avg_acc']              = valid_avg_acc
  results_dict['avg_jerk']             = valid_avg_jerk
  results_dict['in_collision']         = valid_in_coll
  results_dict['avg_penetration']      = valid_avg_penetration
  results_dict['max_penetration']      = valid_max_penetration
  results_dict['coll_intensity']       = valid_coll_intensity
  results_dict['task_loss_per_iter']   = valid_task_loss_per_iter
  results_dict['constraint_violation'] = valid_constraint_violation



  if args.use_static_covs: 
    if args.valid_idxs_file is None:
      filename = 'gpmp2_' + str(obs_params['cost_sigma'].item()) + '_valid_results.yaml'
    else: 
      filename = 'gpmp2_' + str(obs_params['cost_sigma'].item()) + '_valid_results_' + args.valid_idxs_file + '.yaml'

  elif args.valid_idxs_file is not None:
    filename = args.model_file+"_valid_results_" + args.valid_idxs_file + ".yaml"
  else: filename = args.model_file+"_valid_results.yaml"
  with open(results_folder+"/" + filename, 'w') as fp:
    yaml.dump(results_dict, fp)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='PyTorch Example')
  parser.add_argument('--dataset_folders', type=str, nargs='+', required=True, help='Folder with training files')
  parser.add_argument('--in_folder', type=str, required=True, help='Folder with planner, robot, env and learning parameters. Learnt parameters will also be saved here')
  parser.add_argument('--plan_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--robot_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--env_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--learn_param_file', type=str, required=True, help='Name of file with learning parameters')
  parser.add_argument('--valid_idxs_file', type=str,  help='Name of file with validation environment indices')
  parser.add_argument('--model_file', type=str, help="Load model state_dict from this file")
  parser.add_argument('--render', action='store_true', help='Plots the test results')
  parser.add_argument('--step', action='store_true', help='Render intermediate trajectories')
  parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA')
  parser.add_argument('--viz_ftrs', action='store_true', help='Visualize intermediate learned layers.')
  parser.add_argument('--seed_val', type=int, default=1234, help='Seed value shared by numpy as pytorch')
  parser.add_argument('--validation', action='store_true', help='Looks for valid_idxs in train_valid_split file for that experiment and runs only those')
  parser.add_argument('--use_static_covs', action='store_true', help='Use static covariances loaded from folder')
  parser.add_argument('--test_overfit', action='store_true', help='Validate on training environments to test overfitting')
  args = parser.parse_args()
  run_validation(args)



