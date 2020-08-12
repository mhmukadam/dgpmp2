#!/usr/bin/env python
import os, sys
sys.path.insert(0, "../..")
import matplotlib 
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import torch
import pprint
import time
import yaml
import csv
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D
from diff_gpmp2.gpmp2.gp import GPFactor
from diff_gpmp2.utils.helpers import load_params_learn 
from diff_gpmp2.utils.learn_utils import torch_optimizer, torch_loss, init_xavier_uniform, init_xavier_normal, mse_traj
from diff_gpmp2.utils.sdf_utils import bilinear_interpolate
from diff_gpmp2.utils.planner_utils import straight_line_traj, straight_line_trajb
from datasets import PlanningDataset, PlanningDatasetMulti
from initialization_network import InitNet

np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
pp = pprint.PrettyPrinter()
pfiletype = '.yaml'

def get_th_init_batch(start_conf_b, goal_conf_b, planner_params, device):
  batch_size = start_conf_b.shape[0]
  dof = planner_params['dof']
  total_time_step = planner_params['total_time_step']
  th_init_b = torch.zeros((batch_size, int(total_time_step)+1, 
                           planner_params['state_dim']), device=device) 
  for j in xrange(batch_size):
    th_init_b[j] = straight_line_traj(start_conf_b[j][0,0:dof], goal_conf_b[j][0,0:dof], planner_params['total_time_sec'],\
                                      total_time_step, planner_params['dof'], device,
                                      )
  return th_init_b

def one_step_loss(th, th_opt, criterion, learn_params, gp_prior):
  # thx  = torch.index_select(th, -1, torch.tensor(0))
  # thy  = torch.index_select(th, -1, torch.tensor(1))
  # th_pos = torch.cat((thx, thy), dim=-1)
  th_optx  = torch.index_select(th_opt, -1, torch.tensor(0))
  th_opty  = torch.index_select(th_opt, -1, torch.tensor(1))
  th_optpos = torch.cat((th_optx, th_opty), dim=-1)
  err = (th - th_optpos).unsqueeze(-1)
  mse_traj = torch.mean(torch.einsum('bsij,bsjk->bsik', err.transpose(2,3), err))
  return mse_traj


def train_valid_split(dataset_folders, valid_size, expert, batch_size, shuffle, num_workers, num_train_envs=-1, num_train_probs=-1, pin_memory=False):
  train_dataset = PlanningDatasetMulti(dataset_folders, mode='train', num_envs=num_train_envs, num_env_probs=num_train_probs, label_subdir='opt_trajs_'+expert)
  valid_dataset = PlanningDatasetMulti(dataset_folders, mode='train', num_envs=num_train_envs, num_env_probs=num_train_probs, label_subdir='opt_trajs_'+expert)
  num_train = len(train_dataset)
  indices   = list(range(num_train))
  split     = int(np.floor(valid_size * num_train))
  if shuffle:
    np.random.shuffle(indices)
  train_idx, valid_idx = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=batch_size, sampler=train_sampler,
      num_workers=num_workers, pin_memory=pin_memory,
  )
  valid_loader = torch.utils.data.DataLoader(
      valid_dataset, batch_size=batch_size, sampler=valid_sampler,
      num_workers=num_workers, pin_memory=pin_memory,
  )
  return train_loader, valid_loader, train_idx, valid_idx

def check_solved(traj, sdf, robot_radius, cell_size, env_params, use_cuda):
  for state in traj[0]:
    state_pos = state[0:2]
    state_pos = state_pos.reshape(1,1,state_pos.shape[0])
    d_obs, _ = bilinear_interpolate(sdf, state_pos, cell_size, env_params['x_lims'], env_params['y_lims'], use_cuda)
    if d_obs <= robot_radius:
      return 0
  return 1

def smoothness_error(trajb, gp_prior):
  err, _, _ = gp_prior.get_error(trajb)
  mse = torch.mean(torch.einsum('bsij,bsjk->bsik', err.transpose(2,3), err), dim=1)
  return mse

def train(args, model, criterion, gp_prior, optimizer, train_loader, valid_loader, train_idxs, valid_idxs, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, learn_params, model_folder, results_folder, use_cuda):
  start_t = time.time()
  device = torch.device('cuda') if use_cuda else torch.device('cpu')
  if use_cuda: torch.set_default_tensor_type(torch.cuda.DoubleTensor) 
  else: torch.set_default_tensor_type(torch.DoubleTensor)  
  num_epochs = learn_params['optim']['epochs']
  batch_size = learn_params['optim']['batch_size']  
  do_validation = learn_params['optim']['do_validation']
  eval_epoch = learn_params['optim']['eval_epoch']
  save_epoch = learn_params['optim']['save_epoch']
  clip_grad  = learn_params['optim']['clip_grad']
  if clip_grad: print('Gradient clipping enabled')
  num_batches = int(len(train_loader.dataset)/batch_size)
  if int(len(train_loader.dataset)%batch_size) > 0: num_batches = num_batches + 1
  batch_losses = np.zeros((num_epochs-start_epoch, num_batches))
  epoch_losses = np.zeros((num_epochs-start_epoch))
  valid_losses = []; valid_num_solved = []; valid_gp_mse = []

  valid_epochs = range(0, num_epochs, int(eval_epoch*num_epochs)) 
  save_epochs  = range(0, num_epochs, int(save_epoch*num_epochs))
  print valid_epochs
  for epoch in xrange(start_epoch, num_epochs):
    epoch_start_time = time.time()
    model.train()
    for it , sample in enumerate(train_loader, 0):  
      im_b    = sample['im'].to(device)
      sdf_b   = sample['sdf'].to(device)
      start_b = sample['start'].to(device)
      goal_b  = sample['goal'].to(device)
      target  = sample['th_opt'].to(device)
      start_conf_b = start_b[:, :, 0:dof]
      goal_conf_b  = goal_b[:, :, 0:dof]
      th_init_b    = straight_line_trajb(start_conf_b, goal_conf_b, planner_params['total_time_sec'], planner_params['total_time_step'], dof, device)
      data = torch.cat((im_b, sdf_b), dim=1)
      optimizer.zero_grad()
      th_initx   = torch.index_select(th_init_b, -1, torch.tensor(0))
      th_inity   = torch.index_select(th_init_b, -1, torch.tensor(1))
      th_initpos = torch.cat((th_initx, th_inity), dim=-1)
      output = model(data, th_initpos)#, start_b[:,:,0:2], goal_b[:,:,0:2])
      # th_final = th_initpos + output
      loss = one_step_loss(output, target-th_init_b, criterion, learn_params, gp_prior)
      loss.backward()
      if clip_grad:
        norm_val = nn.utils.clip_grad_norm_(model.parameters(), learn_params['optim']['clip_val'])
        print('Clipped gradients. Norm val = {}'.format(norm_val))
        # model.print_gradients()
      # else:
        # print('Raw Gradients')
        # model.print_gradients()
      optimizer.step()
      batch_losses[epoch, it] = loss.item()
      if (it + 1)% 10 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 epoch, (it + 1) * len(data), len(train_loader.dataset),
                 100. * (it + 1) / len(train_loader), loss.item()))  
    epoch_losses[epoch] = np.mean(batch_losses[epoch]) 
    if do_validation and (epoch in valid_epochs):
      print('Epoch = %d, Running validation'%(epoch))
      valid_start_time = time.time()
      valid_loss, num_solved, gp_mse = test(deepcopy(model), criterion, gp_prior, valid_loader, valid_idxs, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, learn_params, model_folder, results_folder, use_cuda, args.render)
      valid_losses.append(valid_loss)
      valid_num_solved.append(num_solved)
      valid_gp_mse.append(gp_mse)
      print('Validation results. Avg. Regression Loss = {}, Avg. problems solved = {}, Avg. GPMSE=  {}, Time taken = {}'.format(valid_loss, num_solved, gp_mse, time.time()-valid_start_time))
      print('Validation done epoch = %d. Saving model and optimizer'%(epoch))
      torch.save(model.state_dict(), model_folder + '/epoch_{}'.format(epoch))
      torch.save(optimizer.state_dict(), model_folder + '/optimizer_epoch_{}'.format(epoch))
    print('Epoch done. Total time taken = %f'%(time.time()-epoch_start_time))
    #Save model 
    if epoch in save_epochs:
      print('Epoch = %d. Saving model and optimizer'%(epoch))
      torch.save(model.state_dict(), model_folder + '/epoch_{}'.format(epoch))
      torch.save(optimizer.state_dict(), model_folder + '/optimizer_epoch_{}'.format(epoch))
    
    with open(results_folder + '/train_batch_losses.csv', 'w') as fp:
      writer = csv.writer(fp, dialect=csv.excel())
      for i in xrange(batch_losses.shape[0]):
        row = batch_losses[i]
        writer.writerow(row)

    with open(results_folder + '/train_epoch_losses.csv', 'w') as fp:
      writer = csv.writer(fp, dialect=csv.excel())
      writer.writerow(epoch_losses)

    #Save current training curves
    fig1, (ax1, ax2) = plt.subplots(2,1)#, sharey=True)
    fig1.tight_layout()
    ax1.plot(batch_losses[0:epoch+1,:].flatten())
    ax1.set_title('Regression loss per batch')
    ax2.plot(epoch_losses)
    ax2.set_title('Average regression loss per epoch')
    fig1.subplots_adjust(hspace=1.0)
    fig1.savefig(results_folder+'/train_curves.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig(results_folder+'/train_curves.png', dpi=300, bbox_inches='tight')
    
    if do_validation:
      fig2, (ax1, ax2, ax3) = plt.subplots(3,1)#, sharey=True)
      fig2.tight_layout()
      ax1.plot(valid_epochs[0:len(valid_losses)], valid_losses)
      ax1.set_title('Average regression loss')
      ax2.plot(valid_epochs[0:len(valid_losses)], valid_num_solved)
      ax2.set_title('Average Num solved')
      ax3.plot(valid_epochs[0:len(valid_losses)], valid_gp_mse)
      ax3.set_title('Average GP MSE')
      fig2.subplots_adjust(hspace=1.0)
      fig2.savefig(results_folder+'/valid_curves.pdf', dpi=300, bbox_inches='tight')
      fig2.savefig(results_folder+'/valid_curves.png', dpi=300, bbox_inches='tight')
      
      with open(results_folder + '/valid_results.csv', 'w') as fp:
        writer = csv.writer(fp, dialect=csv.excel())
        row = ['avg_regr_loss']
        row = row + valid_losses
        writer.writerow(row)
        row = ['avg_num_solved']
        row = row + valid_num_solved
        writer.writerow(row)
        row = ['avg_gpmse']
        row = row + valid_gp_mse
        writer.writerow(row)

  print('Saving last model and optimizer')
  torch.save(model.state_dict(), model_folder + '/epoch_{}'.format(epoch))
  torch.save(optimizer.state_dict(), model_folder + '/optimizer_epoch_{}'.format(epoch))
  print('Done training')
  print('Time taken = %f'%(time.time()-start_t))


def test(model, criterion, gp_prior, valid_loader, valid_idxs, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, learn_params, model_folder, results_folder, use_cuda, render):
  device = torch.device('cuda') if use_cuda else torch.device('cpu')
  if use_cuda: torch.set_default_tensor_type(torch.cuda.DoubleTensor) 
  else: torch.set_default_tensor_type(torch.DoubleTensor)  
  #Create planner object
  env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
  if robot_data['type'] == 'point_robot_2d':
    robot = PointRobot2D(robot_data['sphere_radius'], use_cuda=args.use_cuda)  
  x_min = env_data['x_lims'][0]
  x_max = env_data['x_lims'][1]
  cell_size = (x_max - x_min)/learn_params['im_size']*1.0
  dof = planner_params['dof']
  avg_loss = 0.0
  avg_solved = 0.0
  avg_gpmse = 0.0
  num_envs = len(valid_idxs)
  model.eval()
  with torch.no_grad():
    for i in valid_idxs:
      # print('Environment idx = %d'%i)
      sample = valid_loader.dataset[i]
      im     = sample['im'].unsqueeze(0).to(device)
      sdf    = sample['sdf'].unsqueeze(0).to(device)
      start  = sample['start'].unsqueeze(0).to(device)
      goal   = sample['goal'].unsqueeze(0).to(device)
      target = sample['th_opt'].unsqueeze(0).to(device) 
      start_conf = start[:, :, 0:dof]
      goal_conf  = goal[:, :, 0:dof]
      th_init    = straight_line_trajb(start_conf, goal_conf, planner_params['total_time_sec'], planner_params['total_time_step'], dof, device)
      data   = torch.cat((im, sdf), dim=1)
      th_initx   = torch.index_select(th_init, -1, torch.tensor(0))
      th_inity   = torch.index_select(th_init, -1, torch.tensor(1))
      th_initpos = torch.cat((th_initx, th_inity), dim=-1)
      output = model(data, th_initpos) #start[:,:,0:dof], goal[:,:,0:dof])
      avg_loss = avg_loss + one_step_loss(output, target-th_init, criterion, learn_params, gp_prior).item()
      th_final = th_initpos + output
      avg_solved = avg_solved + check_solved(th_final, sdf, robot_data['sphere_radius'], cell_size, env_params, use_cuda)
      # avg_gpmse = avg_gpmse + smoothness_error(output, gp_prior)
      if args.render:
        env = Env2D(env_params)
        env.initialize_from_image(im[0,0].cpu(), sdf[0,0].cpu())
        env.initialize_plot(start[0,0,0:2].cpu().numpy(), goal[0,0,0:2].cpu().numpy())
        env.plot_signed_distance_transform()
        th_final_np = th_final[0].cpu().detach().numpy()
        th_opt_np   = target[0].cpu().detach().numpy()
        path_final  = [th_final_np[i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
        path_opt    = [th_opt_np[i, 0:planner_params['dof']] for i in xrange(planner_params['total_time_step']+1)]
        env.plot_edge(path_final, color='blue')#, linestyle='-', linewidth=0.01*j , alpha=1.0-(1.0/(j+0.0001)) )
        env.plot_edge(path_opt, color='red', linestyle='--')#, linewidth=0.01*j , alpha=1.0-(1.0/(j+0.0001)) )
        plt.show(block=False)
        # time.sleep(0.002)
        raw_input('Press enter for next step')
        env.close_plot()
      
  return avg_loss/num_envs*1.0, avg_solved/num_envs*1.0, avg_gpmse/num_envs*1.0

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='PyTorch Example')
  parser.add_argument('--dataset_folders', type=str, nargs='+', required=True, help='Folder with training files')
  parser.add_argument('--in_folder', type=str, required=True, help='Folder with planner, robot, env and learning parameters. Learnt parameters will also be saved here')
  parser.add_argument('--plan_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--robot_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--env_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--learn_param_file', type=str, required=True, help='Name of file with learning parameters')
  parser.add_argument('--model_file', type=str, help="Model file for hotstarting or validation")
  parser.add_argument('--optimizer_file', type=str, help="Optimizer file for hotstarting or validation")
  parser.add_argument('--start_epoch', type=int, help="If starting from partially trained network, start from this epoch")
  parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA')
  parser.add_argument('--plot_test', action='store_true', help='Plots the test results and steps through environments one at a time')
  parser.add_argument('--seed_val', type=int, default=1234, help='Seed value shared by numpy as pytorch')
  parser.add_argument('--test', action = 'store_true', help='If enabled, we train, else we load model and test')
  parser.add_argument('--test_overfit', action = 'store_true', help='Test overfitting while validating')
  parser.add_argument('--render', action='store_true', help='Render in validation or test mode')
  args = parser.parse_args()

  use_cuda = torch.cuda.is_available() if args.use_cuda else False
  device = torch.device('cuda') if use_cuda else torch.device('cpu')
  if use_cuda: torch.set_default_tensor_type(torch.cuda.DoubleTensor) 
  else: torch.set_default_tensor_type(torch.DoubleTensor)  
  np.random.seed(args.seed_val)
  torch.manual_seed(args.seed_val)

  dataset_folders   = [os.path.abspath(folder) for folder in args.dataset_folders]
  input_folder      = os.path.abspath(args.in_folder)
  plan_param_file   = os.path.join(input_folder, args.plan_param_file +pfiletype)
  robot_param_file  = os.path.join(input_folder, args.robot_param_file+pfiletype)
  env_param_file    = os.path.join(input_folder, args.env_param_file  +pfiletype)  
  learn_param_file  = os.path.join(input_folder, args.learn_param_file+pfiletype)

  model_folder     = os.path.join(input_folder, "models") #folder to store learnt models in 
  results_folder   = os.path.join(input_folder, "results")
  if not os.path.exists(model_folder): os.makedirs(model_folder)
  if not os.path.exists(results_folder): os.makedirs(results_folder)
  env_data, planner_params, gp_params, obs_params,\
  optim_params, robot_data, learn_params = load_params_learn(plan_param_file, robot_param_file, env_param_file, 
                                                             learn_param_file, device)
  print optim_params, robot_data, learn_params, env_data, planner_params, gp_params, obs_params
  plot_test = args.plot_test
  
  train_loader, valid_loader, train_idxs, valid_idxs = train_valid_split(dataset_folders, 
                                                                       learn_params['data']['valid_size'],
                                                                       learn_params['data']['expert'], 
                                                                       learn_params['optim']['batch_size'],
                                                                       learn_params['data']['shuffle'],
                                                                       learn_params['data']['num_workers'],
                                                                       learn_params['data']['num_train_envs'],
                                                                       learn_params['data']['num_train_env_probs'],
                                                                       learn_params['data']['pin_memory'],
                                                                       )

  learn_params['im_size'] = train_loader.dataset.meta_data[0]['im_size']
  num_traj_states = planner_params['total_time_step'] + 1
  state_dim = planner_params['state_dim']
  im_size = learn_params['data']['im_size']
  dof = planner_params['dof']
  dt = planner_params['total_time_sec']/planner_params['total_time_step']*1.0

  model = InitNet(2, im_size, num_traj_states, dof)
  criterion = torch_loss(learn_params['optim']['criterion'], reduction=learn_params['optim']['loss_reduction'])
  optimizer = torch_optimizer(learn_params['optim']['optimizer'], parameters=model.parameters(), opt_params_dict=learn_params['optim']) 
  gp_prior  = GPFactor(dof, dt, num_traj_states-1, batch_size=learn_params['optim']['batch_size'], use_cuda=use_cuda)


  print("Model", model)
  print("Criterion", criterion)
  print("Optimizer", optimizer)

  if not args.test:  
    if args.model_file:
      print('Loading weights from specified file. Starting training from epoch = %d'%(args.start_epoch))
      start_epoch = args.start_epoch
      model.load_state_dict(torch.load(model_folder+"/"+args.model_file))
    else:
      print('No file specified, initializing weights')
      start_epoch = 0
    train_val_dict = {'train_idxs': train_idxs, 'valid_idxs': valid_idxs}
    with open(results_folder + '/train_val_split.yaml', 'w') as fp:
      yaml.dump(train_val_dict, fp)
    print len(train_idxs), len(valid_idxs)
    train(args, model, criterion, gp_prior, optimizer, train_loader, valid_loader, train_idxs, valid_idxs, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, learn_params, model_folder, results_folder, use_cuda)
  else:
    #Load model file and valid_idxs file
    if not args.model_file: raise ValueError, 'Please enter a model file to validate'
    idxs_file  = os.path.join(results_folder, 'train_val_split'+pfiletype)
    print('Loading weights from specified file.')
    model.load_state_dict(torch.load(model_folder+"/"+args.model_file))
    
    with open(idxs_file , 'r') as fp:
      train_val_split = yaml.load(fp)
    if args.test_overfit:
      print 'Testing overfitting using training data'
      idxs = train_val_split['train_idxs']
    else: idxs = train_val_split['valid_idxs']
    
    test(model, criterion, gp_prior, valid_loader, idxs, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, learn_params, model_folder, results_folder, use_cuda, args.render)
    print('Done testing')