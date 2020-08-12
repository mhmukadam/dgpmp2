#!/usr/bin/env python
import argparse
import os, sys
sys.path.insert(0, "../..")
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from torchvision import transforms, utils
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D
from diff_gpmp2.gpmp2 import DiffGPMP2Planner
from diff_gpmp2.learning import LearnModuleFCN, LearnModuleConv
from diff_gpmp2.utils.helpers import load_params_learn 
from diff_gpmp2.utils.planner_utils import straight_line_traj, straight_line_trajb, check_convergence_batch 
from diff_gpmp2.utils.learn_utils import torch_optimizer, torch_loss, init_xavier_uniform, init_xavier_normal, mse_traj
from datasets import PlanningDataset, PlanningDatasetMulti


np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
pp = pprint.PrettyPrinter()
pfiletype = '.yaml'


def lambda_schedule(epoch_c, epoch_f, lambda0):
  return (1.0 - epoch_c*1.0/(epoch_f)) * lambda0

def get_th_init_batch(start_conf_b, goal_conf_b, planner_params, device):
  batch_size = start_conf_b.shape[0]
  dof = planner_params['dof']
  total_time_step = planner_params['total_time_step']
  th_init_b = torch.zeros((batch_size, int(total_time_step)+1, 
                           planner_params['state_dim']), device=device) 
  for j in range(batch_size):
    th_init_b[j] = straight_line_traj(start_conf_b[j][0,0:dof], goal_conf_b[j][0,0:dof], planner_params['total_time_sec'],\
                                      total_time_step, planner_params['dof'], device,
                                      )
  return th_init_b

def train_valid_split(dataset_folders, valid_size, expert, batch_size, shuffle, num_workers, num_train_envs=-1, num_train_probs=-1, pin_memory=False):
  train_dataset = PlanningDatasetMulti(dataset_folders, mode='train', num_envs=num_train_envs, num_env_probs=num_train_probs, label_subdir='opt_trajs_'+expert)
  valid_dataset = PlanningDatasetMulti(dataset_folders, mode='train', num_envs=num_train_envs, num_env_probs=num_train_probs, label_subdir='opt_trajs_'+expert)
  num_train = len(train_dataset)
  indices   = list(range(num_train))
  split     = int(np.floor(valid_size * num_train))

  if shuffle:
    np.random.shuffle(indices)


  train_idx, valid_idx = indices[split:], indices[:split]
  print(len(train_idx))
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

def one_step_loss(th_curr, th_opt, qc_inv_trajb, obscov_inv_traj_b, err_sg, err_gp, err_obs, criterion, learn_params, dof, epoch):
  #########Loss calculation logic#########
  #Calculate loss based on position and velocity
  # th_opt_pos_b = th_optb[:,:,0:dof]
  # th_opt_vel_b = th_optb[:,:,dof:]
  # th_curr_pos_b = th_currb[:,:,0:dof]
  # th_curr_vel_b = th_currb[:,:,dof:]
  th_currx   = torch.index_select(th_curr, -1, torch.tensor(0))
  th_curry   = torch.index_select(th_curr, -1, torch.tensor(1))
  th_currpos = torch.cat((th_currx, th_curry), dim=-1)
  th_currvx  = torch.index_select(th_curr, -1, torch.tensor(2))
  th_currvy  = torch.index_select(th_curr, -1, torch.tensor(3))
  th_currvel = torch.cat((th_currvx, th_currvy), dim=-1)

  th_optx   = torch.index_select(th_opt, -1, torch.tensor(0))
  th_opty   = torch.index_select(th_opt, -1, torch.tensor(1))
  th_optpos = torch.cat((th_optx, th_opty), dim=-1)
  th_optvx  = torch.index_select(th_opt, -1, torch.tensor(2))
  th_optvy  = torch.index_select(th_opt, -1, torch.tensor(3))
  th_optvel = torch.cat((th_optvx, th_optvy), dim=-1)


  err_pos  = (th_currpos - th_optpos).unsqueeze(-1)
  err_vel  = (th_currvel - th_optvel).unsqueeze(-1)
  pos_loss = torch.mean(torch.einsum('bsij,bsjk->bsik', err_pos.transpose(2,3), err_pos))
  vel_loss = torch.mean(torch.einsum('bsij,bsjk->bsik', err_vel.transpose(2,3), err_vel))
  vel_lam  = learn_params['optim']['vel_loss_lambda']
  expert_loss = pos_loss +  vel_lam * vel_loss

  # pos_loss = criterion(th_curr_pos_b, th_opt_pos_b)
  # vel_loss = criterion(th_curr_vel_b, th_opt_vel_b)   
  gp_loss  = err_gp.mean()
  sg_loss  = err_sg.mean()
  obs_loss = err_obs.mean()
  obs_lam = learn_params['optim']['ext_obs_lambda']
  ext_loss = gp_loss + sg_loss +  obs_lam * obs_loss
  # w = lambda_schedule(epoch, learn_params['optim']['exp_loss_decay_epoch'], learn_params['optim']['exp_loss_weight'])
  #Calculate loss based on smoothness and obstacle cost
  cov_loss = torch.tensor(0.0)
  # if learn_params['optim']['regularize_covs']:            
  #   cov_loss = (torch.norm(qc_inv_trajb, p='fro', dim=(2,3)).mean() + torch.norm(obscov_inv_traj_b, p='fro', dim=(2,3)).mean())
  #   total_loss = total_loss + learn_params['optim']['cov_reg']*cov_loss
  
  total_loss = expert_loss + learn_params['optim']['ext_loss_weight'] * ext_loss
  
  return total_loss, pos_loss, vel_loss, cov_loss, gp_loss, sg_loss, obs_loss, ext_loss


def train(args, planner, criterion, optimizer, train_loader, valid_loader, train_idxs, valid_idxs, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, learn_params, model_folder, results_folder, use_cuda):
  start_t = time.time()
  device = torch.device('cuda') if use_cuda else torch.device('cpu')
  if use_cuda: torch.set_default_tensor_type(torch.cuda.DoubleTensor) 
  else: torch.set_default_tensor_type(torch.DoubleTensor)  

  # dataset_folders   = [os.path.abspath(folder) for folder in args.dataset_folders]
  input_folder      = os.path.abspath(args.in_folder)
  # plan_param_file   = os.path.join(input_folder, args.plan_param_file +pfiletype)
  # robot_param_file  = os.path.join(input_folder, args.robot_param_file+pfiletype)
  # env_param_file    = os.path.join(input_folder, args.env_param_file  +pfiletype)  
  # learn_param_file  = os.path.join(input_folder, args.learn_param_file+pfiletype)

  # model_folder     = os.path.join(input_folder, "models") #folder to store learnt models in 
  # results_folder   = os.path.join(input_folder, "results")
  # if not os.path.exists(model_folder): os.makedirs(model_folder)
  # if not os.path.exists(results_folder): os.makedirs(results_folder)
  # env_data, planner_params, gp_params, obs_params,\
  # optim_params, robot_data, learn_params = load_params_learn(plan_param_file, robot_param_file, env_param_file, 
  #                                                            learn_param_file, device)
  # print optim_params, robot_data, learn_params, env_data, planner_params, gp_params, obs_params
  # plot_test = args.plot_test
  
  # train_loader, valid_loader, train_idx, valid_idx = train_valid_split(dataset_folders, 
  #                                                                      learn_params['data']['valid_size'],
  #                                                                      learn_params['data']['expert'], 
  #                                                                      learn_params['optim']['batch_size'],
  #                                                                      learn_params['data']['shuffle'],
  #                                                                      learn_params['data']['num_workers'],
  #                                                                      learn_params['data']['num_train_envs'],
  #                                                                      learn_params['data']['num_train_env_probs'],
  #                                                                      learn_params['data']['pin_memory'],
  #                                                                      )

  # train_val_dict = {'train_idxs': train_idx, 'valid_idxs': valid_idx}

  # with open(results_folder + '/train_val_split.yaml', 'w') as fp:
  #   yaml.dump(train_val_dict, fp)

  #Create planner object
  # env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
  # if robot_data['type'] == 'point_robot_2d':
  #   robot = PointRobot2D(robot_data['sphere_radius'], use_cuda=args.use_cuda)
  # learn_params['im_size'] = train_loader.dataset.meta_data[0]['im_size']
  # planner = DiffGPMP2Planner(gp_params, obs_params, planner_params, optim_params, env_params, robot, learn_params, batch_size=learn_params['optim']['batch_size'], use_cuda=use_cuda)
  
  # if args.model_file:
  #   print('Loading weights from specified file. Starting training from epoch = %d'%(args.start_epoch))
  #   start_epoch = args.start_epoch
  #   planner.load_state_dict(torch.load(model_folder+"/"+args.model_file))
  # else:#Initialize weights here
  #   print('No file specified, initializing weights')
  #   # gain = 1
  #   # if learn_params['model']['activation'] == 'relu': gain=nn.init.calculate_gain('relu')
  #   # planner.learn_module_conv.apply(init_xavier_uniform)
  #   planner.learn_module_fcn.apply(init_xavier_uniform)
  #   start_epoch = 0
  
  # optimizer = torch_optimizer(learn_params['optim']['optimizer'], parameters=planner.parameters(), opt_params_dict=learn_params['optim'])
  # if args.optimizer_file:
  #   print('Loading optimizer state from file specified')
  #   optimizer.load_state_dict(torch.load(model_folder+"/"+args.optimizer_file))
  # else: print('Initializing optimizer state from scratch') 
  # planner.to(device) #Send planner to correct device
  # num_params = sum(p.numel() for p in planner.parameters() if p.requires_grad)
  # print('Num planner parameters %d'%num_params)
  # if learn_params['optim']['criterion'] == 'norm_mse': criterion = torch_loss(learn_params['optim']['criterion'], reduction=learn_params['optim']['loss_reduction'],
  #                                                                             alpha=learn_params['optim']['alpha'], beta=learn_params['optim']['beta'] )
  # else: criterion = torch_loss(learn_params['optim']['criterion'], reduction=learn_params['optim']['loss_reduction'])
  # print 'Planner: ', planner
  # print 'Optimizer: ', optimizer
  # print 'Criterion: ', criterion
  
  
  fixed_conv = learn_params['dgpmp2']['fixed_conv']
  do_validation = learn_params['optim']['do_validation'] 
  if learn_params['model']['type'] == 'rnn_gru' or learn_params['model']['type'] == 'rnn_lstm': model_type = 'recurrent'
  else: model_type = 'feed_forward'  
  use_dtheta = learn_params['dgpmp2']['use_dtheta'] if 'dtheta' in learn_params['dgpmp2'] else False
  batch_size = learn_params['optim']['batch_size']
  dof = planner_params['dof']
  T = learn_params['dgpmp2']['T']   #Total training horizon
  tk = learn_params['dgpmp2']['tk'] #BPTT horizon
  num_epochs = learn_params['optim']['epochs']
  save_epoch = learn_params['optim']['save_epoch']
  eval_epoch = learn_params['optim']['eval_epoch']
  if 'tk2' in learn_params['dgpmp2']: tk2 = learn_params['dgpmp2']['tk2']
  else: tk2 = tk 
  clip_grad = learn_params['optim']['clip_grad']
  if clip_grad: print('Gradient clipping enabled')
  else: print('No gradient clipping') 
  
  num_batches = int(len(train_loader.dataset)/batch_size)
  retain_graph = tk < tk2
  if learn_params['dgpmp2']['use_inter_loss']: retain_graph = True
  
  valid_epochs = range(0, num_epochs, int(eval_epoch*num_epochs)) 
  save_epochs  = range(0, num_epochs, int(save_epoch*num_epochs))

  #Batch statistics
  pos_loss_per_batch = []
  vel_loss_per_batch = []
  loss_per_batch     = []
  err_per_batch      = []
  gp_err_per_batch   = []
  sg_err_per_batch   = []
  obs_err_per_batch  = []
  err_ext_per_batch  = []
  #Epoch statistics
  pos_loss_per_epoch = []
  vel_loss_per_epoch = []
  loss_per_epoch     = []
  err_per_epoch      = []
  gp_err_per_epoch   = []
  sg_err_per_epoch   = []
  obs_err_per_epoch  = []
  err_ext_per_epoch  = []
  #Validation statistics
  valid_pos_loss_per_epoch = []
  valid_vel_loss_per_epoch = []
  valid_loss_per_epoch = []
  valid_num_solved_per_epoch = []
  valid_gp_mse = []
  valid_epoch_idxs = [] #Index of epochs when you did validation

  best_valid_loss = np.inf
  best_planner = planner
  best_epoch = 0

  print('Saving random planner for comparison')
  torch.save(planner.state_dict(), model_folder + '/init_planner')
  torch.save(optimizer.state_dict(), model_folder + '/init_optim') 
  for epoch in range(start_epoch, num_epochs):
    planner.train()    #Set planner to train mode
    epoch_start_time = time.time()
    for it , sample in enumerate(train_loader, 0):  
      im_b     = sample['im'].to(device)
      sdf_b    = sample['sdf'].to(device)
      start_b  = sample['start'].to(device)
      goal_b   = sample['goal'].to(device)
      th_opt_b = sample['th_opt'].to(device)
      start_conf_b = start_b[:, :, 0:dof]
      goal_conf_b  = goal_b[:,  :, 0:dof]
      th_init_b    = straight_line_trajb(start_conf_b, goal_conf_b, planner_params['total_time_sec'], planner_params['total_time_step'], dof, device)
      sdf_b.requires_grad_(True)
      th_init_b.requires_grad_(True)
      dthetab = torch.zeros_like(th_init_b)
      #Predict conv features
      conv_out = None
      if fixed_conv:
        data = torch.cat((im_b, sdf_b), dim=1) 
        conv_out, _ = planner.learn_module_conv(data)
      print('Zeroing grads ...')
      optimizer.zero_grad()
      
      t = 0
      th_curr_b   = th_init_b
      th_curr_buff = [(None, th_init_b)]
      if model_type == 'recurrent':
        hidden_buff = [(None, planner.learn_module_fcn.init_hidden(batch_size))]
      else: hidden_buff = None

      final_loss        = torch.tensor(0.0, device=device) #This will be backpropogated 
      batch_pos_loss    = 0.0 
      batch_vel_loss    = 0.0 
      batch_total_loss  = 0.0
      batch_cov_loss    = 0.0
      batch_gp_loss     = 0.0
      batch_sg_loss     = 0.0
      batch_obs_loss    = 0.0
      batch_ext_loss    = 0.0
      batch_start_time  = time.time()
      
      #########TBPTT#########
      while t < T:
        #Forward pass
        th_curr_b = th_curr_buff[-1][1].detach()
        # dthetab = dthetab.detach()
        th_curr_b.requires_grad=True
        # dthetab.requires_grad=True
        
        if model_type == 'recurrent':
          h = hidden_buff[-1][1][0].detach()
          c = hidden_buff[-1][1][1].detach()
          h.requires_grad=True
          c.requires_grad=True
          hidden = (h,c)
          dthetab, new_hidden, _, _, qc_inv_trajb, obscov_inv_traj_b, eps_traj_b = planner.step(th_curr_b, start_b, goal_b, im_b, sdf_b, conv_out, dthetab, hidden)
        else: dthetab, _, _, _, qc_inv_trajb, obscov_inv_traj_b, eps_traj_b      = planner.step(th_curr_b, start_b, goal_b, im_b, sdf_b, conv_out, dthetab)
        # print('dtheta = %f'%torch.norm(dthetab))
        th_new_b = th_curr_b + dthetab
        th_curr_buff.append((th_curr_b, th_new_b))
        if model_type == 'recurrent':
          hidden_buff.append((hidden, new_hidden))

        #We only look-back tk2 timesteps for backpropagating - detach everything else upstream
        while len(th_curr_buff) > tk2:
          print('Detaching old stuff ...')
          del th_curr_buff[0]
          if model_type == 'recurrent': del hidden_buff[0]
          # if learn_params['dgpmp2']['use_inter_loss']: del loss_buff[0]
      
        if learn_params['dgpmp2']['use_inter_loss']:
        #   #Add one step loss from every intermediate step
          err_sg, err_gp, err_obs = planner.unweighted_errors_batch(th_new_b, sdf_b)
          curr_total_loss, curr_pos_loss, curr_vel_loss, curr_cov_loss, curr_gp_loss, curr_sg_loss, curr_obs_loss, curr_ext_loss = one_step_loss(dthetab, th_opt_b - th_curr_b, qc_inv_trajb, obscov_inv_traj_b, err_sg, err_gp, err_obs, criterion, learn_params, dof, epoch)          
          final_loss = final_loss + curr_total_loss
          # loss_buff.append(curr_total_loss)
          batch_pos_loss   += curr_pos_loss.item()
          batch_vel_loss   += curr_vel_loss.item()
          batch_cov_loss   += curr_cov_loss.item()
          batch_gp_loss    += curr_gp_loss.item()
          batch_sg_loss    += curr_sg_loss.item()
          batch_obs_loss   += curr_obs_loss.item()
          batch_ext_loss   += curr_ext_loss.item()
          batch_total_loss += curr_total_loss.item()
   
        if (t+1) % tk == 0: 
          if not learn_params['dgpmp2']['use_inter_loss']:
            err_sg, err_gp, err_obs = planner.unweighted_errors_batch(th_new_b, sdf_b)
            curr_total_loss, curr_pos_loss, curr_vel_loss, curr_cov_loss, curr_gp_loss, curr_sg_loss, curr_obs_loss, curr_ext_loss = one_step_loss(dthetab, th_opt_b - th_currb, qc_inv_trajb, obscov_inv_traj_b, err_sg, err_gp, err_obs, criterion, learn_params, dof, epoch)
            final_loss = curr_total_loss
            batch_pos_loss   += curr_pos_loss.item()
            batch_vel_loss   += curr_vel_loss.item()
            batch_cov_loss   += curr_cov_loss.item()
            batch_gp_loss    += curr_gp_loss.item()
            batch_sg_loss    += curr_sg_loss.item()
            batch_obs_loss   += curr_obs_loss.item()
            batch_ext_loss   += curr_ext_loss.item()
            batch_total_loss += curr_total_loss.item()
          else:
            final_loss = final_loss / tk*1.0
            batch_pos_loss   = batch_pos_loss / tk*1.0 
            batch_vel_loss   = batch_vel_loss / tk*1.0
            batch_cov_loss   = batch_cov_loss / tk*1.0
            batch_gp_loss    = batch_gp_loss / tk*1.0
            batch_sg_loss    = batch_sg_loss / tk*1.0
            batch_obs_loss   = batch_obs_loss / tk*1.0
            batch_ext_loss   = batch_ext_loss / tk*1.0
            batch_total_loss = batch_total_loss / tk*1.0

          #Backpropogate in chunks
          bpstart = time.time()
          final_loss.backward(retain_graph=retain_graph)
          for j in xrange(tk2-1):
            if th_curr_buff[-j-2][0] is None:
              break
            if model_type == 'recurrent':
              curr_grad_hidden = hidden_buff[-j-1][0][0].grad
              hidden_buff[-j-2][1][0].backward(curr_grad_hidden, retain_graph=True) #Backward the hidden state
            curr_grad = th_curr_buff[-j-1][0].grad
            th_curr_buff[-j-2][1].backward(curr_grad, retain_graph=retain_graph)

          print('Backprop time = {}'.format(time.time()-bpstart))
          print('Raw Gradients')
          planner.learn_module_conv.print_gradients()
          planner.learn_module_fcn.print_gradients()


          print('Epoch= %d, batch=  %d, after %d iterations, pos loss = %f, vel loss = %f, cov_loss = %f, gp_loss = %f, sg_loss= %f, obs_loss = %f, ext_loss = %f, total_batch_loss = %f'%
                                                                    (epoch,
                                                                     it,
                                                                     t+1, 
                                                                     batch_pos_loss,
                                                                     batch_vel_loss,
                                                                     batch_cov_loss,
                                                                     batch_gp_loss,
                                                                     batch_sg_loss,
                                                                     batch_obs_loss,
                                                                     batch_ext_loss,
                                                                     batch_total_loss))
          
          if learn_params['dgpmp2']['optimize_tk']:
            if clip_grad:
              norm_val = nn.utils.clip_grad_norm_(planner.parameters(), learn_params['optim']['clip_val'])
              print('Clipped gradients. Norm val = {}'.format(norm_val))
              planner.learn_module_conv.print_gradients()
              planner.learn_module_fcn.print_gradients()
            print('Updating parameters after tk')
            optimizer.step()
        t = t + 1

      print('Total batch time %f'%(time.time()-batch_start_time))  
      with torch.no_grad():
        if model_type == 'recurrent':
          _, _, errb, err_extb, _ ,_, _  = planner.step(th_new_b, start_b, goal_b, im_b, sdf_b, conv_out, dthetab, hidden_buff[-1][1])
        else:
          _, _, errb, err_extb, _ ,_, _  = planner.step(th_new_b, start_b, goal_b, im_b, sdf_b, conv_out, dthetab)

      avg_batch_err = errb.mean().item()
      avg_batch_ext_err = err_extb.mean().item()
      print('epoch = {}, batch = {}, avg_batch_loss = {}, avg_traj_err = {}, avg_traj_ext_err = {}'.format(epoch, it, batch_total_loss,
                                                                                                           avg_batch_err,
                                                                                                           avg_batch_ext_err))
      if not learn_params['dgpmp2']['optimize_tk']:
        if clip_grad:
          norm_val = nn.utils.clip_grad_norm_(planner.parameters(), learn_params['optim']['clip_val'])
          print('Clipped gradients')
          planner.learn_module_conv.print_gradients()
          planner.learn_module_fcn.print_gradients()
        print('Updating parameters')
        optimizer.step()

      
      pos_loss_per_batch.append(batch_pos_loss/(T/tk)*1.)
      vel_loss_per_batch.append(batch_vel_loss/(T/tk)*1.)
      loss_per_batch.append(batch_total_loss/(T/tk)*1.)
      gp_err_per_batch.append(batch_gp_loss/(T/tk)*1.) 
      sg_err_per_batch.append(batch_sg_loss/(T/tk)*1.) 
      obs_err_per_batch.append(batch_obs_loss/(T/tk)*1.)
      err_per_batch.append(avg_batch_err)
      err_ext_per_batch.append(avg_batch_ext_err)

    #Get epoch statistics
    epoch_pos_loss = np.mean(pos_loss_per_batch).item()
    epoch_vel_loss = np.mean(vel_loss_per_batch).item()
    epoch_loss     = np.mean(loss_per_batch).item()
    epoch_err      = np.mean(err_per_batch).item()
    epoch_gp_err   = np.mean(gp_err_per_batch).item()
    epoch_sg_err   = np.mean(sg_err_per_batch).item()
    epoch_obs_err  = np.mean(obs_err_per_batch).item()
    epoch_ext_err  = np.mean(err_ext_per_batch).item()
    print('Epoch= {}, pos_loss = {}, vel_loss = {}, total_loss = {}, traj_err = {}, gp_error = {}, sg_error = {}, obs_error = {}, traj_ext_err = {}'.format(epoch,\
                                                                                                    epoch_pos_loss, epoch_vel_loss, epoch_loss, \
                                                                                                    epoch_err, epoch_gp_err, epoch_sg_err, epoch_obs_err,
                                                                                                    epoch_ext_err))
    pos_loss_per_epoch.append(epoch_pos_loss)
    vel_loss_per_epoch.append(epoch_vel_loss)
    loss_per_epoch.append(epoch_loss)
    err_per_epoch.append(epoch_err)
    err_ext_per_epoch.append(epoch_ext_err)
    gp_err_per_epoch.append(epoch_gp_err)
    sg_err_per_epoch.append(epoch_sg_err)
    obs_err_per_epoch.append(epoch_obs_err)

    if do_validation and (epoch in valid_epochs):
      valid_start_time = time.time()
      print('Epoch = %d, Running validation'%(epoch))
      valid_pos_loss, valid_vel_loss, num_solved, gp_mse = test(deepcopy(model), criterion, gp_prior, valid_loader, idxs, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, learn_params, model_folder, results_folder, use_cuda, render=args.render)
      valid_total_loss = valid_pos_loss + learn_params['optim']['vel_weight']*valid_vel_loss
      valid_pos_loss_per_epoch.append(valid_pos_loss)
      valid_vel_loss_per_epoch.append(valid_vel_loss)
      valid_loss_per_epoch.append(valid_total_loss)
      valid_num_solved_per_epoch.append(valid_err)
      valid_gp_mse.append(gp_mse)
      print('Validation results. avg. Pos Loss = {}, avg. vel loss = {} avg. problems solved = {}, avg. GPMSE=  {}, Time taken = {}'.format(valid_pos_loss, valid_vel_loss, num_solved, gp_mse, time.time()-valid_start_time))
      print('Epoch = {}, Best planner so far. Saving ...'.format(epoch))
      torch.save(planner.state_dict(), model_folder + '/epoch_{}'.format(epoch))
      torch.save(optimizer.state_dict(), model_folder + '/optimizer_epoch_{}'.format(epoch))
    print('Epoch done. Total time taken = %f'%(time.time()-epoch_start_time))
    #Save model 
    if epoch in save_epochs:
      print('Epoch = %d. Saving model and optimizer'%(epoch))
      torch.save(planner.state_dict(), model_folder + '/epoch_{}'.format(epoch))
      torch.save(optimizer.state_dict(), model_folder + '/optimizer_epoch_{}'.format(epoch))
    
    print('Dumping training results so far')
    loss_dict = {'pos_loss_per_batch'  : pos_loss_per_batch,
                 'vel_loss_per_batch'  : vel_loss_per_batch,
                 'total_loss_per_batch': loss_per_batch,
                 'err_per_batch': err_per_batch,
                 'err_ext_per_batch': err_ext_per_batch,
                 'pos_loss_per_epoch'  : pos_loss_per_epoch,
                 'vel_loss_per_epoch'  : vel_loss_per_epoch,
                 'total_loss_per_epoch': loss_per_epoch, 
                 'err_per_epoch': err_per_epoch,
                 'err_ext_per_epoch': err_ext_per_epoch,
                 'gp_err_per_epoch': gp_err_per_epoch,
                 'sg_err_per_epoch': sg_err_per_epoch,
                 'obs_err_per_epoch': obs_err_per_epoch,
                 'valid_pos_loss_per_epoch': valid_pos_loss_per_epoch,
                 'valid_vel_loss_per_epoch': valid_vel_loss_per_epoch, 
                 'valid_loss_per_epoch': valid_loss_per_epoch, 
                 'valid_num_solved_per_epoch': valid_num_solved_per_epoch,
                 'valid_gp_mse': valid_gp_mse}
    with open(results_folder + '/train_losses.yaml', 'w') as fp:
      yaml.dump(loss_dict, fp)
    #Save current training curves
    fig1, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8,1)#, sharey=True)
    fig1.tight_layout()
    ax1.plot(pos_loss_per_batch)
    ax1.set_title('Position loss per batch')
    ax2.plot(vel_loss_per_batch)
    ax2.set_title('Velocity loss per batch')
    ax3.plot(loss_per_batch)
    ax3.set_title('Total loss per batch')
    ax4.plot(err_per_batch)
    ax4.set_title('Error per batch (t=T)')
    ax5.plot(err_ext_per_batch)
    ax5.set_title('External error per batch (t=T)')
    ax6.plot(gp_err_per_batch)
    ax6.set_title('GP (unweghted) error per batch (t=T)')
    ax7.plot(obs_err_per_batch)
    ax7.set_title('Obstacle (unweighted) error per batch (t=T)')
    ax8.plot(sg_err_per_batch)
    ax8.set_title('Start Goal (unweighted) error per batch (t=T)')
    fig1.subplots_adjust(hspace=1.0)
    fig1.savefig(results_folder+'/train_curves_batch.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig(results_folder+'/train_curves_batch.png', dpi=300, bbox_inches='tight')
    
    fig2, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8,1)#, sharey=True)
    fig2.tight_layout()
    ax1.plot(pos_loss_per_epoch)
    ax1.set_title('Position loss per epoch')
    ax2.plot(vel_loss_per_epoch)
    ax2.set_title('Velocity loss per epoch')
    ax3.plot(loss_per_epoch)
    ax3.set_title('Total loss per epoch')
    ax4.plot(err_per_epoch)
    ax4.set_title('Error per epoch (t=T)')
    ax5.plot(err_ext_per_epoch)
    ax5.set_title('External error per epoch (t=T)')
    ax6.plot(gp_err_per_epoch)
    ax6.set_title('GP (unweghted) error per epoch')
    ax7.plot(obs_err_per_epoch)
    ax7.set_title('Obstacle (unweighted) error per epoch')
    ax8.plot(sg_err_per_epoch)
    ax8.set_title('Start Goal (unweighted) error per epoch')

    fig2.subplots_adjust(hspace=1.0)
    fig2.savefig(results_folder+'/train_curves_epoch.pdf', dpi=300, bbox_inches='tight')
    fig2.savefig(results_folder+'/train_curves_epoch.png', dpi=300, bbox_inches='tight')

    if do_validation:
      fig3, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1)#, sharey=True)
      fig3.tight_layout()
      ax1.plot(valid_epochs[0:len(valid_pos_loss_per_epoch)], valid_pos_loss_per_epoch)
      ax1.set_title('Average Position loss (Validation)')
      ax2.plot(valid_epochs[0:len(valid_pos_loss_per_epoch)], valid_vel_loss_per_epoch)
      ax2.set_title('Average velocity loss (Validation)')
      ax3.plot(valid_epochs[0:len(valid_pos_loss_per_epoch)], valid_loss_per_epoch)
      ax3.set_title('Total loss (Validation)')
      ax4.plot(valid_epochs[0:len(valid_pos_loss_per_epoch)], valid_num_solved_per_epoch)
      ax4.set_title('Trajectory error (validation)')
      ax5.plot(valid_epochs[0:len(valid_pos_loss_per_epoch)], valid_gp_mse)
      ax5.set_title('External trajectory error (validation)')
      fig3.subplots_adjust(hspace=0.7)
      fig3.savefig(results_folder+'/train_curves_valid.pdf', dpi=300, bbox_inches='tight')
      fig3.savefig(results_folder+'/train_curves_valid.png', dpi=300, bbox_inches='tight')
    

  print('Total training time = {}'.format(time.time()- start_t))
  plt.show()

def test(model, criterion, valid_loader, idxs, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, learn_params, model_folder, results_folder, use_cuda, render=False):
  pos_loss = 0.0
  vel_loss = 0.0

  planner.eval()
  num_batches = int(len(data_loader.dataset)/learn_params['optim']['batch_size'])
  dof = planner_params['dof']

  with torch.no_grad():
    for it , sample in enumerate(data_loader, 0):
      im_b     = sample['im'].to(device)
      sdf_b    = sample['sdf'].to(device)
      start_b  = sample['start'].to(device)
      goal_b   = sample['goal'].to(device)
      th_opt_b = sample['th_opt'].to(device)
    
      start_conf_b = start_b[:, 0:dof]
      goal_conf_b  = goal_b[:,  0:dof]
      th_init_b    = get_th_init_batch(start_conf_b, goal_conf_b, planner_params, device)
      itr = 0
      th_currb = th_init_b
      conv_vec = torch.zeros(batch_size,1,1).byte() 
      stp = time.time()
      while True:
        print "Current iteration, %d"%itr
        dthetab, _, _, _, _, _, _ = planner.step(th_currb, start_b, goal_b, im_b, sdf_b)
        dthetab = torch.where(conv_vec < 1, dthetab, torch.zeros(batch_size, total_time_step+1, planner_params['state_dim'], device=device))
        err_old  = planner.error_batch(th_currb, sdfb)
        th_currb = th_currb + dthetab 
        err_new  = planner.error_batch(th_currb, sdfb)
        err_delta = err_new - err_old
        itr = itr + 1
        conv_vec = check_convergence_batch(dthetab, itr, err_delta, optim_params['tol_err'], optim_params['tol_delta'], optim_params['max_iters'], device=device)
        if torch.sum(conv_vec) == batch_size:
          print('All converged')
          break

      print('Planning time = %f'%(time.time()-stp))
      th_finalb = th_currb
      pos_loss = criterion(th_final_b[:,:,0:dof], th_opt_b[:,:,0:dof])
      vel_loss = criterion(th_final_b[:,:,dof:], th_opt_b[:,:,dof:])
      total_loss = pos_loss + learn_params['optim']['vel_weight'] * vel_loss
      valid_pos_loss += pos_loss.item()
      valid_vel_loss += vel_loss.item()
      valid_loss += total_loss.item()
   
      valid_ext_err += np.mean(err_ext_per_iter_b[:,-1]).item()
      valid_err += np.mean(err_per_iter_b[:,-1]).item()

  avg_pos_loss = valid_pos_loss/num_batches*1.0
  avg_vel_loss = valid_vel_loss/num_batches*1.0
  avg_valid_loss = valid_loss/num_batches*1.0
  avg_valid_err = valid_err/num_batches*1.0
  avg_valid_ext_err = valid_ext_err/num_batches*1.0
  return avg_pos_loss, avg_vel_loss, avg_valid_err, avg_valid_ext_err 

if __name__ == "__main__":
  # global use_cuda, device, args
  parser = argparse.ArgumentParser(description='PyTorch Example')
  parser.add_argument('--dataset_folders', type=str, nargs='+', required=True, help='Folder with training files')
  parser.add_argument('--in_folder', type=str, required=True, help='Folder with planner, robot, env and learning parameters. Learnt parameters will also be saved here')
  parser.add_argument('--plan_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--robot_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--env_param_file', type=str, required=True, help='Name of file with planning parameters')
  parser.add_argument('--learn_param_file', type=str, required=True, help='Name of file with learning parameters')
  parser.add_argument('--model_file', type=str, help="Load model from here. If specified, we hotstart the network from this model")
  parser.add_argument('--optimizer_file', type=str, help="Load optimizer state from here. If specified, we hotstart the network from this model")
  parser.add_argument('--start_epoch', type=int, help="If starting from partially trained network, start from this epoch")
  parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA')
  # parser.add_argument('--plot_test', action='store_true', help='Plots the test results and steps through environments one at a time')
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
  print(optim_params, robot_data, learn_params, env_data, planner_params, gp_params, obs_params)
  env_params = {'x_lims': env_data['x_lims'], 'y_lims': env_data['y_lims']}
  if robot_data['type'] == 'point_robot_2d':
    robot = PointRobot2D(robot_data['sphere_radius'], use_cuda=args.use_cuda)
  
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

  # model = InitNet(2, im_size, num_traj_states, dof)
  planner   = DiffGPMP2Planner(gp_params, obs_params, planner_params, optim_params, env_params, robot, learn_params, batch_size=learn_params['optim']['batch_size'], use_cuda=use_cuda)
  criterion = torch_loss(learn_params['optim']['criterion'], reduction=learn_params['optim']['loss_reduction'])
  optimizer = torch_optimizer(learn_params['optim']['optimizer'], parameters=planner.parameters(), opt_params_dict=learn_params['optim']) 


  print("Planner", planner)
  print("Criterion", criterion)
  print("Optimizer", optimizer)

  if not args.test:  
    if args.model_file:
      print('Loading weights from specified file. Starting training from epoch = %d'%(args.start_epoch))
      start_epoch = args.start_epoch
      planner.load_state_dict(torch.load(model_folder+"/"+args.model_file))
    else:
      print('No file specified, initializing weights')
      start_epoch = 0
    train_val_dict = {'train_idxs': train_idxs, 'valid_idxs': valid_idxs}
    with open(results_folder + '/train_val_split.yaml', 'w') as fp:
      yaml.dump(train_val_dict, fp)
    print len(train_idxs), len(valid_idxs)
    train(args, planner, criterion, optimizer, train_loader, valid_loader, train_idxs, valid_idxs, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, learn_params, model_folder, results_folder, use_cuda)
  else:
    #Load model file and valid_idxs file
    if not args.model_file: raise ValueError, 'Please enter a model file to validate'
    idxs_file  = os.path.join(results_folder, 'train_val_split'+pfiletype)
    print('Loading weights from specified file.')
    planner.load_state_dict(torch.load(model_folder+"/"+args.model_file))
    
    with open(idxs_file , 'r') as fp:
      train_val_split = yaml.load(fp)
    if args.test_overfit:
      print 'Testing overfitting using training data'
      idxs = train_val_split['train_idxs']
    else: idxs = train_val_split['valid_idxs']
    
    test(planner, criterion, valid_loader, idxs, env_data, planner_params, gp_params, obs_params, optim_params, robot_data, learn_params, model_folder, results_folder, use_cuda, args.render)
    print('Done testing')







