#!/usr/bin/env python
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gp import PriorFactor, GPFactor
from .custom_factors import NonHolonomicFactor
from .obstacle import ObstacleFactor
from .plan_layer import PlanLayer
import matplotlib.pyplot as plt
from diff_gpmp2.utils.planner_utils import check_convergence
from diff_gpmp2.utils import mat_utils, sdf_utils
from diff_gpmp2.learning import LearnModuleFCN, LearnModuleConv

class DiffGPMP2Planner(nn.Module):
  def __init__(self, gp_params, obs_params, planner_params, optim_params, env_params, robot_model, learn_params=None, batch_size=1, use_cuda=False):
    super(DiffGPMP2Planner, self).__init__()
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
    self.dof = planner_params['dof']
    self.state_dim = planner_params['state_dim']
    self.total_time_sec = planner_params['total_time_sec']   
    self.total_time_step = planner_params['total_time_step']   #Number of states in the trajectory 
    self.num_traj_states = self.total_time_step + 1  #+1 for goal at the end
    self.num_gp_factors = self.num_traj_states - 1
    self.num_obs_factors = self.num_traj_states   
    self.optim_params = optim_params
    self.gp_params = gp_params
    self.obs_params = obs_params
    self.robot_model = robot_model
    self.env_params = env_params
    self.learn_params = learn_params
    self.model_type = 'feed_forward'
    self.non_holonomic = planner_params['non_holonomic'] if 'non_holonomic' in planner_params else False
    self.use_vel_limits = planner_params['use_vel_limits'] if 'use_vel_limits' in planner_params else False
    if self.non_holonomic: self.num_dynamics_factors = self.num_traj_states  
    if self.use_vel_limits:
      self.num_vel_factors = self.num_traj_states
    self.batch_size=batch_size
    if learn_params is None:
      qc_inv = self.gp_params['Q_c_inv']
      sigma = self.obs_params['cost_sigma']
      eps = self.obs_params['epsilon_dist']
      inv_cov      = mat_utils.isotropic_matrix(1.0/torch.pow(sigma,2.0), self.robot_model.nlinks, self.device)
      self.qc_inv_traj = torch.zeros(self.num_gp_factors, self.dof, self.dof, device=self.device)
      self.qc_inv_traj = self.qc_inv_traj + qc_inv 
      self.obscov_inv_traj = torch.zeros(self.num_traj_states, self.robot_model.nlinks, 1, device=self.device)
      self.obscov_inv_traj = self.obscov_inv_traj + inv_cov
      self.eps_traj = torch.zeros(self.num_traj_states, self.robot_model.nlinks, 1, device=self.device)
      self.eps_traj = self.eps_traj + eps
      self.learn_module_conv = None          
      self.learn_module_fcn  = None          
    else:
      self.model_type  = self.learn_params['model']['type'] if 'type' in self.learn_params['model'] else False 
      self.learn_eps   = self.learn_params['dgpmp2']['learn_eps'] if 'learn_eps' in self.learn_params['dgpmp2'] else False
      self.sdf_predict = self.learn_params['dgpmp2']['sdf_predict']
      self.use_dtheta  = self.learn_params['dgpmp2']['dtheta_predict'] if 'dtheta_predict' in self.learn_params['dgpmp2'] else False 
      self.res = (self.env_params['x_lims'][1] - self.env_params['x_lims'][0])/(self.learn_params['data']['im_size']*1.0)
      
      learn_params['num_traj_states'] = self.num_traj_states
      learn_params['state_dim'] = self.state_dim
      if self.use_dtheta: learn_params['num_traj_states'] = 2 * self.num_traj_states
      self.dynamics_mode = learn_params['dgpmp2']['dynamics_mode'] 
      if  self.dynamics_mode == 'fix_dynamics':
        learn_params['out_dim'] = self.num_obs_factors*self.robot_model.nlinks
        qc_inv = self.gp_params['Q_c_inv']
        self.qc_inv_traj = torch.zeros(self.num_gp_factors, self.dof, self.dof, device=self.device)
        self.qc_inv_traj = self.qc_inv_traj + qc_inv  
      elif self.dynamics_mode == 'diag_identity':
        learn_params['out_dim'] = self.num_gp_factors + self.num_obs_factors*self.robot_model.nlinks
      elif self.dynamics_mode == 'diag':
        learn_params['out_dim'] = self.num_gp_factors* self.dof + self.num_obs_factors*self.robot_model.nlinks
      elif self.dynamics_mode == 'qc_full':
        learn_params['out_dim'] = self.num_gp_factors* self.dof + self.num_obs_factors*self.robot_model.nlinks 
      elif self.dynamics_mode == 'q_full':
        learn_params['out_dim'] = self.num_gp_factors* self.state_dim + self.num_obs_factors*self.robot_model.nlinks 
      #if not learning epsilon, set it to user defined
      if self.learn_eps:
        learn_params['out_dim'] = learn_params['out_dim'] + self.num_obs_factors*self.robot_model.nlinks
      else:
        self.eps = self.obs_params['epsilon_dist']
        self.eps_traj = torch.zeros(self.num_traj_states, self.robot_model.nlinks, 1, device=self.device)
        self.eps_traj = self.eps_traj + self.eps
      
      self.fixed_conv = learn_params['dgpmp2']['fixed_conv'] if 'fixed_conv' in learn_params['dgpmp2'] else False
      # self.learn_module_conv = None
      self.learn_module_conv = LearnModuleConv(learn_params, env_params, robot_model, use_cuda=self.use_cuda)  
      self.learn_module_fcn = LearnModuleFCN(learn_params, env_params, obs_params, robot_model, use_cuda=self.use_cuda)
    
    self.plan_layer = PlanLayer(gp_params, obs_params, planner_params, optim_params, env_params, robot_model,learn_params,self.batch_size, self.use_cuda)

  def forward(self, th_initb, startb, goalb, imb, sdfb, hiddenb=None):
    start_t = time.time()
    nsamples = th_initb.shape[0]
    th_currb_temp = torch.zeros(th_initb.shape, device = self.device)
    if hiddenb is not None: hidden_newb_temp = torch.zeros(hiddenb.shape, device=self.device) 
    jb = []
    timeb = []
    err_initb = []
    err_finalb = []
    err_per_iterb = []
    err_ext_per_iterb = []

    for i in range(nsamples):
      th_curr = th_initb[i].unsqueeze(0)
      start   = startb[i].unsqueeze(0)
      goal    = goalb[i].unsqueeze(0)
      im      = imb[i].unsqueeze(0)
      sdf     = sdfb[i].unsqueeze(0)
      conv_out = None
       
      if self.learn_module_fcn is not None:
        if self.sdf_predict: im_in = sdf.unsqueeze(0)
        else: im_in = im.unsqueeze(0)
        if self.fixed_conv:
          conv_out, _ = self.learn_module_conv(im_in)
      if hiddenb is not None: 
        hidden_curr = (hiddenb[0][i].unsqueeze(0), hiddenb[1][i].unsqueeze(0))      
      j = 0
      err_per_iter = []
      err_ext_per_iter = []
      while True:
        if self.learn_module_fcn is not None:
          #Output conv features is not provided externally
          if not self.fixed_conv: conv_out = self.learn_module_conv(im_in)
          if self.model_type == 'feed_forward': out= self.learn_module_fcn(th_curr, conv_out); hidden = None
          else: out, hidden = self.learn_module_fcn(th_curr, conv_out, hidden_curr.unsqueeze(0))

          if self.dynamics_mode == 'fix_dynamics':
            qc_inv_traj = self.qc_inv_traj
            obscov_inv_traj = self.get_covariances(out, self.dynamics_mode, self.learn_eps)
          else:
            qc_inv_traj, obscov_inv_traj = self.get_covariances(out, self.dynamics_mode)
        else:
          qc_inv_traj = self.qc_inv_traj.unsqueeze(0)
          obscov_inv_traj = self.obscov_inv_traj.unsqueeze(0)
        eps_traj = self.eps_traj.unsqueeze(0)        
        
        dtheta, err_old, err_ext_old = self.plan_layer(th_curr, start, goal, im, sdf, qc_inv_traj, obscov_inv_traj, eps_traj)
        err_per_iter.append(err_old.item())
        err_ext_per_iter.append(err_ext_old.item())
        if j == 0: err_init = err_old.item()
        th_old = th_curr
        th_new = th_curr + dtheta
        err_new = self.plan_layer.error_batch(th_new, sdf)
        err_ext_new = self.plan_layer.error_ext_batch(th_new, sdf)
        err_delta = err_new - err_old 
        err_ext_delta = err_ext_new - err_ext_old 
        j = j + 1
        
        th_curr = th_new
        if check_convergence(dtheta, j, err_delta, self.optim_params['tol_err'], self.optim_params['tol_delta'], self.optim_params['max_iters']):
          break
        if time.time() - start_t > float(self.optim_params['plan_time']):
          print('Plan time over')
          break                
      th_currb_temp[i,:,:] = th_curr
      if hiddenb is not None: 
        hidden_newb_temp[0][i] = hidden[0]
        hidden_newb_temp[1][i] = hidden[1]
      err_initb.append(err_init)
      err_finalb.append(err_new.item())
      err_per_iterb.append(err_per_iter)
      err_ext_per_iterb.append(err_ext_per_iter)
      jb.append(j)
      timeb.append(time.time() - start_t)

    th_currb = th_currb_temp
    if hiddenb is not None:
      h_newb = hidden_newb_temp[0]
      c_newb = hidden_newb_temp[1]
      hidden_newb = (h_newb, c_newb)
    else: hidden_newb = None
    return th_currb, hidden_newb, err_initb, err_finalb, err_per_iterb, err_ext_per_iterb, jb, timeb
  
  def step(self, th_currb, startb, goalb, imb, sdfb, conv_out=None, dtheta_currb=None, hiddenb=None):
    """Does one iteration of non-linear optimization on a batch of environments"""
    n_samples = th_currb.shape[0]
    dthetab_temp = torch.zeros(th_currb.shape, device=self.device)
    if hiddenb is not None: 
      hidden_newb_temp = (torch.zeros(hiddenb[0].shape, device=self.device), torch.zeros(hiddenb[1].shape, device=self.device))
    
    if self.learn_module_fcn is not None:
      #Define inputs for prediction
      if not self.fixed_conv:
        if self.sdf_predict: im_in = torch.cat((imb, sdfb), dim=1)
        else: im_in = imb
        conv_out, _ = self.learn_module_conv(im_in)
      if self.use_dtheta: th_in = torch.cat((th_currb, dtheta_currb), dim=-1)
      else: th_in = th_currb
      #Predict outputs
      if self.model_type == 'feed_forward': out = self.learn_module_fcn(th_in, conv_out); hidden = None
      else: out, hidden= self.learn_module_fcn(th_in, conv_out, hiddenb)
      #Get appropirate covariances
      if self.dynamics_mode == 'fix_dynamics':
        obscov_inv_curr = self.get_covariances(out, self.dynamics_mode, self.learn_eps); eps_traj = self.eps_traj
        qc_inv_curr = self.qc_inv_traj.unsqueeze(0).repeat(n_samples,1,1,1)
      else:
        qc_inv_curr, obscov_inv_curr = self.get_covariances(out,self.dynamics_mode, self.learn_eps)
    
    else:
      qc_inv_curr = self.qc_inv_traj.unsqueeze(0).repeat(n_samples,1,1,1) 
      obscov_inv_curr = self.obscov_inv_traj.unsqueeze(0).repeat(n_samples,1,1,1)
    
    eps_curr = self.eps_traj.unsqueeze(0).repeat(n_samples,1,1,1) 
    dthetab, err_oldb, err_ext_oldb = self.plan_layer(th_currb, startb, goalb, imb, sdfb, qc_inv_curr, obscov_inv_curr, eps_curr)
    
    if hiddenb is not None:
      hidden_newb = hidden #(h_newb, c_newb)
    else: hidden_newb = None
    return dthetab, hidden_newb, err_oldb, err_ext_oldb, qc_inv_curr, obscov_inv_curr, eps_curr 

  def error(self, th):
    err = self.plan_layer.error(th)
    return err

  def error_batch(self, thb, sdfb):
    err = self.plan_layer.error_batch(thb, sdfb)
    return err
  
  def error_ext(self, th):
    err = self.plan_layer.error_ext(th)
    return err

  def error_ext_batch(self, thb, sdfb):
    errb = self.plan_layer.error_ext_batch(thb, sdfb)
    return errb

  def unweighted_errors_batch(self, thb, sdfb):
    err_sg = self.plan_layer.start_goal_error(thb)
    err_gp = self.plan_layer.gp_error(thb)
    err_obs = self.plan_layer.obs_error(thb, sdfb) 
    # if self.non_holonomic:
    #   sg_errorb, gp_errorb, obs_errorb, dyn_errorb = self.plan_layer.unweighted_errors_batch(thb)
    #   return sg_errorb, gp_errorb, obs_errorb, dyn_errorb  
    # sg_errorb, gp_errorb, obs_errorb = self.plan_layer.unweighted_errors_batch(thb)
    return err_sg, err_gp, err_obs
  
  def linear_error(self, thb, startb, goalb, imb, sdfb, eps_trajb):
    batch_size = thb.shape[0]
    lin_err_bt = torch.zeros(batch_size, self.plan_layer.M, 1, device=self.device)
    for i in range(batch_size):
      lin_err_bt[i] = self.plan_layer.linear_error(thb[i], startb[i], goalb[i], imb[i], sdfb[i], eps_trajb[i])
    lin_err_b = lin_err_bt
    return lin_err_b

  def get_covariances(self, out, mode='diag_identity', learn_eps=False):
    num_obs_terms = self.num_obs_factors*self.robot_model.nlinks
    num_eps_terms = self.num_obs_factors*self.robot_model.nlinks
    if mode == 'fix_dynamics':
      num_gp_terms = 0
      obs_part = out[:, 0, 0:num_obs_terms].reshape(out.shape[0], self.num_obs_factors, self.robot_model.nlinks, 1)
      # print out.shape, obs_part.shape
    elif mode == "diag_identity":
      num_gp_terms = self.num_gp_factors
      qc_part     = out[:, 0, 0:num_gp_terms].reshape(out.shape[0], self.num_gp_factors, 1, 1)
      obs_part    = out[:, 0, num_gp_terms:num_gp_terms+num_obs_terms].reshape(out.shape[0], self.num_obs_factors, self.robot_model.nlinks, 1)
      # print out.shape, qc_part.shape, obs_part.shape
      eye_batch   = torch.eye(self.dof, device=self.device).unsqueeze(0).repeat(out.shape[0], self.num_gp_factors,1,1)
      qc_inv_traj = torch.mul(torch.mul(qc_part, qc_part.transpose(2,3)), eye_batch)
    elif mode == 'diag':
      num_gp_terms = self.num_gp_factors*self.dof
      qc_part     = out[:, 0, 0:num_gp_terms].reshape(out.shape[0], self.num_gp_factors, self.dof,1)
      obs_part    = out[:, 0, num_gp_terms:num_gp_terms+num_obs_terms].reshape(out.shape[0], self.num_obs_factors, self.robot_model.nlinks,1)
      qc_inv_traj = None
      raise NotImplementedError
    elif mode == "qc_full":
      num_gp_terms = self.num_gp_factors*self.dof
      qc_part     = out[:, 0, 0:num_gp_terms].reshape(out.shape[0], self.num_gp_factors, self.dof,1)
      obs_part    = out[:, 0, num_gp_terms:num_gp_terms+num_obs_terms].reshape(out.shape[0], self.num_obs_factors, self.robot_model.nlinks,1)
      qc_inv_traj = torch.mul(qc_part, qc_part.transpose(2,3))
    elif mode == "q_full":
      num_gp_terms = self.num_gp_factors*self.state_dim
      qc_part     = out[:, 0, 0:num_gp_terms].reshape(out.shape[0], self.num_gp_factors, self.state_dim, 1)
      obs_part    = out[:, 0, num_gp_terms:num_gp_terms+num_obs_terms].reshape(out.shape[0], self.num_obs_factors, self.robot_model.nlinks,1)
      qc_inv_traj = torch.mul(qc_part, qc_part.transpose(2,3))
        
    obscov_inv_traj  = torch.mul(obs_part, obs_part.transpose(2,3))
    
    if learn_eps:
      eps_part = out[:, 0, num_gp_terms+num_obs_terms:].reshape(out.shape[0], self.num_obs_factors, self.robot_model.nlinks, 1)
      eps_traj = torch.mul(eps_part, eps_part.transpose(2,3))
      if mode == 'fix_dynamics':
        return obscov_inv_traj, eps_traj
      return qc_inv_traj, obscov_inv_traj, eps_traj
    
    elif mode == 'fix_dynamics':
      return obscov_inv_traj

    return qc_inv_traj, obscov_inv_traj


  def get_obs_covariance(self, out):
    obs_inv = torch.mul(out,out).reshape(self.num_obs_factors, 1, 1) 
    obscov_inv_traj = torch.zeros(self.num_obs_factors, self.robot_model.nlinks, self.robot_model.nlinks, device=self.device) + torch.eye(self.robot_model.nlinks, device=self.device)
    obscov_inv_traj = torch.mul(obscov_inv_traj, obs_inv)
    return obscov_inv_traj


 