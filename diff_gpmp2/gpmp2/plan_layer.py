#!/usr/bin/env python
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gp import PriorFactor, GPFactor
from .obstacle import ObstacleFactor
from .custom_factors import NonHolonomicFactor, VelocityLimitFactor
import matplotlib.pyplot as plt
from diff_gpmp2.env import Env2D
from diff_gpmp2.utils import mat_utils, sdf_utils

class PlanLayer(nn.Module):
  def __init__(self, gp_params, obs_params, planner_params, optim_params, env_params, robot_model, learn_params=None, batch_size=1, use_cuda=False):
    super(PlanLayer, self).__init__()
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
    self.gp_params = gp_params
    self.obs_params = obs_params
    self.planner_params = planner_params
    self.optim_params = optim_params
    self.env_params = env_params
    self.robot_model = robot_model
    self.learn_params = learn_params
    self.batch_size = batch_size
    self.dof = planner_params['dof']
    self.state_dim = planner_params['state_dim']
    self.total_time_sec = planner_params['total_time_sec']   
    self.total_time_step = planner_params['total_time_step']   #Number of states in the trajectory 
    self.num_traj_states = self.total_time_step + 1  #+1 for goal at the end  
    self.dt = self.total_time_sec*1.0/self.total_time_step*1.0
    self.non_holonomic = planner_params['non_holonomic'] if 'non_holonomic' in planner_params else False
    if self.non_holonomic: self.num_dynamics_factors = self.num_traj_states 
    self.use_vel_limits = planner_params['use_vel_limits'] if 'use_vel_limits' in planner_params else False
    if self.use_vel_limits:
      self.num_vel_factors = self.num_traj_states

    #Dimensions of the linear system
    self.num_gp_factors = self.num_traj_states - 1 
    self.num_prior_factors = 2 #Prior on start and goal
    self.num_obs_factors = self.num_traj_states
    self.nlinks = self.robot_model.nlinks
    self.M = self.state_dim * (self.num_gp_factors + self.num_prior_factors) + self.num_obs_factors * self.nlinks 
    if self.non_holonomic: self.M = self.M + self.num_dynamics_factors
    if self.use_vel_limits: self.M = self.M + self.dof*self.num_vel_factors
    self.N = self.state_dim * self.num_traj_states
    if self.state_dim == 4:
      self.env = Env2D(self.env_params, self.use_cuda)
    #Create masks for forming A,b,K
    self.create_factor_masks()
    #Create factor objects
    self.gp_prior    = GPFactor(self.dof, self.dt, self.num_gp_factors, batch_size=self.batch_size, use_cuda=self.use_cuda)
    self.start_prior = PriorFactor(self.state_dim, self.gp_params['K_s'], batch_size=self.batch_size,  use_cuda=self.use_cuda)
    self.goal_prior  = PriorFactor(self.state_dim,  self.gp_params['K_g'], batch_size=self.batch_size, use_cuda=self.use_cuda)
    self.obs_factor  = ObstacleFactor(self.state_dim, self.num_obs_factors, self.obs_params['epsilon_dist'], self.env_params, self.robot_model, self.batch_size, self.use_cuda)
    if self.non_holonomic: self.dyn_factor = NonHolonomicFactor(self.dof, self.gp_params['K_d'], self.num_dynamics_factors, self.batch_size, self.use_cuda) 
    if self.use_vel_limits:
      self.vel_factor = VelocityLimitFactor(self.state_dim, self.num_vel_factors, self.gp_params['K_v'], self.batch_size, self.use_cuda)
      vx = torch.tensor(gp_params['v_x'],device=self.device)
      vy = torch.tensor(gp_params['v_y'],device=self.device)
      vx_traj = vx.unsqueeze(0).expand(self.num_vel_factors,1)
      vy_traj = vy.unsqueeze(0).expand(self.num_vel_factors,1)
      self.vel_factor.set_v_traj(vx_traj, vy_traj)
    K_s_batch = mat_utils.isotropic_matrix(1.0/torch.pow(self.gp_params['K_s'],2.0),self.state_dim,self.device).unsqueeze(0).repeat(self.batch_size,1,1)
    K_g_batch = mat_utils.isotropic_matrix(1.0/torch.pow(self.gp_params['K_g'],2.0),self.state_dim,self.device).unsqueeze(0).repeat(self.batch_size,1,1)

    self.start_prior.set_inv_cov(K_s_batch)
    self.goal_prior.set_inv_cov(K_g_batch)
    #External cost function used to measure trajectory quality
    qc_inv = self.gp_params['Q_c_inv']
    sigma  = self.obs_params['cost_sigma']
    self.qc_inv_traj_fix = torch.zeros(self.batch_size, self.num_gp_factors, self.dof, self.dof, device=self.device)
    self.qc_inv_traj_fix = self.qc_inv_traj_fix + qc_inv 
    inv_cov = mat_utils.isotropic_matrix(1.0/torch.pow(sigma,2.0), self.robot_model.nlinks, self.device)
    self.obscov_inv_traj_fix = torch.zeros(self.batch_size, self.num_traj_states, self.robot_model.nlinks, self.robot_model.nlinks, device=self.device)
    self.obscov_inv_traj_fix = self.obscov_inv_traj_fix + inv_cov
    
    self.gp_prior_fix = GPFactor(self.dof, self.dt, self.num_gp_factors, batch_size=self.batch_size, use_cuda=self.use_cuda)
    self.obs_factor_fix = ObstacleFactor(self.state_dim, self.num_obs_factors, self.obs_params['epsilon_dist'], self.env_params, self.robot_model, self.batch_size, self.use_cuda)
    self.gp_prior_fix.set_Q_c_inv(self.qc_inv_traj_fix)
    self.obs_factor_fix.set_inv_cov(self.obscov_inv_traj_fix)

    if self.learn_params is not None:
      self.dynamics_mode = self.learn_params['dgpmp2']['dynamics_mode']
    

  def forward(self, thb, startb, goalb, imb, sdfb, qc_inv_trajb, obscov_inv_trajb, eps_trajb):
    self.start_prior.set_mean(startb)
    self.goal_prior.set_mean(goalb)
    if self.learn_params is not None and self.dynamics_mode == 'q_full': self.gp_prior.set_inv_cov(qc_inv_trajb)
    else: self.gp_prior.set_Q_c_inv(qc_inv_trajb) 
    self.obs_factor.set_inv_cov(obscov_inv_trajb)
    # self.obs_factor.set_im_sdf(imb[0,:,:], sdfb[0,:,:])
    self.obs_factor.set_eps(eps_trajb)
    A, b, K = self.construct_linear_system_batch(thb, sdfb)
    dthetab = self.solve_linear_system_batch(A, b, K, delta=self.optim_params['reg'])
    err_new = self.error_batch(thb, sdfb)
    err_ext = self.error_ext_batch(thb, sdfb)
    return dthetab, err_new, err_ext
      
  def construct_linear_system_fast(self, th):
    A = torch.zeros(self.M, self.N, device=self.device)
    b = torch.zeros(self.M, 1, device=self.device)
    K = torch.zeros(self.M, self.M, device=self.device)

    err_p, H_p = self.start_prior.get_error(th[:,0])
    start_inv_cov = self.start_prior.get_inv_cov()

    err_gp, H1_gp, H2_gp = self.gp_prior.get_error(th)
    gp_inv_cov = self.gp_prior.get_inv_cov_full()

    err_g, H_g = self.goal_prior.get_error(th[:,-1])
    goal_inv_cov = self.goal_prior.get_inv_cov()

    err_obs, H_obs = self.obs_factor.get_error(th)
    obs_inv_cov = self.obs_factor.get_inv_cov_full()
    

    A.masked_scatter_(self.mask_Astart, H_p)
    b.masked_scatter_(self.mask_bstart, err_p)
    K.masked_scatter_(self.mask_Kstart, start_inv_cov)
    
    A.masked_scatter_(self.mask_A1gp, H1_gp)
    A.masked_scatter_(self.mask_A2gp, H2_gp)
    b.masked_scatter_(self.mask_bgp, err_gp)
    K.masked_scatter_(self.mask_Kgp, gp_inv_cov)

    A.masked_scatter_(self.mask_Agoal, H_g)
    b.masked_scatter_(self.mask_bgoal, err_g)
    K.masked_scatter_(self.mask_Kgoal, goal_inv_cov)

    A.masked_scatter_(self.mask_Aobs, H_obs)
    b.masked_scatter_(self.mask_bobs, err_obs)
    K.masked_scatter_(self.mask_Kobs, obs_inv_cov)

    if self.non_holonomic:
      err_dyn, H_dyn = self.dyn_factor.get_error_full(th)
      dyn_inv_cov = self.dyn_factor.get_inv_cov_full()
      A.masked_scatter_(self.mask_Adyn, H_dyn)
      b.masked_scatter_(self.mask_bdyn, err_dyn)
      K.masked_scatter_(self.mask_Kdyn, dyn_inv_cov)
    if self.use_vel_limits:
      err_vel, H_vel = self.vel_factor.get_error_full(th)
      vel_inv_cov = self.vel_factor.get_inv_cov_full()
      A.masked_scatter_(self.mask_Avel, H_vel)
      b.masked_scatter_(self.mask_bvel, err_vel)
      K.masked_scatter_(self.mask_Kvel, vel_inv_cov)

    return A, b, K


  def construct_linear_system_batch(self, thb, sdfb):
    A = torch.zeros(thb.shape[0], self.M, self.N, device=self.device)
    b = torch.zeros(thb.shape[0], self.M, 1, device=self.device)
    K = torch.zeros(thb.shape[0], self.M, self.M, device=self.device)
    
    err_p, H_p = self.start_prior.get_error(torch.index_select(thb, 1, torch.tensor(0, device=self.device) ))
    start_inv_cov = self.start_prior.get_inv_cov()

    err_gp, H1_gp, H2_gp = self.gp_prior.get_error(thb)
    gp_inv_cov = self.gp_prior.get_inv_cov_full()

    err_g, H_g = self.goal_prior.get_error(torch.index_select(thb, 1, torch.tensor(self.num_traj_states-1, device=self.device) ))
    goal_inv_cov = self.goal_prior.get_inv_cov()

    err_obs, H_obs = self.obs_factor.get_error(thb, sdfb)
    obs_inv_cov = self.obs_factor.get_inv_cov_full()

    A.masked_scatter_(self.mask_Astartb, H_p)
    b.masked_scatter_(self.mask_bstartb, err_p)
    K.masked_scatter_(self.mask_Kstartb, start_inv_cov)
    
    A.masked_scatter_(self.mask_A1gpb, H1_gp)
    A.masked_scatter_(self.mask_A2gpb, H2_gp)
    b.masked_scatter_(self.mask_bgpb, err_gp)
    K.masked_scatter_(self.mask_Kgpb, gp_inv_cov)

    A.masked_scatter_(self.mask_Agoalb, H_g)
    b.masked_scatter_(self.mask_bgoalb, err_g)
    K.masked_scatter_(self.mask_Kgoalb, goal_inv_cov)

    A.masked_scatter_(self.mask_Aobsb, H_obs)
    b.masked_scatter_(self.mask_bobsb, err_obs)
    K.masked_scatter_(self.mask_Kobsb, obs_inv_cov)

    if self.non_holonomic:
      err_dyn, H_dyn = self.dyn_factor.get_error_full(thb)
      dyn_inv_cov = self.dyn_factor.get_inv_cov_full()
      A.masked_scatter_(self.mask_Adynb, H_dyn)
      b.masked_scatter_(self.mask_bdynb, err_dyn)
      K.masked_scatter_(self.mask_Kdynb, dyn_inv_cov)
    
    if self.use_vel_limits:
      err_vel, H_vel = self.vel_factor.get_error_full(thb)
      vel_inv_cov = self.vel_factor.get_inv_cov_full()
      A.masked_scatter_(self.mask_Avelb, H_vel)
      b.masked_scatter_(self.mask_bvelb, err_vel)
      K.masked_scatter_(self.mask_Kvelb, vel_inv_cov)

    return A, b, K

  def solve_linear_system(self, A, b, K, delta = 0.0):
    I = torch.eye(self.N, self.N, device=self.device)
    A_t_K = torch.mm(A.t(), K)
    A_t_A = torch.mm(A_t_K, A)
    LAM = A_t_A + delta*I
    R = torch.mm(A_t_K, b)
    #Solve using cholesky
    l = torch.cholesky(LAM, upper=False)
    z = torch.trtrs(R, l, transpose=False, upper=False)[0]
    dtheta = torch.trtrs(z, l, transpose=True, upper=False)[0]
    return dtheta.view(self.num_traj_states, self.state_dim)

  def solve_linear_system_batch(self, A, b, K, delta = 0.0):
    I = torch.eye(self.N, self.N, device=self.device).unsqueeze(0).repeat(self.batch_size,1,1)
    # K = K + 1e-8 * torch.eye(K.shape[1], K.shape[2], device=self.device).repeat(self.batch_size,1,1)
    A_t_K = torch.bmm(A.transpose(1,2), K)
    A_t_A = torch.bmm(A_t_K, A)
    LAM = A_t_A + delta*I
    R = torch.bmm(A_t_K, b)
    #Solve using cholesky
    # l = torch.cholesky(LAM, upper=False)
    # z = torch.trtrs(R, l, transpose=False, upper=False)[0]
    # dtheta = torch.trtrs(z, l, transpose=True, upper=False)[0]

    u = torch.cholesky(LAM, upper=True)
    z = torch.bmm(torch.inverse(u.transpose(1,2)), R)
    dtheta = torch.bmm(torch.inverse(u), z) 

    # u = torch.cholesky(LAM, upper=True)
    # laminv = torch.potri(u, upper=True)
    # dtheta = torch.bmm(laminv, R)

    return dtheta.view(self.batch_size, self.num_traj_states, self.state_dim)

  def error(self, th):
    """Return the non-linear normalized error for the factor graph at the current trajectory"""
    with torch.no_grad():
      err = 0.0

      err_p, H_p = self.start_prior.get_error(th[0])
      start_inv_cov = self.start_prior.get_inv_cov()
      err = err + 0.5 * torch.mm(torch.mm(err_p.t(), self.start_prior.get_inv_cov()), err_p)

      err_gp, H1_gp, H2_gp = self.gp_prior.get_error_full(th)
      gp_inv_cov = self.gp_prior.get_inv_cov_full()
      err_gp_f = 0.5 * torch.bmm(torch.bmm(err_gp.unsqueeze(1), gp_inv_cov), err_gp.unsqueeze(-1))
      err = err + torch.sum(err_gp_f)

      err_g, H_g = self.goal_prior.get_error(th[-1])
      goal_inv_cov = self.goal_prior.get_inv_cov()
      err = err + 0.5 * torch.mm(torch.mm(err_g.t(), self.goal_prior.get_inv_cov()), err_g)
      
      err_obs, H_obs = self.obs_factor.get_error_full(th)
      obs_inv_cov = self.obs_factor.get_inv_cov_full()
      err_obs_f = 0.5 * torch.bmm(torch.bmm(err_obs.unsqueeze(1), obs_inv_cov), err_obs.unsqueeze(-1))
      err = err + torch.sum(err_obs_f)   

      if self.non_holonomic:
        err_dyn, H_dyn = self.dyn_factor.get_error_full(th)
        dyn_inv_cov = self.dyn_factor.get_inv_cov()
        err_dyn_f = 0.5 * torch.bmm(torch.bmm(err_dyn.unsqueeze(1), dyn_inv_cov), err_dyn.unsqueeze(-1))
        err = err + torch.sum(err_dyn_f)

      if self.use_vel_limits:
        err_vel, H_vel = self.vel_factor.get_error_full(th)
        vel_inv_cov = self.vel_factor.get_inv_cov_full()
        err_vel_f = 0.5 * torch.bmm(torch.bmm(err_vel.unsqueeze(1), vel_inv_cov), err_vel.unsqueeze(-1))
        err = err + torch.sum(err_vel_f)

      return err/self.M

  def error_batch(self, thb, sdfb):
    """Return the non-linear normalized error for the factor graph at the current trajectory"""
    with torch.no_grad():
      err = 0.0
      err_p, _ = self.start_prior.get_error(torch.index_select(thb, 1, torch.tensor(0, device=self.device) ))
      start_inv_cov = self.start_prior.get_inv_cov()
      
      err = err + 0.5 * torch.bmm(torch.bmm(err_p.transpose(1,2), start_inv_cov), err_p)
      err_gp, _, _ = self.gp_prior.get_error(thb)
      gp_inv_cov = self.gp_prior.get_inv_cov_full()
      err_maha_gp = 0.5 * torch.einsum('bsij,bsjk->bsik', torch.einsum('bsij,bsjk->bsik', err_gp.transpose(2,3), gp_inv_cov), err_gp) 
      err = err + torch.sum(err_maha_gp, dim=1)

      err_g, _ = self.goal_prior.get_error(torch.index_select(thb, 1, torch.tensor(self.num_traj_states-1, device=self.device) ))
      goal_inv_cov = self.goal_prior.get_inv_cov()

      err = err + 0.5 * torch.bmm(torch.bmm(err_g.transpose(1,2), goal_inv_cov), err_g)

      err_obs, _ = self.obs_factor.get_error(thb, sdfb)
      obs_inv_cov = self.obs_factor.get_inv_cov_full()
      err_maha_obs = 0.5 * torch.einsum('bsij,bsjk->bsik', torch.einsum('bsij,bsjk->bsik', err_obs.transpose(2,3), obs_inv_cov), err_obs) 
      err = err + torch.sum(err_maha_obs, dim=1)

      if self.non_holonomic:
        err_dyn, H_dyn = self.dyn_factor.get_error_full(thb)
        dyn_inv_cov = self.dyn_factor.get_inv_cov_full()
        err_maha_dyn = 0.5 * torch.einsum('bsij,bsjk->bsik', torch.einsum('bsij,bsjk->bsik', err_dyn.transpose(2,3), dyn_inv_cov), err_dyn) 
        err = err + torch.sum(err_maha_dyn, dim=1)
      
      if self.use_vel_limits:
        err_vel, H_vel = self.vel_factor.get_error_full(thb)
        vel_inv_cov = self.vel_factor.get_inv_cov_full()
        err_maha_vel = 0.5 * torch.einsum('bsij,bsjk->bsik', torch.einsum('bsij,bsjk->bsik', err_vel.transpose(2,3), vel_inv_cov), err_vel) 
        err = err + torch.sum(err_maha_vel, dim=1)
      
      return err/self.M      

  def error_ext_batch(self, thb, sdfb):
    """Return the non-linear normalized error for the factor graph at the current trajectory"""
    err = 0.0
    err_p, _ = self.start_prior.get_error(torch.index_select(thb, 1, torch.tensor(0, device=self.device) ))
    start_inv_cov = self.start_prior.get_inv_cov()

    err = err + 0.5 * torch.bmm(torch.bmm(err_p.transpose(1,2), start_inv_cov), err_p)
    err_gp, _, _ = self.gp_prior.get_error(thb)
    gp_inv_cov = self.gp_prior_fix.get_inv_cov_full()
    # err_gp = err_gp.unsqueeze()
    err_maha_gp = 0.5 * torch.einsum('bsij,bsjk->bsik', torch.einsum('bsij,bsjk->bsik', err_gp.transpose(2,3), gp_inv_cov), err_gp)
    err = err + torch.sum(err_maha_gp, dim=1)

    err_g, _ = self.goal_prior.get_error(torch.index_select(thb, 1, torch.tensor(self.num_traj_states-1, device=self.device) ))
    goal_inv_cov = self.goal_prior.get_inv_cov()

    err = err + 0.5 * torch.bmm(torch.bmm(err_g.transpose(1,2), goal_inv_cov), err_g)


    err_obs, _ = self.obs_factor.get_error(thb, sdfb)
    obs_inv_cov = self.obs_factor_fix.get_inv_cov_full()
    err_maha_obs = 0.5 * torch.einsum('bsij,bsjk->bsik', torch.einsum('bsij,bsjk->bsik', err_obs.transpose(2,3), obs_inv_cov), err_obs)
    err = err + torch.sum(err_maha_obs, dim=1)

    if self.non_holonomic:
      err_dyn, H_dyn = self.dyn_factor.get_error_full(thb)
      dyn_inv_cov = self.dyn_factor.get_inv_cov_full()
      err_maha_dyn = 0.5 * torch.einsum('bsij,bsjk->bsik', torch.einsum('bsij,bsjk->bsik', err_dyn.transpose(2,3), dyn_inv_cov), err_dyn)
      err = err + torch.sum(err_maha_dyn, dim=1)
    if self.use_vel_limits:
      err_vel, H_vel = self.vel_factor.get_error_full(thb)
      vel_inv_cov = self.vel_factor.get_inv_cov_full()
      err_maha_vel = 0.5 * torch.einsum('bsij,bsjk->bsik', torch.einsum('bsij,bsjk->bsik', err_vel.transpose(2,3), vel_inv_cov), err_vel)
      err = err + torch.sum(err_maha_vel, dim=1)

    return err/self.M

  def linear_error(self, th, start, goal, im, sdf, eps_traj):
    b = torch.zeros(self.M, 1, device=self.device)
    self.start_prior.set_mean(start)
    self.goal_prior.set_mean(goal)
    self.obs_factor.set_im_sdf(im[0,:,:], sdf[0,:,:])
    self.obs_factor.set_eps(eps_traj)

    err_p, H_p = self.start_prior.get_error(th[0])

    err_gp, H1_gp, H2_gp = self.gp_prior.get_error_full(th)

    err_g, H_g = self.goal_prior.get_error(th[-1])

    err_obs, H_obs = self.obs_factor.get_error_full(th)

    b.masked_scatter_(self.mask_bstart, err_p)
    
    b.masked_scatter_(self.mask_bgp, err_gp)

    b.masked_scatter_(self.mask_bgoal, err_g)

    b.masked_scatter_(self.mask_bobs, err_obs)
    if self.non_holonomic:
      err_dyn, H_dyn = self.dyn_factor.get_error_full(th)
      b.masked_scatter_(self.mask_bdyn, err_dyn)
    return b

  def gp_error(self, thb):
    err_gp, _, _ = self.gp_prior.get_error(thb)
    err_gp_l2 =  torch.mean(0.5 * torch.einsum('bsij,bsjk->bsik', err_gp.transpose(2,3), err_gp), dim=1)
    return err_gp_l2

  def obs_error(self, thb, sdfb):
    err_obs, _ = self.obs_factor.get_error(thb, sdfb)
    err_obs_l2 = torch.mean(0.5 * torch.einsum('bsij,bsjk->bsik', err_obs.transpose(2,3), err_obs),dim=1) 
    return err_obs_l2

  def start_goal_error(self, thb):
    err_p, _ = self.start_prior.get_error(torch.index_select(thb, 1, torch.tensor(0, device=self.device)))
    err_g, _ = self.goal_prior.get_error(torch.index_select(thb, 1, torch.tensor(self.num_traj_states-1, device=self.device) ))
    err_sg   = torch.mean(0.5 * torch.bmm(err_p.transpose(1,2), err_p) + 0.5 * torch.bmm(err_g.transpose(1,2), err_g), dim=1) 
    return err_sg

    
  def create_factor_masks(self):  
    self.mask_Astart   = torch.zeros(self.M, self.N, requires_grad=False, device=self.device).byte()
    self.mask_A1gp     = torch.zeros(self.M, self.N, requires_grad=False, device=self.device).byte()
    self.mask_A2gp     = torch.zeros(self.M, self.N, requires_grad=False, device=self.device).byte()
    self.mask_Agoal    = torch.zeros(self.M, self.N, requires_grad=False, device=self.device).byte()
    self.mask_Aobs     = torch.zeros(self.M, self.N, requires_grad=False, device=self.device).byte()

    self.mask_bstart  = torch.zeros(self.M, 1, requires_grad=False, device=self.device).byte()
    self.mask_bgp     = torch.zeros(self.M, 1, requires_grad=False, device=self.device).byte()
    self.mask_bgoal   = torch.zeros(self.M, 1, requires_grad=False, device=self.device).byte()
    self.mask_bobs    = torch.zeros(self.M, 1, requires_grad=False, device=self.device).byte()

    self.mask_Kstart  = torch.zeros(self.M, self.M, requires_grad=False, device=self.device).byte()
    self.mask_Kgp     = torch.zeros(self.M, self.M, requires_grad=False, device=self.device).byte()
    self.mask_Kgoal   = torch.zeros(self.M, self.M, requires_grad=False, device=self.device).byte()
    self.mask_Kobs    = torch.zeros(self.M, self.M, requires_grad=False, device=self.device).byte()

    self.mask_Astart[0:self.state_dim, 0:self.state_dim] = 1
    self.mask_bstart[0:self.state_dim] = 1
    self.mask_Kstart[0:self.state_dim, 0:self.state_dim] = 1

    for i in range(self.num_gp_factors):
      self.mask_A1gp[(i+1)*self.state_dim:(i+2)*self.state_dim, i*self.state_dim:(i+1)*self.state_dim] = 1
      self.mask_A2gp[(i+1)*self.state_dim:(i+2)*self.state_dim, (i+1)*self.state_dim:(i+2)*self.state_dim] = 1
      self.mask_bgp[(i+1)*self.state_dim:(i+2)*self.state_dim] = 1
      self.mask_Kgp[(i+1)*self.state_dim:(i+2)*self.state_dim, (i+1)*self.state_dim:(i+2)*self.state_dim] = 1
    
    #Goal prior factor
    offset = self.state_dim * (self.num_gp_factors + 1)
    self.mask_Agoal[offset:offset+self.state_dim, -self.state_dim:] = 1
    self.mask_bgoal[offset:offset+self.state_dim] = 1
    self.mask_Kgoal[offset:offset+self.state_dim, offset:offset+self.state_dim] = 1 

    #obstacle factors
    offset = offset + self.state_dim    
    for i in range(self.num_traj_states):
      r = offset + i*self.nlinks
      self.mask_Aobs[r: r + self.nlinks, i*self.state_dim:(i+1)*self.state_dim] = 1
      self.mask_bobs[r: r + self.nlinks] = 1
      self.mask_Kobs[r: r + self.nlinks, r: r+self.nlinks] = 1
    offset = offset + self.num_traj_states*self.nlinks

    if self.non_holonomic:
      self.mask_Adyn = torch.zeros(self.M, self.N, requires_grad=False, device=self.device).byte()
      self.mask_bdyn = torch.zeros(self.M, 1, requires_grad=False, device=self.device).byte()
      self.mask_Kdyn = torch.zeros(self.M, self.M, requires_grad=False, device=self.device).byte()
      for i in range(self.num_dynamics_factors):
        r = offset + i
        self.mask_Adyn[r, i*self.state_dim:(i+1)*self.state_dim] = 1
        self.mask_bdyn[r] = 1 
        self.mask_Kdyn[r: r + 1, r: r+1] = 1
      # offset = Update ofset here
    if self.use_vel_limits:
      self.mask_Avel = torch.zeros(self.M, self.N, requires_grad=False, device=self.device).byte()
      self.mask_bvel = torch.zeros(self.M, 1, requires_grad=False, device=self.device).byte()
      self.mask_Kvel = torch.zeros(self.M, self.M, requires_grad=False, device=self.device).byte()
      for i in range(self.num_vel_factors):
        r = offset + i*self.dof
        self.mask_Avel[r:r+self.dof, i*self.state_dim:(i+1)*self.state_dim] = 1
        self.mask_bvel[r:r+self.dof] = 1
        self.mask_Kvel[r:r+self.dof, r:r+self.dof] = 1



  ########Create 3D versions of the masks for use with batches
    self.mask_Astartb   = self.mask_Astart.unsqueeze(0).repeat(self.batch_size,1,1)
    self.mask_A1gpb     = self.mask_A1gp.unsqueeze(0).repeat(self.batch_size,1,1)
    self.mask_A2gpb     = self.mask_A2gp.unsqueeze(0).repeat(self.batch_size,1,1)
    self.mask_Agoalb    = self.mask_Agoal.unsqueeze(0).repeat(self.batch_size,1,1)
    self.mask_Aobsb     = self.mask_Aobs.unsqueeze(0).repeat(self.batch_size,1,1)

    self.mask_bstartb  = self.mask_bstart.unsqueeze(0).repeat(self.batch_size,1,1)
    self.mask_bgpb     = self.mask_bgp.unsqueeze(0).repeat(self.batch_size,1,1)
    self.mask_bgoalb   = self.mask_bgoal.unsqueeze(0).repeat(self.batch_size,1,1)
    self.mask_bobsb    = self.mask_bobs.unsqueeze(0).repeat(self.batch_size,1,1)

    self.mask_Kstartb  = self.mask_Kstart.unsqueeze(0).repeat(self.batch_size,1,1)
    self.mask_Kgpb     = self.mask_Kgp.unsqueeze(0).repeat(self.batch_size,1,1)
    self.mask_Kgoalb   = self.mask_Kgoal.unsqueeze(0).repeat(self.batch_size,1,1)
    self.mask_Kobsb    = self.mask_Kobs.unsqueeze(0).repeat(self.batch_size,1,1)
    if self.non_holonomic:
      self.mask_Adynb = self.mask_Adyn.unsqueeze(0).repeat(self.batch_size,1,1)
      self.mask_bdynb = self.mask_bdyn.unsqueeze(0).repeat(self.batch_size,1,1)
      self.mask_Kdynb = self.mask_Kdyn.unsqueeze(0).repeat(self.batch_size,1,1)

    if self.use_vel_limits:
      self.mask_Avelb = self.mask_Avel.unsqueeze(0).repeat(self.batch_size,1,1)
      self.mask_bvelb = self.mask_bvel.unsqueeze(0).repeat(self.batch_size,1,1)
      self.mask_Kvelb = self.mask_Kvel.unsqueeze(0).repeat(self.batch_size,1,1)



  # def error_ext(self, th):
  #   """
  #   Return the non-linear normalized error for the factor graph at the current trajectory
  #   """
  #   with torch.no_grad():
  #     err = 0.0
  #     err_p, H_p = self.start_prior.get_error(th[0, :])
  #     start_inv_cov = self.start_prior.get_inv_cov()
  #     err = err + 0.5 * torch.mm(torch.mm(err_p.t(), self.start_prior.get_inv_cov()), err_p)
     
  #     err_gp, H1_gp, H2_gp = self.gp_prior.get_error_full(th)
  #     gp_inv_cov = self.gp_prior_fix.get_inv_cov_full()
  #     err_gp_f = 0.5 * torch.bmm(torch.bmm(err_gp.unsqueeze(1), gp_inv_cov), err_gp.unsqueeze(-1))
  #     err = err + torch.sum(err_gp_f)
     
  #     err_g, H_g = self.goal_prior.get_error(th[-1, :])
  #     goal_inv_cov = self.goal_prior.get_inv_cov()
  #     err = err + 0.5 * torch.mm(torch.mm(err_g.t(), self.goal_prior.get_inv_cov()), err_g)
      
  #     err_obs, H_obs = self.obs_factor.get_error_full(th)
  #     obs_inv_cov = self.obs_factor_fix.get_inv_cov_full()
  #     err_obs_f = 0.5 * torch.bmm(torch.bmm(err_obs.unsqueeze(1), obs_inv_cov), err_obs.unsqueeze(-1))
  #     err = err + torch.sum(err_obs_f)     

  #     if self.non_holonomic:
  #       err_dyn, H_dyn = self.dyn_factor.get_error_full(th)
  #       dyn_inv_cov = self.dyn_factor.get_inv_cov()
  #       err_dyn_f = 0.5 * torch.bmm(torch.bmm(err_dyn.unsqueeze(1), dyn_inv_cov), err_dyn.unsqueeze(-1))
  #       err = err + torch.sum(err_dyn_f)
      
  #     if self.use_vel_limits:
  #       err_vel, H_vel = self.vel_factor.get_error_full(th)
  #       vel_inv_cov = self.vel_factor.get_inv_cov_full()
  #       err_vel_f = 0.5 * torch.bmm(torch.bmm(err_vel.unsqueeze(1), vel_inv_cov), err_vel.unsqueeze(-1))
  #       err = err + torch.sum(err_vel_f)

  #     return err/self.M
    