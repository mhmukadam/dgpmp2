#!/usr/bin/env python
import numpy as np
import torch
import time
from .gp import PriorFactor, GPFactor
from .obstacle import ObstacleFactor
from diff_gpmp2.utils.planner_utils import check_convergence
from diff_gpmp2.utils import mat_utils


class GPMP2Planner(object):
  def __init__(self, gp_params, obs_params, planner_params, env_params, robot, use_cuda=False):
    """
      The GPMP2 Planner class
      
    """
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu') 

    # self.env = env
    self.robot = robot
    self.gp_params = gp_params
    self.obs_params = obs_params
    self.env_params = env_params
    self.dof = planner_params['dof']
    self.state_dim = planner_params['state_dim']
    self.total_time_sec = planner_params['total_time_sec']   
    self.total_time_step = planner_params['total_time_step']   #Number of states in the trajectory 
    self.total_check_step = planner_params['total_check_step'] 
    self.use_gp_inter = planner_params['use_gp_inter']
    self.num_traj_states = self.total_time_step + 1  #+1 for goal at the end  
    self.dt = self.total_time_sec*1.0/self.total_time_step*1.0
    self.check_inter = (self.total_check_step/self.total_time_step) - 1.0;

    #Dimensions of the linear system
    self.num_gp_factors = self.num_traj_states - 1 
    self.num_prior_factors = 2 #Prior on start and goal
    if self.use_gp_inter:
      self.num_obs_factors = self.num_traj_states + self.num_traj_states*self.total_check_step
    else:
      self.num_obs_factors = self.num_traj_states
    self.nlinks = self.robot.nlinks
    self.M = self.state_dim * (self.num_gp_factors + self.num_prior_factors) + self.num_obs_factors * self.nlinks 
    self.N = self.state_dim * self.num_traj_states

    qc_inv      = self.gp_params['Q_c_inv']
    qc_inv_traj = torch.zeros(self.num_gp_factors, self.dof, self.dof, device=self.device)
    qc_inv_traj = qc_inv_traj + qc_inv
    inv_cov = mat_utils.isotropic_matrix(1.0/torch.pow(self.obs_params['cost_sigma'],2.0), self.robot.nlinks, self.device)
    inv_obscov_traj = torch.zeros(self.num_traj_states, self.robot.nlinks, self.robot.nlinks, device=self.device)
    inv_obscov_traj = inv_obscov_traj + inv_cov

    self.gp_prior = GPFactor(self.dof, self.dt, self.num_gp_factors, self.use_cuda)
    self.start_prior = PriorFactor(self.state_dim, self.gp_params['K_s'])
    self.goal_prior  = PriorFactor(self.state_dim,  self.gp_params['K_g'])
    self.obs_factor = ObstacleFactor(self.state_dim, self.num_obs_factors, self.obs_params['epsilon_dist'], self.env_params, self.robot, self.use_cuda)
    self.gp_prior.set_Q_c_inv(qc_inv_traj) 
    self.obs_factor.set_inv_cov(inv_obscov_traj)



  def plan(self, start, goal, th_init, im, sdf, optim_params = {'method': 'gauss_newton', 'plan_time':np.inf, 'max_iters':np.inf, 'tol_err':1e-2, 'tol_delta':1e-3, 'reg':0.0}):
    assert th_init.shape[0] == self.num_traj_states and th_init.shape[1] == self.state_dim
    self.obs_factor.set_im_sdf(im, sdf)
    self.start_prior.set_mean(start)
    self.goal_prior.set_mean(goal)
    start_t = time.time()
    th_curr = th_init
    # print th_init.grad
    j = 0 #Current iteration counter
    err_per_iter = []
    err_init = self.error(th_init)
    err_old = err_init
    lam = 1e-4 #Used for LM method

    while True:
      print ("Current iteration, %d"%j)
      #Create linear system
      A, b, K = self.create_Ab_linear(th_curr)
      err_per_iter.append(err_old)
      if optim_params['method'] == 'gauss_newton':
        #get parameter update
        delta = optim_params['reg'] 
        dtheta = self.solve_linear_system(A, b, K, delta=delta, trust_region=False, method='chol')
        th_new = th_curr + dtheta
        err_new = self.error(th_new)
        err_delta = err_new - err_old
        th_curr = th_new
        err_old = err_new
        print('dtheta = %f, err_delta = %f'%(torch.norm(dtheta), err_delta)) 

      elif optim_params['method'] == 'lm':
        dtheta = self.solve_linear_system(A, b, K, delta=lam, trust_region=True, method='chol')
        th_new = th_curr + dtheta
        err_new = self.error(th_new)
        err_delta = err_new - err_old
        if err_delta < 0:
          th_curr = th_new
          lam = lam/10.0
          err_old = err_new
        else:
          lam = lam*10.0
        print('dtheta = %f, err_delta = %f, lambda= %f'%(torch.norm(dtheta), err_delta, lam)) 

      j = j + 1
      #Check convergence
      if check_convergence(dtheta, j, err_delta, optim_params['tol_err'], optim_params['tol_delta'], optim_params['max_iters']):
        break
      if time.time() - start_t > optim_params['plan_time']:
        print('Plan time over')
        break

    time_taken = time.time() - start_t
    return th_curr, err_init, err_new, err_per_iter, j, time_taken


  def create_Ab_linear(self, th):
    temp_A = torch.zeros(self.M, self.N, device=self.device)
    temp_b = torch.zeros(self.M, 1, device=self.device)
    temp_K = torch.zeros(self.M, self.M, device=self.device)
    
    err_p, H_p = self.start_prior.get_error(th[0, :])
    temp_A[0:self.state_dim,0:self.state_dim] = H_p
    temp_b[0:self.state_dim] = err_p
    temp_K[0:self.state_dim, 0:self.state_dim] = self.start_prior.get_inv_cov()

    #GP factors
    err_gp, H1_gp, H2_gp = self.gp_prior.get_error(th)
    for i in range(self.num_gp_factors):
      temp_A[(i+1)*self.state_dim:(i+2)*self.state_dim, i*self.state_dim:(i+1)*self.state_dim] = H1_gp
      temp_A[(i+1)*self.state_dim:(i+2)*self.state_dim, (i+1)*self.state_dim:(i+2)*self.state_dim] = H2_gp
      temp_b[(i+1)*self.state_dim:(i+2)*self.state_dim] = err_gp[i].unsqueeze(-1)
      temp_K[(i+1)*self.state_dim:(i+2)*self.state_dim, (i+1)*self.state_dim:(i+2)*self.state_dim] = self.gp_prior.get_inv_cov(i)#scipy.linalg.sqrtm(inv_cov)

    #Goal prior factor
    offset = self.state_dim * (self.num_gp_factors + 1)
    err_g, H_g = self.goal_prior.get_error(th[-1, :])
    temp_A[offset:offset+self.state_dim,-self.state_dim:] = H_g
    temp_b[offset:offset+self.state_dim] = err_g
    temp_K[offset:offset+self.state_dim, offset:offset+self.state_dim] = self.goal_prior.get_inv_cov()

    #obstacle factors
    offset = offset + self.state_dim
    for i in range(self.num_traj_states):
      err_obs, H_obs = self.obs_factor.get_error(th[i, :])
      r = offset + i*self.nlinks
      temp_A[r: r + self.nlinks, i*self.state_dim:(i+1)*self.state_dim] = H_obs
      temp_b[r: r + self.nlinks] = err_obs
      temp_K[r: r + self.nlinks, r: r+self.nlinks] = self.obs_factor.get_inv_cov(i)
    return temp_A, temp_b, temp_K

  def error(self, th):
    """Return the non-linear normalized error for the factor graph at the current trajectory"""
    err = 0.0
    err_p, _ = self.start_prior.get_error(th[0, :])
    err = err + 0.5 * torch.mm(torch.mm(err_p.t(), self.start_prior.get_inv_cov()), err_p)

    #GP factors
    err_gp, _, _ = self.gp_prior.get_error(th)
    for i in range(self.num_gp_factors):
      # err_gp, _, _ = self.gp_prior.get_error(th[i, :], th[i+1, :])
      # print err_gp[i,:].shape
      err_gp_curr = err_gp[i].unsqueeze(-1)
      err = err + 0.5 * torch.mm(torch.mm(err_gp_curr.t(), self.gp_prior.get_inv_cov(i)), err_gp_curr)

    #obstacle factors
    for i in range(self.num_traj_states):
      err_obs, H_obs = self.obs_factor.get_error(th[i, :])
      err = err + 0.5 * torch.mm(torch.mm(err_obs.t(), self.obs_factor.get_inv_cov(i)), err_obs)

    #Goal prior factor
    err_g, H_g = self.goal_prior.get_error(th[-1, :])
    err = err + 0.5 * torch.mm(torch.mm(err_g.t(), self.goal_prior.get_inv_cov()), err_g)

    return err/self.M

  def solve_linear_system(self, A, b, K, delta=0.0, trust_region=False, method='chol'):
    I = torch.eye(self.N, self.N, device=self.device)
    A_t_K = torch.mm(A.t(), K)
    A_t_A = torch.mm(A_t_K, A) 
    if not trust_region:
      LAM = A_t_A + delta*I
    else:
      LAM = A_t_A + delta*(np.diag(np.diag(A_t_A)))
    R = torch.mm(A_t_K, b)
    if method=='chol':
      l = torch.cholesky(LAM, upper=False)
      z = torch.trtrs(R, l, transpose=False, upper=False)[0]
      dtheta = torch.trtrs(z, l, transpose=True, upper=False)[0]
      # dtheta = torch.potrs(R, l, upper=False)
    else:
      raise NotImplementedError
    return dtheta.view(self.num_traj_states, self.state_dim)

  # def check_convergence(self, dtheta, j, err_delta, tol_err, tol_delta, max_iters):
  #   if torch.norm(dtheta) < tol_delta:
  #     print('Update got too small at iter %d: %f'%(j,torch.norm(dtheta)))
  #     return True
  #   if torch.norm(err_delta) < tol_err:
  #     print('Difference in error got too small at iter %d: %f'%(j ,torch.norm(err_delta)))
  #     return True
  #   if j >= max_iters:
  #     print('Max iters done')
  #     return True
  #   return False

  def step(self, th_curr, start, goal, optim_params={'method': 'gauss_newton', 'plan_time':np.inf, 'max_iters':np.inf, 'tol_err':1e-2, 'tol_delta':1e-3, 'reg':0.0}):
    """Does one iteration of non-linear optimization on one environment"""
    self.start_prior = PriorFactor(self.state_dim, start, self.gp_params['K_s'])
    self.goal_prior  = PriorFactor(self.state_dim, goal,  self.gp_params['K_g'])
    err_old = self.error(th_curr)
    #Create linear system
    A, b, K = self.create_Ab_linear(th_curr)
    if optim_params['method'] == 'gauss_newton':
      #get parameter update
      delta = optim_params['reg'] 
      dtheta = self.solve_linear_system(A, b, K, delta=delta, trust_region=False, method='chol')

    elif optim_params['method'] == 'lm':
      dtheta = self.solve_linear_system(A, b, K, delta=lam, trust_region=True, method='chol')
      # err_delta = err_new - err_old
      # if err_delta < 0:
      #   # th_curr = th_new
      #   lam = lam/10.0
      #   err_old = err_new
      # else:
      #   dtheta = 0.0
      #   lam = lam*10.0
    
    # print('|dtheta| = %f'%(torch.norm(dtheta))) 
    # th_curr = th_curr + dtheta
    return dtheta, err_old