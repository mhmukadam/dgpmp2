"""Obstacle factor to be applied to states with known covariances"""

import numpy as np
import torch
from .obstacle_cost import HingeLossObstacleCost
from diff_gpmp2.utils import mat_utils
from diff_gpmp2.env import Env2D

class ObstacleFactor(object):
  def __init__(self, state_dim, num_obs_factors, eps, env_params, robot_model, batch_size=1, use_cuda=False):
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu') 
    self.robot_model = robot_model
    self.num_obs_factors = num_obs_factors
    self.state_dim = state_dim
    self.env_params = env_params
    self.eps = eps
    self.obs_cost = HingeLossObstacleCost(env_params, use_cuda=self.use_cuda)
    
  def get_cov(self, idx):
    inv_cov = self.inv_cov[idx]
    cov = torch.pinverse(inv_cov)
    return cov
  
  def get_inv_cov(self, idx):
    return self.inv_cov[idx]

  def get_error_full(self, traj):
    sphere_centers, H_fk = self.robot_model.get_sphere_centers_full(traj)
    r_vec = self.robot_model.get_sphere_radii()
    error_full, H_e = self.obs_cost.hinge_loss_signed_full(sphere_centers, r_vec, self.eps)
    H1_full = torch.bmm(H_e, H_fk)
    return error_full, H1_full

  def get_error(self, trajb, sdfb):
    sphere_centersb, H_fkb = self.robot_model.get_sphere_centers_batch(trajb)
    r_vec = self.robot_model.get_sphere_radii()
    errorb, H_eb = self.obs_cost.hinge_loss_signed_batch(sphere_centersb, r_vec, self.eps, sdfb)
    H1b = torch.einsum('bsij,bsjk->bsik', H_eb, H_fkb)
    return errorb, H1b

  def set_inv_cov(self, inv_cov):
    self.inv_cov = inv_cov

  def get_inv_cov_full(self):
    return self.inv_cov

  def set_im_sdf(self, im, sdf):
    self.obs_cost.env.initialize_from_image(im, sdf)
  
  def set_eps(self, eps):
    self.eps = eps


  def eval_prob(self, state):
    error , _= self.get_error(state)
    f = np.exp(-1.0/2.0*np.transpose(error).dot(self.inv_cov).dot(error))
    norm = np.power(2.0*np.pi, self.ndims/2.0)*np.sqrt(np.linalg.det(self.inv_cov))
    f = f/norm
    return f