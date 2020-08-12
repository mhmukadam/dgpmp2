#!/usr/bin/env python
"""Velocity Limit Factor
"""
import torch
from diff_gpmp2.utils import mat_utils

class VelocityLimitFactor(object):
  def __init__(self, ndims, num_vel_factors, sig, batch_size=1, use_cuda=False):
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu') 
    self.num_vel_factors = num_vel_factors
    self.ndims = torch.tensor(ndims,device=self.device)
    self.cov = mat_utils.isotropic_matrix(torch.pow(sig,2.0),self.ndims, self.device)
    self.batch_size = batch_size
    self.inv_cov = mat_utils.isotropic_matrix(1.0/torch.pow(sig,2.0),self.ndims/2,self.device).unsqueeze(0).repeat(self.num_vel_factors,1,1)

  def get_error_full(self, traj):
    vx = torch.index_select(traj, 1, torch.tensor(2))#column vector
    vy = torch.index_select(traj, 1, torch.tensor(3))#column vector
    
    cost_x = torch.where(torch.abs(vx) >= self.vx_traj, torch.abs(vx) - self.vx_traj, torch.tensor(0.0, device = self.device))
    H_x    = torch.where(torch.abs(vx) >= self.vx_traj, -torch.sign(vx) * torch.tensor([[1.0, 0.0]], device=self.device), torch.tensor([[0.0, 0.0]], device=self.device)).reshape(traj.shape[0], 1, self.ndims/2)
    cost_y = torch.where(torch.abs(vy) >= self.vy_traj, torch.abs(vy) - self.vy_traj, torch.tensor(0.0, device = self.device))
    H_y    = torch.where(torch.abs(vy) >= self.vy_traj, -torch.sign(vy) * torch.tensor([[0.0, 1.0]], device=self.device), torch.tensor([[0.0, 0.0]], device=self.device)).reshape(traj.shape[0], 1, self.ndims/2)
    cost   = torch.cat((cost_x, cost_y), dim=1)
    Z      = torch.zeros(self.ndims/2, self.ndims/2, device=self.device).unsqueeze(0).repeat(traj.shape[0], 1,1)
    H_xy   = torch.cat((H_x, H_y), dim=1)
    H = torch.cat((Z, H_xy), dim=-1)
    return cost, H
  


  def get_cov(self):
    return self.cov
  
  def get_inv_cov_full(self):
    return self.inv_cov

  def set_v_traj(self, vx_traj, vy_traj):
    self.vx_traj = vx_traj
    self.vy_traj = vy_traj

  def hinge_loss_signed_full(self, sphere_centers, r_vec, eps):
    eps_tot = eps + r_vec
    z2 = torch.zeros(1, sphere_centers.shape[-1], device=self.device)
    dist_signed, J = self.env.get_signed_obstacle_distance(sphere_centers.view(sphere_centers.shape[0]*sphere_centers.shape[1], 1, -1))
    cost = torch.where(dist_signed <= eps_tot, eps_tot - dist_signed, torch.tensor(0.0,device=self.device))
    H = torch.where(dist_signed <= eps_tot, -1.0*J, z2)
    return cost.reshape(sphere_centers.shape[0], sphere_centers.shape[1]), H
