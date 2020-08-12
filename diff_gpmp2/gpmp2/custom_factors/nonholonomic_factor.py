#!/usr/bin/env python
"""Factor that imposes non-holonomic constraints
"""
import torch
from diff_gpmp2.utils import mat_utils

class NonHolonomicFactor(object):
  def __init__(self, dof, sig, num_dyn_factors, batch_size=1, use_cuda=False):
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu') 
    self.dof = dof
    self.num_dyn_factors = num_dyn_factors
    self.cov = mat_utils.isotropic_matrix(torch.pow(sig,2.0), 1, self.device).unsqueeze(0).repeat(self.num_dyn_factors,1,1)
    self.inv_cov = mat_utils.isotropic_matrix(1.0/torch.pow(sig,2.0), 1,self.device).unsqueeze(0).repeat(self.num_dyn_factors,1,1)

  def get_error_full(self, traj):
    vx = torch.index_select(traj, 1, torch.tensor(3))#column vector
    vy = torch.index_select(traj, 1, torch.tensor(4))#column vector
    th = torch.index_select(traj, 1, torch.tensor(2))#column vector
    err = vy * torch.cos(th) - vx * torch.sin(th)
    
    h1 = torch.zeros(traj.shape[0], 2, device = self.device)
    h2 = -vy * torch.sin(th) + vx * torch.cos(th)
    h3 = torch.cat((-torch.sin(th), torch.cos(th)), -1) 
    h4 = torch.zeros(traj.shape[0], 1)
    Hp = torch.cat((h1, h2), -1)
    Hv = torch.cat((h3, h4), -1)

    H1 = torch.cat((Hp, Hv), -1) 
    return err, H1
  
  def get_cov(self):
    return self.cov
  
  def get_inv_cov(self):
    return self.inv_cov

  def get_inv_cov_full(self):
    return self.inv_cov

