#!/usr/bin/env python
import numpy as np
import torch
from diff_gpmp2.env import Env2D
from diff_gpmp2.utils.sdf_utils import bilinear_interpolate

class HingeLossObstacleCost(object):
  def __init__(self, env_params, batch_size=1, use_cuda=False):
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu') 
    self.env_params = env_params
    self.env = Env2D(env_params, use_cuda=self.use_cuda)

  def hinge_loss_signed(self, sphere_centers, r_vec, eps):
    eps_tot = eps + r_vec
    dist_signed, J = self.env.get_signed_obstacle_distance_vec(sphere_centers)
    cost = torch.where(dist_signed <= eps_tot, eps_tot - dist_signed, torch.zeros(dist_signed.shape, device=self.device))
    H = torch.where(dist_signed <= eps_tot, -1.0*J, torch.zeros(J.shape, device=self.device))
    return cost, H

  def hinge_loss_signed_full(self, sphere_centers, r_vec, eps):
    eps_tot = eps + r_vec
    z2 = torch.zeros(1, sphere_centers.shape[-1], device=self.device)
    dist_signed, J = self.env.get_signed_obstacle_distance(sphere_centers.view(sphere_centers.shape[0]*sphere_centers.shape[1], 1, -1))
    cost = torch.where(dist_signed <= eps_tot, eps_tot - dist_signed, torch.tensor(0.0,device=self.device))
    H = torch.where(dist_signed <= eps_tot, -1.0*J, z2)
    return cost.reshape(sphere_centers.shape[0], sphere_centers.shape[1]), H

  def hinge_loss_signed_batch(self, sphere_centersb, r_vec, epsb, sdfb):
    eps_tot = epsb + r_vec
    z2 = torch.zeros(1, sphere_centersb.shape[-1], device=self.device)
    qpts = sphere_centersb.view(sphere_centersb.shape[0], sphere_centersb.shape[1]*sphere_centersb.shape[2], -1)
    eps_tot = eps_tot.view(eps_tot.shape[0], eps_tot.shape[1]*eps_tot.shape[2], 1) 
    res = (self.env_params['x_lims'][1] - self.env_params['x_lims'][0])/(sdfb.shape[-1])
    dist_signed, J = bilinear_interpolate(sdfb[:,0,:,:], qpts, res, self.env_params['x_lims'], self.env_params['y_lims'], self.use_cuda)
    cost = torch.where(dist_signed <= eps_tot, eps_tot - dist_signed, torch.tensor(0.0,device=self.device))
    H = torch.where(dist_signed <= eps_tot, -1.0*J, z2)
    return cost.view(sphere_centersb.shape[0], sphere_centersb.shape[1], sphere_centersb.shape[2], 1), H.view(sphere_centersb.shape)    


