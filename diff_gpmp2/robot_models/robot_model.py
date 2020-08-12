#!/usr/bin/env python
import torch

class RobotModel(object):
  def __init__(self, dofs, nlinks, wksp_dim, state_dim, sphere_radii = [], batch_size=1, num_traj_states=1, use_cuda=False):
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu') 
    self.dofs =dofs
    self.nlinks =nlinks
    self.wksp_dim =wksp_dim
    self.state_dim =state_dim
    self.sphere_radii =sphere_radii
    self.batch_size = batch_size
    self.num_traj_states = num_traj_states



  def forward_kinematics(self, pose_config, vel_config=None):
    return NotImplementedError

  def get_sphere_centers(self, pose_config):
    return NotImplementedError

  def get_sphere_radii(self):
    return self.sphere_radii