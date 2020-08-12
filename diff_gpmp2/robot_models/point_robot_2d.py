#!/usr/bin/env python
import torch
from .robot_model import RobotModel

class PointRobot2D(RobotModel):
  def __init__(self, sphere_radii, batch_size=1, num_traj_states=1, use_cuda=False):
    dofs = 2
    nlinks = 1 
    wksp_dim = 2
    state_dim = 4
    super(PointRobot2D, self).__init__(dofs, nlinks, wksp_dim, state_dim, sphere_radii, batch_size, num_traj_states, use_cuda)
    self.create_sphere_masks()

  def forward_kinematics(self, pose_config, vel_config=None):
    pose_wksp = pose_config
    vel_wksp = vel_config
    Jfk = torch.eye(self.state_dim, device=self.device)
    return pose_wksp, vel_wksp, Jfk
  
  def forward_kinematics_full(self, pose_config, vel_config=None):
    pose_wksp = pose_config.view(pose_config.shape[0], self.nlinks, self.wksp_dim)
    vel_wksp = vel_config
    Jfk = torch.eye(self.state_dim, device=self.device)
    Jfk = Jfk.unsqueeze(0)
    Jfk = Jfk.repeat(pose_config.shape[0], 1, 1)
    return pose_wksp, vel_wksp, Jfk

  def forward_kinematics_batch(self, pose_configb, vel_configb=None):
    pose_wksp = pose_configb.view(pose_configb.shape[0], pose_configb.shape[1], self.nlinks, self.wksp_dim)
    vel_wksp = vel_configb.view(vel_configb.shape[0], vel_configb.shape[1], self.nlinks, self.wksp_dim)
    Jfk = torch.eye(self.state_dim, device=self.device)
    Jfk = Jfk.unsqueeze(0).unsqueeze(0).repeat(pose_configb.shape[0], pose_configb.shape[1], 1, 1)
    return pose_wksp, vel_wksp, Jfk

  def get_sphere_centers(self, state):
    idx1 = torch.arange(0, self.dofs, device=self.device)
    idx2 = torch.arange(self.dofs, 2*self.dofs, device=self.device)
    idxj = torch.arange(0, self.nlinks*self.wksp_dim, device=self.device)
    pose_config = torch.index_select(state, 0, idx1)
    vel_config  = torch.index_select(state, 0, idx2)
    pose_wksp, _, j = self.forward_kinematics(pose_config, vel_config)
    jac             = torch.index_select(j, 0, idxj)
    return pose_wksp.view(self.nlinks,self.wksp_dim), jac

  def get_sphere_centers_full(self, traj):
    m1 = torch.zeros_like(traj).byte()
    m2 = torch.zeros_like(traj).byte()
    mj = torch.zeros(traj.shape[0], self.state_dim, self.state_dim, device=self.device).byte()
    m1[:,0:self.dofs] = 1
    m2[:,self.dofs:] = 1
    mj[:,0:self.nlinks*self.wksp_dim,:] = 1
    pose_config = torch.masked_select(traj, m1).view(-1, self.dofs)
    vel_config  = torch.masked_select(traj, m2).view(-1, self.dofs)
    pose_wksp, vel_wksp, J = self.forward_kinematics_full(pose_config, vel_config)
    J = torch.masked_select(J, mj).view(-1,self.nlinks*self.wksp_dim,self.state_dim)
    return pose_wksp, J

  def get_sphere_centers_batch(self, trajb):
    pose_config = torch.masked_select(trajb, self.m1).view(trajb.shape[0], trajb.shape[1], self.dofs)
    vel_config  = torch.masked_select(trajb, self.m2).view(trajb.shape[0], trajb.shape[1], self.dofs)
    pose_wksp, vel_wksp, J = self.forward_kinematics_batch(pose_config, vel_config)
    J = torch.masked_select(J, self.mj).view(trajb.shape[0], trajb.shape[1], self.nlinks*self.wksp_dim,self.state_dim)
    return pose_wksp, J

  def create_sphere_masks(self):
    self.m1 = torch.zeros(self.batch_size, self.num_traj_states, self.state_dim).byte()
    self.m2 = torch.zeros(self.batch_size, self.num_traj_states, self.state_dim).byte()
    self.mj = torch.zeros(self.batch_size, self.num_traj_states, self.state_dim, self.state_dim, device=self.device).byte()
    self.m1[:,:,0:self.dofs] = 1
    self.m2[:,:,self.dofs:] = 1
    self.mj[:,:,0:self.nlinks*self.wksp_dim,:] = 1