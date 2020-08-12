#!/usr/bin/env python
import torch
from .robot_model import RobotModel

class PointRobotXYH(RobotModel):
  def __init__(self, sphere_radii, use_cuda=False):
    dofs = 3
    nlinks = 1 
    wksp_dim = 2
    state_dim = 6
    super(PointRobotXYH, self).__init__(dofs, nlinks, wksp_dim, state_dim, sphere_radii, use_cuda)
    

  def forward_kinematics(self, pose_config, vel_config=None):
    pose_wksp = pose_config
    vel_wksp = vel_config
    Jfk = torch.eye(self.state_dim, device=self.device)
    return pose_wksp, vel_wksp, Jfk
  
  def forward_kinematics_full(self, pose_config, vel_config=None):
    pose_x = torch.index_select(pose_config, 1, torch.tensor(0))
    pose_y = torch.index_select(pose_config, 1, torch.tensor(1))
    vel_x  = torch.index_select(vel_config, 1, torch.tensor(0))
    vel_y  = torch.index_select(vel_config, 1, torch.tensor(1))
    pose_wksp = torch.cat((pose_x, pose_y), dim=-1)
    vel_wksp  = torch.cat((vel_x, vel_y), dim=-1)
    
    Jfk = torch.zeros(2*self.wksp_dim, self.state_dim, device=self.device)
    J1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=self.device)
    z1 = torch.zeros(self.wksp_dim, self.dofs, device=self.device)
    z2 = torch.zeros(self.wksp_dim, self.dofs, device=self.device)
    J2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=self.device)
    Jfk1 = torch.cat((J1, z1), dim=-1)
    Jfk2 = torch.cat((z2, J2), dim=-1)
    Jfk  = torch.cat((Jfk1, Jfk2), dim=0)
    Jfk = Jfk.repeat(pose_config.shape[0], 1, 1)
    
    return pose_wksp.view(pose_config.shape[0], 1, self.wksp_dim), vel_wksp, Jfk

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
    mj = torch.zeros(traj.shape[0], 2*self.wksp_dim, self.state_dim, device=self.device).byte()
    m1[:,0:self.dofs] = 1
    m2[:,self.dofs:] = 1
    mj[:,0:self.nlinks*self.wksp_dim,:] = 1
    pose_config = torch.masked_select(traj, m1).view(-1,self.dofs)
    vel_config  = torch.masked_select(traj, m2).view(-1,self.dofs)
    pose_wksp, vel_wksp, J = self.forward_kinematics_full(pose_config, vel_config)
    J = torch.masked_select(J, mj).view(-1,self.nlinks*self.wksp_dim,self.state_dim)
    return pose_wksp, J