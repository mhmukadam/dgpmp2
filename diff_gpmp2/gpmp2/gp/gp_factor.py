#!/usr/bin/env python
"""The constant velocity GP prior"""
import numpy as np
import torch
import time
from torch.autograd import Variable

class GPFactor(object):
  def __init__(self, dof, delta_t, num_gp_factors, batch_size=1, use_cuda=False):
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu') 
    self.dof = dof
    self.delta_t = delta_t
    self.state_dim = self.dof * 2 #position+velocity
    self.A, self.u, self.F = self.calc_system_matrices()
    self.num_gp_factors = num_gp_factors
    self.idx1 = torch.arange(0, self.num_gp_factors, device=self.device)
    self.idx2 = torch.arange(1, self.num_gp_factors+1, device=self.device)
    # self.phi = self.calc_phi()


  def calc_system_matrices(self):
    A = torch.zeros(self.state_dim, self.state_dim, device=self.device)
    I  = torch.eye(self.dof, device=self.device)
    A[0:self.dof, self.dof:2*self.dof] = I
    u = 0.0
    F = torch.zeros(self.state_dim, self.dof, device=self.device)
    F[self.dof:2*self.dof, :] = I
    return A, u, F

  def calc_phi(self):
    I = torch.eye(self.dof, device=self.device)
    Z = torch.zeros(self.dof, self.dof, device=self.device)
    phi_u = torch.cat((I, self.delta_t*I),dim=1)
    phi_l = torch.cat((Z, I),dim=1)
    phi = torch.cat((phi_u, phi_l),dim=0)
    return phi

  def calc_Q(self):
    Q = torch.zeros(self.state_dim, self.state_dim, device=self.device)
    Q[0:self.dof, 0:self.dof] = (1.0/3.0) * np.power(self.delta_t, 3) * self.Q_c
    Q[0:self.dof, self.dof:2*self.dof] = (1.0/2.0) * np.power(self.delta_t, 2) * self.Q_c
    Q[self.dof:2*self.dof, 0:self.dof] = (1.0/2.0) * np.power(self.delta_t, 2) * self.Q_c
    Q[self.dof:2*self.dof, self.dof:2*self.dof] = self.delta_t * self.Q_c
    return Q
    
  def calc_Q_inv(self,idx):
    Q_inv = torch.zeros(self.state_dim, self.state_dim, device=self.device)
    Q_inv[0:self.dof, 0:self.dof] = 12.0 * np.power(self.delta_t, -3.0) * self.Q_c_inv[idx]
    Q_inv[0:self.dof, self.dof:2*self.dof] = -6.0 * np.power(self.delta_t, -2.0) * self.Q_c_inv[idx]
    Q_inv[self.dof:2*self.dof, 0:self.dof] = -6.0 * np.power(self.delta_t, -2.0) * self.Q_c_inv[idx]
    Q_inv[self.dof:2*self.dof, self.dof:2*self.dof] = 4.0 * np.power(self.delta_t, -1.0) * self.Q_c_inv[idx]
    return Q_inv
  
  def calc_Q_inv_full(self):
    m1 = 12.0 * (self.delta_t ** -3.0) * self.Q_c_inv
    m2 = -6.0 * (self.delta_t ** -2.0) * self.Q_c_inv
    m3 = 4.0 *  (self.delta_t ** -1.0) * self.Q_c_inv
    
    Q_inv_u = torch.cat((m1,m2), dim=2)
    Q_inv_l = torch.cat((m2,m3), dim=2)
    Q_inv   = torch.cat((Q_inv_u,Q_inv_l), dim=1)
    return Q_inv

  def calc_Q_inv_batch(self):
    m1 = 12.0 * (self.delta_t ** -3.0) * self.Q_c_inv
    m2 = -6.0 * (self.delta_t ** -2.0) * self.Q_c_inv
    m3 = 4.0 *  (self.delta_t ** -1.0) * self.Q_c_inv
    
    Q_inv_u = torch.cat((m1,m2), dim=-1)
    Q_inv_l = torch.cat((m2,m3), dim=-1)
    Q_inv   = torch.cat((Q_inv_u,Q_inv_l), dim=-2)
    return Q_inv

  # def get_error(self, traj):#state_1, state_2):
  #   idx1 = torch.arange(0, self.num_gp_factors, device=self.device)
  #   idx2 = torch.arange(1, self.num_gp_factors+1, device=self.device)
  #   state_1 = torch.index_select(traj, 0, idx1)#traj[:-1]
  #   state_2 = torch.index_select(traj, 0, idx2)#traj[1:]
  #   phi = self.calc_phi()
  #   error = state_2.t() - torch.mm(phi, state_1.t())#(self.calc_phi().numpy()).dot(state_1) 
  #   H1 = phi
  #   H2 = -1.0 * torch.eye(self.state_dim, device=self.device)
  #   return error.t(), H1, H2 #.reshape(self.state_dim, 1)

  def get_error_full(self, traj):

    idx1 = torch.arange(0, self.num_gp_factors, device=self.device)
    idx2 = torch.arange(1, self.num_gp_factors+1, device=self.device)
    phi = self.calc_phi()
    state_1 = torch.index_select(traj, 0, idx1)#traj[:-1]
    state_2 = torch.index_select(traj, 0, idx2)#traj[1:]
    error = state_2.t() - torch.mm(phi, state_1.t())#(self.calc_phi().numpy()).dot(state_1) 
    
    H1_full = phi.unsqueeze(0).repeat(self.num_gp_factors, 1, 1)
    H2_full = -1.0 * torch.eye(self.state_dim, device=self.device).unsqueeze(0).repeat(self.num_gp_factors, 1, 1)
    return error.t(), H1_full, H2_full


  def get_error(self, trajb):

    phi = self.calc_phi().unsqueeze(0).repeat(trajb.shape[0], 1, 1)
    state_1 = torch.index_select(trajb, 1, self.idx1)
    state_2 = torch.index_select(trajb, 1, self.idx2)
    error = state_2.transpose(1,2) - torch.bmm(phi, state_1.transpose(1,2))#(self.calc_phi().numpy()).dot(state_1) 
    
    H1_full = phi.unsqueeze(1).repeat(1, self.num_gp_factors, 1, 1)
    H2_full = -1.0 * torch.eye(self.state_dim, device=self.device).unsqueeze(0).unsqueeze(0).repeat(trajb.shape[0], self.num_gp_factors, 1, 1)
    
    return error.transpose(1,2).unsqueeze(-1), H1_full, H2_full



  def get_cov(self, idx):
    return self.calc_Q(idx)

  def get_inv_cov(self, idx):
    return self.calc_Q_inv(idx)

  def get_inv_cov_full(self):
    return self.Q_inv


  def set_Q_c_inv(self, Q_c_inv):
    self.Q_c_inv = Q_c_inv
    self.Q_inv = self.calc_Q_inv_batch()

  def set_inv_cov(self, Q_inv):
    self.Q_inv = Q_inv

  # def eval_prob(self, state_1, state_2):
  #   error, _, _ = self.get_error(state_1, state_2)
  #   f = np.exp(-1.0/2.0*np.transpose(error).dot(self.Qinv).dot(error))
  #   norm = np.power(2.0*np.pi, self.state_dim/2.0)*np.sqrt(np.linalg.det(self.Q))
  #   f = f/norm
  #   return f