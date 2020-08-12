#!/usr/bin/env python
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class InitNet(nn.Module):
  def __init__(self, in_channels, im_size, num_states, state_dim):
    super(InitNet, self).__init__()
    self.in_channels = in_channels
    self.num_states = num_states
    self.state_dim = state_dim
    self.im_size = im_size
    self.features = nn.Sequential(
      nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(16, 16, kernel_size=3, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(16, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      # nn.MaxPool2d(kernel_size=2, stride=2)
    )
      
    self.classifier = nn.Sequential(
      nn.Dropout(p = 0.5),
      nn.Linear(32 * self.im_size/16 * self.im_size/16 + self.num_states*self.state_dim, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(inplace=True),
      nn.Dropout(p = 0.5),
      nn.Linear(512, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(inplace=True),
      nn.Dropout(p = 0.5),
      nn.Linear(512, (self.num_states-2)*self.state_dim),
    )
      
    for m in self.features.children():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, np.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    
    for m in self.classifier.children():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
      elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
              

  def forward(self, x, th):#startb, goalb):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    #conc = torch.cat((x,startb.view(startb.size(0),-1),goalb.view(goalb.size(0),-1)),dim=1)
    conc = torch.cat((x, th.view(th.size(0),-1)), dim=1)
    out = self.classifier(conc)
    traj = self.get_traj(out)#, startb, goalb)
    return traj    

  def get_traj(self, out):#, startb, goalb):
    out = out.reshape(-1, self.num_states-2, self.state_dim)
    z = torch.zeros(out.size(0), 1, self.state_dim)
    traj = torch.cat((z, out, z), dim=1)
    return traj


  def print_gradients(self):
    for name, param in self.named_parameters():
      if param.requires_grad:
        print("Layer = {}, grad norm = {}".format(name, param.grad.data.norm(2).item()))
  
  def get_gradient_dict(self):
    grad_dict = {}
    for name, param in self.named_parameters():
      if param.requires_grad:
        grad_dict[name] = param.grad.data.norm(2).item()
    return grad_dict
    
  def print_parameters(self):
    for name, param in self.named_parameters():
      if param.requires_grad:
        print(name, param.data)  
