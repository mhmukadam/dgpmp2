#!/usr/bin/env python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LearnModuleConv(nn.Module):
  def __init__(self, learn_params, env_params, robot_model, use_cuda=False):
    super(LearnModuleConv, self).__init__()
    self.learn_params = learn_params
    self.im_size = learn_params['im_size']
    self.drop_prob = learn_params['model']['dropout_prob']
    self.robot_radius = robot_model.sphere_radii.item()
    self.use_cuda = use_cuda    
    self.in_channels = 2

    # self.conv1 = nn.Conv2d(1, 1, 3, stride=2, dilation=2, padding=2)  
    # self.conv2 = nn.Conv2d(1, 1, 3, stride=2, dilation=2, padding=2) 
    # self.conv3 = nn.Conv2d(1, 1, 3, stride=2, dilation=2, padding=2) 
    self.features = nn.Sequential(
       nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
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

#    self.features = nn.Sequential(
#      nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=2, dilation=2, padding=2),
#      nn.BatchNorm2d(16),
#      nn.ReLU(inplace=True),
#      nn.Conv2d(16, 16, kernel_size=3, stride=2, dilation=2, padding=2),
#      nn.BatchNorm2d(16),
#      nn.ReLU(inplace=True),
#      nn.Conv2d(16, 8, kernel_size=3, stride=2, dilation=2, padding=2),
#      nn.BatchNorm2d(8),
#      nn.ReLU(inplace=True),
#      nn.Conv2d(8, 8, kernel_size=3, stride=2, dilation=2, padding=2),
#      nn.BatchNorm2d(8),
#      nn.ReLU(inplace=True))


    for m in self.features.children():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, np.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    
    # for m in self.classifier.children():
    #   if isinstance(m, nn.Linear):
    #     nn.init.xavier_uniform(m.weight)
    #   elif isinstance(m, nn.BatchNorm1d):
    #     m.weight.data.fill_(1)
    #     m.bias.data.zero_()






  def forward(self, x):
    # im1 = F.relu(self.conv1(im))
    # im2 = F.relu(self.conv2(im1))
    # im3 = self.conv3(im2)
    x = self.features(x)
    x = x.view(x.size(0), -1)
    return x, ()


  def normalize_im(self, im):
    max_vec = (torch.max(torch.max(im, dim=2)[0], dim=-1)[0]).view(im.shape[0], im.shape[1], 1, 1)
    min_vec = (torch.min(torch.min(im, dim=2)[0], dim=-1)[0]).view(im.shape[0], im.shape[1], 1, 1)
    im_norm = 2.0 * (torch.div(im - min_vec, max_vec - min_vec + 1e-6) - 0.5)
    return im_norm


  
  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

  
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
