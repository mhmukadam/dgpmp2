#!/usr/bin/env python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from diff_gpmp2.utils.learn_utils import torch_activation
from diff_gpmp2.utils.sdf_utils import bilinear_interpolate, costmap_2d

class LearnModuleFCN(nn.Module):
  def __init__(self, learn_params, env_params, obs_params, robot_model, use_cuda=False):
    super(LearnModuleFCN, self).__init__()
    self.learn_params = learn_params
    self.num_traj_states = learn_params['num_traj_states']
    self.state_dim = learn_params['state_dim']
    self.im_size = learn_params['im_size']
    self.out_dim = learn_params['out_dim']
    self.drop_prob = learn_params['model']['dropout_prob']
    self.env_params = env_params
    self.obs_params = obs_params
    self.res = (self.env_params['x_lims'][1] - self.env_params['x_lims'][0])/(self.learn_params['data']['im_size']*1.0)
    self.robot_radius = robot_model.sphere_radii.item()
    self.safety_dist = obs_params['epsilon_dist'] + self.robot_radius
    self.use_dtheta = learn_params['dgpmp2']['dtheta_predict'] if 'dtheta_predict' in learn_params['dgpmp2'] else False 
    self.model_type = learn_params['model']['type'] 
    self.costmap_predict = learn_params['dgpmp2']['costmap_predict'] if 'costmap_predict' in learn_params['dgpmp2'] else False
    self.use_cuda = use_cuda    

    # self.dropout = nn.Dropout(self.drop_prob)

    if not self.use_dtheta: self.feature_len = 32*8*8 + self.num_traj_states*5
    else: self.feature_len = 2755
    if self.model_type == 'feed_forward':
      # self.fc1 = nn.Linear(self.feature_len, self.feature_len)# 500)
      # self.fc2 = nn.Linear(self.feature_len, int(self.feature_len/2))
      # self.fc3 = nn.Linear(int(self.feature_len/2), self.out_dim) #500
      # #self.fc1 = nn.Linear(self.feature_len, self.feature_len)# 500)
      #self.fc2 = nn.Linear(self.feature_len, self.out_dim)
      #self.fc3 = nn.Linear(int(self.feature_len/2), self.out_dim) #500
      self.classifier = nn.Sequential(
        nn.Dropout(p = 0.5),
        nn.Linear(32 * self.im_size/16 * self.im_size/16 + self.num_traj_states * (self.state_dim)/2, 1000),
        nn.BatchNorm1d(1000),
        nn.ReLU(inplace=True),
        nn.Dropout(p = 0.5),
        nn.Linear(1000, 640),
        nn.BatchNorm1d(640),
        nn.ReLU(inplace=True),
        nn.Dropout(p = 0.5),
        nn.Linear(640, self.out_dim),
      )
      for m in self.classifier.children():
        if isinstance(m, nn.Linear):
          nn.init.xavier_uniform(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()


    elif self.model_type == 'rnn_gru' or self.model_type == 'rnn_lstm':
      rnn_in = self.feature_len
      self.hidden_dim = learn_params['model']['hidden_dim']
      self.num_hidden = learn_params['model']['num_hidden']      
      if  self.model_type == "rnn_gru":
        self.rnn = nn.GRU(rnn_in, self.hidden_dim, self.num_hidden)
      elif self.model_type == "rnn_lstm":
        self.rnn = nn.LSTM(rnn_in, self.hidden_dim, self.num_hidden)
      self.fc = nn.Linear(self.hidden_dim, self.out_dim)


  def forward(self, th, features, hidden=None):
    batch_size = th.shape[0]
    thx  = torch.index_select(th, -1, torch.tensor(0))
    thy  = torch.index_select(th, -1, torch.tensor(1))
    th_pos = torch.cat((thx, thy), dim=-1)
    # sdf = sdf - self.robot_radius
    # c_obs, J_obs = bilinear_interpolate(x[:,1], th_pos, self.res, self.env_params['x_lims'], self.env_params['y_lims'], self.use_cuda)
    
    # th_vel = 
    # if self.costmap_predict:
    #   costmap = costmap_2d(im, self.safety_dist)
    #   c_obs0, J_obs0 = bilinear_interpolate(costmap, th_pos, self.res, self.env_params['x_lims'], self.env_params['y_lims'], self.use_cuda)
    #   costmap1 = self.conv1(costmap)
    #   c_obs1, J_obs1 = bilinear_interpolate(costmap1, th_pos, self.res, self.env_params['x_lims'], self.env_params['y_lims'], self.use_cuda)
    #   costmap2 = self.conv2(costmap1)
    #   c_obs2, J_obs2 = bilinear_interpolate(costmap2, th_pos, self.res, self.env_params['x_lims'], self.env_params['y_lims'], self.use_cuda)

    # else:
    #   sdf_shifted = im - self.robot_radius
    #   # print(sdf_shifted)
    #   # raw_input('..')
    #   c_obs0, J_obs0 = bilinear_interpolate(sdf_shifted, th_pos, self.res, self.env_params['x_lims'], self.env_params['y_lims'], self.use_cuda)
    #   im1 = F.leaky_relu(self.conv1(sdf_shifted))
    #   #c_obs1, J_obs1 = bilinear_interpolate(im1, th_pos, self.res, self.env_params['x_lims'], self.env_params['y_lims'], self.use_cuda)
    #   im2 = F.leaky_relu(self.conv2(im1))
    #   #c_obs2, J_obs2 = bilinear_interpolate(im2, th_pos, self.res, self.env_params['x_lims'], self.env_params['y_lims'], self.use_cuda)
    #   im3 = F.leaky_relu(self.conv3(im2))
    #   c_obs3, _ = bilinear_interpolate(im3, th_pos, self.res, self.env_params['x_lims'], self.env_params['y_lims'], self.use_cuda)


    # if not self.use_dtheta:

    # conc = torch.cat((th, c_obs, J_obs), -1)#,J_obs1, J_obs2  
    # else:
      # conc = torch.cat((th, c_obs, J_obs), -1)
      # dthx = torch.index_select(th, -1, torch.tensor(2))
      # dthy = torch.index_select(th, -1, torch.tensor(3))
      # dth_pos = torch.cat((thx, thy), dim=-1)
      # conc = torch.cat((th_pos, dth_pos, J_obs0, c_obs0, J_obs1, c_obs1), -1)
    #Flatten the trajectory features

    # conc_flat = conc.view(-1, conc.shape[1]*conc.shape[2])
    #Append the conv output

    # if features is not None:
    conc = torch.cat((features,th_pos.view(th_pos.size(0),-1)),dim=1)
    # conc_flat = torch.cat((conc_flat, features), dim=-1)
    if self.model_type == 'feed_forward':
      # conc_flat = self.dropout(F.relu(self.fc1(conc_flat)))
      # conc_flat = self.dropout(F.relu(self.fc2(conc_flat)))
      # conc_flat = self.fc3(conc_flat)
      #conc_flat = F.leaky_relu(self.fc1(conc_flat))
      #conc_flat = self.fc2(conc_flat)
      out = self.classifier(conc)
      return out.view(batch_size, 1, -1)
    elif self.model_type == 'rnn_gru' or self.model_type == 'rnn_lstm':
      rnn_out, new_hidden = self.rnn(conc_flat.view(1, batch_size, -1),  (hidden[0].view(self.num_hidden, batch_size, self.hidden_dim),
                                                                    hidden[1].view(self.num_hidden, batch_size, self.hidden_dim)))
      conc_flat = self.fc(rnn_out)

      return conc_flat.view(batch_size, 1, -1), (new_hidden[0].view(batch_size, self.num_hidden, self.hidden_dim), 
                                            new_hidden[1].view(batch_size, self.num_hidden, self.hidden_dim))


  def normalize_im(self, im):
    max_vec = (torch.max(torch.max(im, dim=2)[0], dim=-1)[0]).view(im.shape[0], im.shape[1], 1, 1)
    min_vec = (torch.min(torch.min(im, dim=2)[0], dim=-1)[0]).view(im.shape[0], im.shape[1], 1, 1)
    im_norm = 2.0 * (torch.div(im - min_vec, max_vec - min_vec + 1e-6) - 0.5)
    return im_norm

  def normalize_th(self, th):
    max_vec = (torch.max(th, dim=1)[0]).view(th.shape[0],1,th.shape[-1])
    min_vec = (torch.min(th, dim=1)[0]).view(th.shape[0],1,th.shape[-1])
    th_norm = 2.0 * (torch.div(th - min_vec, max_vec - min_vec + 1e-6) - 0.5)
    return th_norm

  
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

  def init_hidden(self, batch_size):
    if self.model_type == "rnn_lstm":
      return (Variable(torch.zeros(self.num_hidden, batch_size, self.hidden_dim)),
              Variable(torch.zeros(self.num_hidden, batch_size, self.hidden_dim)))
    return Variable(torch.zeros(self.num_hidden, batch_size, self.hidden_dim))        
