import torch
from torch.nn.modules.loss import _Loss



class NormMSELoss(torch.nn.Module):
  def __init__(self, reduction='none', alpha=0.1, beta=1e-3):
    super(LearnModule, self).__init__()
    self.reduction = reduction
    self.alpha = alpha
    self.beta = beta

  def forward(self, input, target, weights):
    weights = self.alpha * weights + self.beta
    ret = (input - target) ** 2
    ret = torch.div(ret, weights)
    if reduction != 'none':
      ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


# class MSETraj(torch.nn.Module):
#   def __init__(self, reduction='none'):
#     super(LearnModule, self).__init__()
#     self.reduction = reduction

#   def forward(self, input, target, weights):
#     weights = self.alpha * weights + self.beta
#     ret = (input - target) ** 2
#     ret = torch.div(ret, weights)
#     if reduction != 'none':
#       ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
#     return ret

def mse_traj(input, target):
  res = (input-target) ** 2
  res = res.sum(dim=-1)
  res = res.mean()
  return res




def torch_optimizer(string, parameters, opt_params_dict):
  if string == "adam":
    return torch.optim.Adam(parameters, lr=opt_params_dict['alpha'], weight_decay=opt_params_dict['reg_strength'])
  elif string =="sgd":
    return torch.optim.SGD(parameters, lr=opt_params_dict['alpha'], momentum=opt_params_dict['momentum'],\
                           weight_decay=opt_params_dict['reg_strength'], nesterov=opt_params_dict['nesterov'])
  elif string == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=opt_params_dict['alpha'], momentum=opt_params_dict['momentum'],\
                               weight_decay=opt_params_dict['reg_strength'], centered=opt_params_dict['centered'])

def torch_loss(string, **kwargs):
  if string == "mse":
    return torch.nn.MSELoss(reduction=kwargs['reduction'])
  elif string == "norm_mse":
    return NormMSELoss(reduction=kwargs['reduction'], alpha=kwargs['alpha'], beta=kwargs['beta'])
  elif string =='mse_traj':
    return mse_traj
  elif string == 'huber':
    return torch.nn.SmoothL1Loss(reduction=kwargs['reduction'])


def torch_activation(string):
  if string == "relu":
    return torch.nn.functional.relu

def init_xavier_uniform(m, gain=1):
  if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
    torch.nn.init.xavier_uniform(m.weight, gain=gain)
    if m.bias is not None:
      m.bias.data.fill_(0.0)

def init_xavier_normal(m, gain=torch.nn.init.calculate_gain('relu')):
  if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
    torch.nn.init.xavier_normal(m.weight, gain=gain)
    if m.bias is not None:
      m.bias.data.fill_(0.0)



