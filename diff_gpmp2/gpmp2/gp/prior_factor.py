"""Prior factor to be applied to states with known covariances"""

import torch
from diff_gpmp2.utils import mat_utils

class PriorFactor(object):
  def __init__(self, ndims, sig, batch_size=1, use_cuda=False):
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu') 

    self.ndims = torch.tensor(ndims,device=self.device)
    self.cov = mat_utils.isotropic_matrix(torch.pow(sig,2.0),self.ndims, self.device)
    # self.inv_cov = mat_utils.isotropic_matrix(1.0/torch.pow(sig,2.0),self.ndims,self.device)#np.linalg.inv(cov)

  def get_error(self, stateb):
    error = self.meanb - stateb
    H = torch.eye(self.ndims, device=self.device).unsqueeze(0).repeat(stateb.shape[0], 1, 1)
    return error.view(stateb.shape[0], self.ndims, 1), H
  
  def get_cov(self):
    return self.cov
  
  def get_inv_cov(self):
    return self.inv_cov

  def set_mean(self, meanb):
    self.meanb = meanb
  def set_inv_cov(self, inv_covb):
    self.inv_cov = inv_covb

  # def eval_prob(self, state):
  #   error , _= self.get_error(state)
  #   f = np.exp(-1.0/2.0*np.transpose(error).dot(self.inv_cov).dot(error))
  #   norm = np.power(2.0*np.pi, self.ndims/2.0)*np.sqrt(np.linalg.det(self.inv_cov))
  #   f = f/norm
  #   return f