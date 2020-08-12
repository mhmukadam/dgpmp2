import numpy as np
import torch
from scipy import ndimage


def sdf_2d(image, padlen=1, res=1.0):
  """
  Returns signed distance transform for the input image. 
  Remember to convert it to actual metric values when using with planner by multiplying it with 
  environment resolution.
  """
  
  im = np.array(image > 0.75, dtype=np.float64)
  
  if padlen > 0: im = np.pad(im, (padlen, padlen), 'constant', constant_values=(1.0,1.0))
  inv_im = np.array(1.0 - im, dtype=np.float64)
  dist_func = ndimage.distance_transform_edt
  im_dist = dist_func(im)
  inv_im_dist = dist_func(inv_im)
  sedt = (im_dist - inv_im_dist)*res
  return sedt 

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def costmap_2d(sdf, eps):
  loss = -1.0 * sdf + eps
  hinge = sdf <= eps
  hinge = hinge.double()
  cost_map = hinge * loss
  return cost_map

def safe_sdf(sdf, eps):
  loss = -1.0 * sdf + eps
  return loss


def bilinear_interpolate(imb, stateb, res, x_lims, y_lims, use_cuda=False):
  """bilinear interpolation
  imb - [batch x r x c]
  stateb - [batch x num_traj_states x 2]
  """
  imb = imb.squeeze(1)

  if use_cuda: 
    dtype=torch.cuda.DoubleTensor
    dtype_long = torch.cuda.LongTensor
    device = torch.device('cuda')
  else: 
    dtype = torch.DoubleTensor
    dtype_long = torch.LongTensor
    device = torch.device('cpu')

  J = torch.zeros_like(stateb)
  MAX_D = (x_lims[1] - x_lims[0])

  orig_pix_x = (0. - x_lims[0]/res) #x coordinate of origin in pixel space
  orig_pix_y = (0. - y_lims[0]/res) #y coordinate of origin in pixel space
  orig_pix = torch.tensor([orig_pix_x, orig_pix_y], device=device) 

  px = (orig_pix[0] +  stateb[:,:,0]/res).contiguous().view(-1)
  py = (orig_pix[1] - stateb[:,:,1]/res).contiguous().view(-1)
  
  px1 = torch.floor(px).type(dtype_long)
  px2 = px1+1
  py1 = torch.floor(py).type(dtype_long)
  py2 = py1+1

  px1 = torch.clamp(px1, 0, imb.shape[-1]-1)
  px2 = torch.clamp(px2, 0, imb.shape[-1]-1)
  py1 = torch.clamp(py1, 0, imb.shape[1]-1)
  py2 = torch.clamp(py2, 0, imb.shape[1]-1)
  pz = torch.arange(imb.shape[0], device=device).repeat(stateb.shape[1],1)
  pz = pz.t().contiguous().view(-1).long()

  dx1y1 = imb[pz, py1, px1]
  dx2y1 = imb[pz, py1, px2]
  dx1y2 = imb[pz, py2, px1]
  dx2y2 = imb[pz, py2, px2]

  wa = (px2.type(dtype) - px) * (py2.type(dtype) - py)
  wb = (px - px1.type(dtype)) * (py2.type(dtype) - py)
  wc = (px2.type(dtype) - px) * (py - py1.type(dtype))
  wd = (px - px1.type(dtype)) * (py - py1.type(dtype))
  
  wja = (py2.type(dtype) - py)
  wjb = (py - py1.type(dtype))
  wjc = (px2.type(dtype) - px)
  wjd = (px - px1.type(dtype))
  d_obs = wa * dx1y1 + wb * dx2y1 + wc * dx1y2 + wd * dx2y2
  d_obs = d_obs.reshape(stateb.shape[0], stateb.shape[1], 1)

  J[:, :, 0] = (-1.0*(wja*(dx2y1 - dx1y1) + wjb*(dx2y2 - dx1y2))/res).view(stateb.shape[0], stateb.shape[1])
  J[:, :, 1] = ((wjc*(dx1y2 - dx1y1) + wjd*(dx2y2 - dx2y1))/res).view(stateb.shape[0], stateb.shape[1])

  inlimxu = stateb[:,:,0] <= x_lims[1]
  inlimxl = stateb[:,:,0] >= x_lims[0]
  inlimx = (inlimxu + inlimxl) == 1
  inlimyu = stateb[:,:,1] <= y_lims[1]
  inlimyl = stateb[:,:,1] >= y_lims[0]
  inlimy = (inlimyu + inlimyl) == 1
  inlimcond = (inlimx + inlimy) == 1
  inlimcond = inlimcond.reshape(stateb.shape[0], stateb.shape[1], 1)

  d_obs = torch.where(inlimcond, d_obs, torch.tensor(MAX_D, device=device))
  J     = torch.where(inlimcond, J, torch.zeros(1,2, device=device))

  return d_obs, J