from scipy import ndimage
import numpy as np

def sdf_2d(image, padlen=1, res=1.0):
  """
  Returns signed distance transform for the input image. 
  Remember to convert it to actual metric values when using with planner by multiplying it with 
  environment resolution.
  """
  
  im = np.array(image > 0.75, dtype=np.float64)
  im = np.pad(im, (padlen, padlen), 'constant', constant_values=(1,1))
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