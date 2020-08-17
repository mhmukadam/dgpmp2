#!/usr/bin/env python
""" @package environment_interface
Loads an environment file from a database and returns a 2D
occupancy grid.

Inputs : file_name, x y resolution (meters to pixel conversion)
Outputs:  - 2d occupancy grid of the environment
          - ability to check states in collision
"""
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
from scipy import ndimage
from diff_gpmp2.utils import helpers, sdf_utils

class Env2D():
  def __init__(self, params, use_cuda=False):
    self.plot_initialized = False
    self.image = None
    self.sedt_available = False
    self.ndims = 2
    # self.pad_len = params['padlen']
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
    self.x_lims = params['x_lims']
    self.y_lims = params['y_lims']
    # self.pad_fn = torch.nn.ReplicationPad2d(self.pad_len)
    self.sedt_plot = False
    self.costmap_plot=  False

  def initialize_from_file(self, envfile):
    try:
      self.image = plt.imread(envfile)
      if len(self.image.shape) > 2:
        self.image = helpers.rgb2gray(self.image)
    except IOError:
      print("File doesn't exist. Please use correct naming convention for database eg. 0.png, 1.png .. and so on. You gave, %s"%(envfile))
    self.res = (self.x_lims[1] - self.x_lims[0])/((self.image.shape[1])*1.)

    orig_pix_x = (0 - self.x_lims[0]/self.res) #x coordinate of origin in pixel space
    orig_pix_y = (0 - self.y_lims[0]/self.res) #y coordinate of origin in pixel space
    self.orig_pix = torch.tensor([orig_pix_x, orig_pix_y], device=self.device)
    self.calculate_signed_distance_transform()
    self.MAX_D = (self.x_lims[1] - self.x_lims[0])
    self.sedt_available = True
  
  def initialize_from_image(self, img, sedt=None):
    self.image = img
    if len(self.image.shape) > 2:
      self.image = helpers.rgb2gray(self.image)
    self.res = (self.x_lims[1] - self.x_lims[0])/((self.image.shape[1])*1.)
    self.sedt = sedt
    if type(self.sedt).__module__ == np.__name__:
      self.sedt = torch.tensor(self.sedt, device=self.device)
    self.MAX_D = (self.x_lims[1] - self.x_lims[0])
    self.sedt_available = True

    orig_pix_x = (0 - self.x_lims[0]/self.res) #x coordinate of origin in pixel space
    orig_pix_y = (0 - self.y_lims[0]/self.res) #y coordinate of origin in pixel space
    self.orig_pix = torch.tensor([orig_pix_x, orig_pix_y], device=self.device)

  def in_limits(self, state):
    """Filters a state to lie between the environment limits

    @param state - input state
    @return 1 - in limits
          0 - not in limits
    """
    if self.x_lims[0] <= state[0] < self.x_lims[1] and self.y_lims[0] <= state[1] < self.y_lims[1]:
      return True
    return False


  def to_image_coordinates(self, state):
    """Helper function that returns pixel coordinates for a state in
    continuous coordinates

    @param  - state in continuous world coordinates
    @return - state in pixel coordinates """
    pix_x = self.orig_pix[0] + state[0]/self.res
    pix_y = self.orig_pix[1] - state[1]/self.res
    return (pix_x,pix_y)
  
  def is_feasible(self, state, eps):
    d, _ = self.get_signed_obstacle_distance(state.reshape(1,1,state.shape[0]))
    #print d, eps
    result = d > eps
    return  result.item()

  def to_world_coordinates(self, pix):
    """Helper function that returns world coordinates for a pixel

    @param  - state in continuous world coordinates
    @return - state in pixel coordinates """
    world_x = (pix[0] - self.orig_pix[0])*self.res 
    world_y = (self.orig_pix[1] - pix[1])*self.res   
    return (world_x, world_y)

  def get_env_lims(self):
    return self.x_lims, self.y_lims
  

  def calculate_signed_distance_transform(self, pad_len=1):
    if not self.sedt_available:
      im = np.array(self.image > 0.75, dtype=np.float64)
      # im = np.pad(im, (padlen, padlen), 'constant', constant_values=(1,1))
      inv_im = np.array(1.0 - im, dtype=np.float64)
      dist_func = ndimage.distance_transform_edt
      im_dist = dist_func(im)
      inv_im_dist = dist_func(inv_im)
      self.sedt = (im_dist - inv_im_dist)*self.res 
      self.sedt = torch.tensor(self.sedt, device=self.device)
      self.sedt_available = True
      print('Calculated Signed Distance Transform')


  def get_signed_obstacle_distance(self, stateb):
    #Return signed distance for a point via bilinear interpolation
    if self.use_cuda: 
      dtype=torch.cuda.DoubleTensor
      dtype_long = torch.cuda.LongTensor
    else: 
      dtype = torch.DoubleTensor
      dtype_long = torch.LongTensor

    J = torch.zeros_like(stateb)


    px = self.orig_pix[0] +  stateb[:,:,0]/self.res
    py = self.orig_pix[1] - stateb[:,:,1]/self.res
    px1 = torch.floor(px).type(dtype_long)
    px2 = px1+1
    py1 = torch.floor(py).type(dtype_long)
    py2 = py1+1

    px1 = torch.clamp(px1, 0, self.sedt.shape[1]-1)
    px2 = torch.clamp(px2, 0, self.sedt.shape[1]-1)
    py1 = torch.clamp(py1, 0, self.sedt.shape[0]-1)
    py2 = torch.clamp(py2, 0, self.sedt.shape[0]-1)
    
    dx1y1 = self.sedt[py1, px1]
    dx2y1 = self.sedt[py1, px2]
    dx1y2 = self.sedt[py2, px1]
    dx2y2 = self.sedt[py2, px2]

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

    J[:, :, 0] = -1.0*(wja*(dx2y1 - dx1y1) + wjb*(dx2y2 - dx1y2))/self.res
    J[:, :, 1] = (wjc*(dx1y2 - dx1y1) + wjd*(dx2y2 - dx2y1))/self.res

    inlimxu = stateb[:,:,0] <= self.x_lims[1]
    inlimxl = stateb[:,:,0] >= self.x_lims[0]
    inlimx = (inlimxu + inlimxl) > 1
    inlimyu = stateb[:,:,1] <= self.y_lims[1]
    inlimyl = stateb[:,:,1] >= self.y_lims[0]
    inlimy = (inlimyu + inlimyl) > 1
    inlimcond = (inlimx + inlimy) > 1
    inlimcond = inlimcond.reshape(stateb.shape[0], stateb.shape[1], 1)

    d_obs = torch.where(inlimcond, d_obs, torch.tensor(self.MAX_D, device=self.device))
    J     = torch.where(inlimcond, J, torch.zeros(1,self.ndims, device=self.device))

    return d_obs, J


  def get_signed_obstacle_distance_vec(self, state_vec):
    assert len(state_vec.shape) ==2 and state_vec.shape[1] == self.ndims, "State vector must 2D tensor with a different 2D state on each row"    
    d_obs_vec = torch.zeros((len(state_vec), 1), device=self.device)
    J = torch.zeros((len(state_vec), len(state_vec) * self.ndims), device=self.device)
    for i, state in enumerate(state_vec):
      d_obs, J_i = self.get_signed_obstacle_distance(state)
      d_obs_vec[i] = d_obs
      J[i, self.ndims*i: self.ndims*(i+1)] = J_i
    return d_obs_vec, J



  def initialize_plot(self, start, goal, grid_res=None, plot_grid=False):
    self.figure, self.axes = plt.subplots()
    self.figure.patch.set_facecolor('white')
    self.axes.set_xlim([self.x_lims[0]-1.8, self.x_lims[1]+1.8])
    self.axes.set_ylim([self.y_lims[0]-1.8, self.y_lims[1]+1.8])
    if plot_grid and grid_res:
      self.axes.set_xticks(np.arange(self.x_lims[0], self.x_lims[1], grid_res[0]))
      self.axes.set_yticks(np.arange(self.y_lims[0], self.y_lims[1], grid_res[1]))
      self.axes.grid(which='both')
    self.figure.show()
    self.visualize_environment()
    self.line, = self.axes.plot([],[])
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox) 
    self.plot_state(start, 'cyan')
    self.plot_state(goal, 'green')
    self.figure.canvas.draw()
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox) 
    self.plot_initialized = True

  def plot_signed_distance_transform(self):
    if self.sedt_available:
      self.fig2, self.ax2 = plt.subplots()

      im = self.ax2.imshow(self.sedt.cpu(), extent = (self.x_lims[0], self.x_lims[1], self.y_lims[0], self.x_lims[1]), cmap = 'hsv')
      self.fig2.colorbar(im)
      self.sedt_plot = True
      plt.axis('off')
  
  def plot_costmap(self, eps):
    if self.sedt_available:
      self.fig3, self.ax3 = plt.subplots()
      cost_map = sdf_utils.costmap_2d(self.sedt, eps)
      im = self.ax3.imshow(cost_map.cpu(), extent = (self.x_lims[0], self.x_lims[1], self.y_lims[0], self.x_lims[1]), cmap = 'hsv')
      self.fig3.colorbar(im)
      self.costmap_plot = True


  def reset_plot(self, start, goal, grid_res=None):
    if self.plot_initialized:
      plt.close(self.figure) 
      self.initialize_plot(start, goal, grid_res)
    if self.sedt_plot:
      plt.close(self.fig2)
      self.plot_signed_distance_transform()

  def visualize_environment(self):
    self.axes.imshow(self.image, extent = (self.x_lims[0], self.x_lims[1], self.y_lims[0], self.x_lims[1]), cmap='gray')
    # plt.axis('off')

  def plot_edge(self, edge, linestyle='solid', color='blue', linewidth=2, alpha=1.0, markerstyle='o', markersize=4.0, label=None):
    x_list = []
    y_list = []
    for s in edge:
      x_list.append(s[0])
      y_list.append(s[1])
    self.figure.canvas.restore_region(self.background)
    line = plt.Line2D(x_list, y_list)
    # self.line.set_xdata(x_list)
    # self.line.set_ydata(y_list)
    line.set_linestyle(linestyle)
    line.set_linewidth(linewidth)
    line.set_color(color)
    line.set_alpha(alpha)
    line.set_label(label)
    # line.set_marker(markerstyle)
    # line.set_markersize(markersize)
    self.axes.add_line(line)
    self.axes.legend()
    # if len(self.axes.lines) > 12:
    #   self.axes.lines.pop(0)
    # self.axes.draw_artist(line)
    self.figure.canvas.blit(self.axes.bbox)
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox) 

  def clear_edges(self):
    # x_list = []
    # y_list = []
    # for s in edge:
    #   x_list.append(s[0])
    #   y_list.append(s[1])
    # self.figure.canvas.restore_region(self.background)
    # line = plt.Line2D(x_list, y_list)
    # # self.line.set_xdata(x_list)
    # # self.line.set_ydata(y_list)
    # line.set_linestyle(linestyle)
    # line.set_linewidth(linewidth)
    # line.set_color(color)
    # line.set_alpha(alpha)
    # # line.set_marker(markerstyle)
    # # line.set_markersize(markersize)
    # self.axes.add_line(line)
    # while len(self.axes.lines) > 1:
    self.axes.lines[-1].remove()


    # self.figure.canvas.blit(self.axes.bbox)
    # self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox) 


  def plot_edges(self, edges,linestyle='solid', color='blue', linewidth=2):
    """Helper function that simply calls plot_edge for each edge"""
    for edge in edges:
      self.plot_edge(edge, linestyle, color, linewidth)

  def plot_state(self, state, color = 'red'):
    """Plot a single state on the environment"""
    # self.figure.canvas.restore_region(self.background)
    self.axes.plot(state[0], state[1], marker='o', markersize=10, color = color)
    self.figure.canvas.blit(self.axes.bbox)
    self.figure.canvas.draw()
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)
  
  def plot_path(self, path, linestyle='solid', color='blue', linewidth=2):
    flat_path = [item for sublist in path for item in sublist]
    self.plot_edge(flat_path, linestyle, color, linewidth)

  def close_plot(self):
    if self.plot_initialized:
      plt.close(self.figure)
      self.plot_initialized = False
    if self.sedt_plot:
      plt.close(self.fig2)
      self.sedt_plot = False
    if self.costmap_plot:
      plt.close(self.fig3)
      self.costmap_plot = False

  def clear(self):
    if self.plot_initialized:
      plt.close(self.figure)
      self.plot_initialized = False
    if self.sedt_available:
      self.s_edt = None
    if self.sedt_plot:
      plt.close(self.fig2)
      self.sedt_plot = False
    self.image = None

  