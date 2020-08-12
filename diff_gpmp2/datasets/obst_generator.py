import sys, os
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import seed as randseed 
from random import randint
from math import ceil
# from abc import ABC, abstractmethod

class Obstacle(object):
    """
    Base 2D Obstacle class
    """

    def __init__(self, center_x, center_y):
      self.center_x = center_x
      self.center_y = center_y

    # @abstractmethod
    def _obstacle_collision_check(self):
      pass

    # @abstractmethod
    def _point_collision_check(self):
      pass

    # @abstractmethod
    def _add_to_map(self):
      pass

class ObstacleRectangle(Obstacle):
    """
    Derived 2D rectangular Obstacle class
    """

    def __init__(self,center_x=0,center_y=0,width=None,height=None):
      super(ObstacleRectangle, self).__init__(center_x, center_y)
      self.center_x = center_x
      self.center_y = center_y
      self.width = width 
      self.height = height

    def _obstacle_collision_check(self, obst_map, patch_size=0):
      valid=True
      obst_map_test = self._add_to_map(np.copy(obst_map), patch_size)
      if (np.any( obst_map_test > 1)):
        valid=False
      return valid

    def _point_collision_check(self, obst_map, pts, patch_size):
      valid=True
      if pts is not None:
          obst_map_test = self._add_to_map(np.copy(obst_map)) 
          for pt in pts:
            obst_map_test_pt = self._add_point_to_map(pt, np.copy(obst_map_test), patch_size)
            if (np.any(obst_map_test_pt > 1 )):
              valid=False
              break
            # if (obst_map_test[ int(ceil(pt[1])), int(ceil(pt[0]))] == 1):
            #     valid=False
            #     break
      return valid
    
    def _add_point_to_map(self, pt, obst_map, patch_size=0):
      obst_map[int(ceil(pt[1]) - ceil(patch_size/2)) : int(ceil(pt[1]) + ceil(patch_size/2)) , 
               int(ceil(pt[0]) - ceil(patch_size/2)) : int(ceil(pt[0]) + ceil(patch_size/2))] += 1
      return obst_map


    def _add_to_map(self,obst_map, patch_size=0):
      # obst_map[ int(self.center_x - ceil(self.width/2) ) : int(self.center_x + ceil(self.width/2)),
      #           int(self.center_y - ceil(self.height/2)) : int(self.center_y + ceil(self.height/2))] += 1
      obst_map[ int(self.center_y - ceil(self.height/2) - ceil(patch_size/2)) : int(self.center_y + ceil(self.height/2) + ceil(patch_size/2) ),
                int(self.center_x - ceil(self.width/2) - ceil(patch_size/2)) : int(self.center_x + ceil(self.width/2) + ceil(patch_size/2) )] += 1
      return obst_map

class ObstacleWall(Obstacle):
  """
  Derived 2D Wall Obstacle with gap class
  """
  def __init__(self, center_x=0, width=None, gap_y=0, gap_width=None):
    super(ObstacleWall, self).__init__(center_x, 0)
    self.gap_y = gap_y
    self.width = width 
    self.gap_width = gap_width

  def _obstacle_collision_check(self, obst_map):
    valid=True
    obst_map_test = self._add_to_map(np.copy(obst_map))
    if (np.any( obst_map_test > 1)):
        valid=False
    return valid

  def _point_collision_check(self,obst_map,pts, patch_size):
    valid=True
    if pts is not None:
        obst_map_test = self._add_to_map(np.copy(obst_map)) 
        for pt in pts:
          obst_map_test_pt = self._add_point_to_map(pt, np.copy(obst_map_test), patch_size)
          if (np.any(obst_map_test_pt > 1 )):
            valid=False
            break
          # if (obst_map_test[ int(ceil(pt[1])), int(ceil(pt[0]))] == 1):
          #     valid=False
          #     break
    return valid

  def _add_point_to_map(self, pt, obst_map, patch_size):
    obst_map[int(ceil(pt[1]) - ceil(patch_size/2)) : int(ceil(pt[1]) + ceil(patch_size/2)) , 
             int(ceil(pt[0]) - ceil(patch_size/2)) : int(ceil(pt[0]) + ceil(patch_size/2))] += 1
    return obst_map

  def _add_to_map(self,obst_map):
    # obst_map[ self.center_x - ceil(self.width/2)  : self.center_x + ceil(self.width/2),
    #           0: self.gap_y - ceil(self.gap_width/2)] += 1
    # obst_map[ self.center_x - ceil(self.width/2)  : self.center_x + ceil(self.width/2),
    #           self.gap_y + ceil(self.gap_width/2) :] += 1
    obst_map[0: int(self.gap_y) - int(ceil(self.gap_width/2)),
             int(self.center_x) - int(ceil(self.width/2))  : int(self.center_x) + int(ceil(self.width/2))] += 1
    
    obst_map[int(self.gap_y) + int(ceil(self.gap_width/2)) :, 
             int(self.center_x) - int(ceil(self.width/2))  : int(self.center_x) + int(ceil(self.width/2))] += 1

    return obst_map



def random_rect(map_dim,w_min=2,w_max=10,h_min=2,h_max=10,start_x=0,start_y=0, end_x=0, end_y=0):
  """
  Generates an rectangular obstacle object, with random location and dimensions.
  """

  w = randint(w_min,w_max)
  h = randint(h_min,h_max)
  cx = randint(start_x+ceil(w/2),end_x-ceil(w/2)) 
  cy = randint(start_y+ceil(h/2),end_y-ceil(h/2)) 
  return ObstacleRectangle(cx,cy,w,h)

def random_wall(map_dim,w_min=2,w_max=10,gw_min=2,gw_max=10,start_x=0,gap_y=0):
  w = randint(w_min, w_max)
  gw = randint(gw_min, gw_max)
  cx = randint(start_x+ceil(w/2),map_dim[0]-ceil(w/2))
  gy = randint(gap_y+ceil(gw/2),map_dim[1]-ceil(gw/2))
  return ObstacleWall(cx, w, gy, gw)

def save_map_image(obst_map=None,start_pts=None,goal_pts=None,dir='.',name='obst_map'):
  try:
    if not os.path.exists(dir):
      os.makedirs(dir)
    obs_map_t = obst_map
    # fig = plt.figure(frameon=False, dpi=1)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(obs_map_t,cmap='gray')
    # print obs_map_t.shape
    # print "img", name
    # if start_pts is not None:
    #   for pt in start_pts: obs_map_t[int(ceil(pt[1])),int(ceil(pt[0]))] = 0.0#plt.plot(pt[0],pt[1],'.g') #
    # if goal_pts is not None:
    #   for pt in goal_pts: obs_map_t[int(ceil(pt[1])),int(ceil(pt[0]))] = 0.0#plt.plot(pt[0],pt[1],'.r')#
    # ax.invert_yaxis()# plt.gca().invert_yaxis()
    # fig.subplots_adjust(bottom = 0)
    # fig.subplots_adjust(top = 1)
    # fig.subplots_adjust(right = 1)
    # fig.subplots_adjust(left = 0)    
    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig('{}/{}.png'.format(dir,name), bbox_inches='tight', transparent=True, pad_inches=0)#, obs_map_t, cmap=cm.gray)
    # plt.close(fig)
    plt.imsave('{}/{}.png'.format(dir,name), obs_map_t, cmap=cm.gray)    

  except Exception as err:
    print("Error: could not save map.")
    print(err)
  return

def generate_rect_obstacle_map(map_dim=(256,256),num_obst=0,start_pts=None,goal_pts=None, w_min=30, w_max=40, h_min=30, h_max=40, start_x=0, start_y=0, end_x=0, end_y=0, patch_size=0, patch_size_obs=0, seed=None):
    """
    Args
    ---
    map_dim : (int,int)
        2D tuple containing dimensions of obstacle/occupancy grid.
        Treat as [x,y] coordinates, with origin at bottom-left corner.
    num_obst : int
        Number of obstacles
    start_pts : float
        Array of x-y points along robot arm for Start configuration.
        Dim: [Num. of points, 2]
    goal_pts : float
        Array of x-y points along robot arm for Goal configuration.
        Dim: [Num. of points, 2]
    seed : int or None
    """

    # randseed(seed) 

    origin_x = map_dim[0]/2
    origin_y = map_dim[1]/2

    obst_map = np.zeros(map_dim) # White - empty space, Black - Obstacles

    while True:
      for i in range(num_obst):
        valid=False
        while not valid:
          rect = random_rect(map_dim,w_min,w_max,h_min,h_max,start_x,start_y, end_x, end_y)
          valid = rect._obstacle_collision_check(obst_map, patch_size_obs) & \
                  rect._point_collision_check(obst_map, start_pts, patch_size) & \
                  rect._point_collision_check(obst_map, goal_pts, patch_size)
        obst_map = rect._add_to_map(obst_map)
      if not np.any(obst_map>1):
          break

    obst_map = 1-obst_map # Invert values

    #save_map_image(obst_map)
    # save_map_image(obst_map,start_pts,goal_pts)

    return obst_map    




def generate_wall_obstacle_map(map_dim=(256,256),num_obst=0,start_pts=None,goal_pts=None, w_min=2, w_max=10, gw_min=2, gw_max=10, start_x=0, gap_y=0, patch_size=1, seed=None):
    """
    Args
    ---
    map_dim : (int,int)
        2D tuple containing dimensions of obstacle/occupancy grid.
        Treat as [x,y] coordinates, with origin at bottom-left corner.
    num_obst : int
        Number of obstacles
    start_pts : float
        Array of x-y points along robot arm for Start configuration.
        Dim: [Num. of points, 2]
    goal_pts : float
        Array of x-y points along robot arm for Goal configuration.
        Dim: [Num. of points, 2]
    seed : int or None
    """

    # randseed(seed) 

    origin_x = map_dim[0]/2
    origin_y = map_dim[1]/2

    obst_map = np.zeros(map_dim) # White - empty space, Black - Obstacles

    while True:
        for i in range(num_obst):
            valid=False
            while not valid:
                wall = random_wall(map_dim,w_min,w_max,gw_min,gw_max,start_x,gap_y)
                valid = wall._obstacle_collision_check(obst_map) & \
                        wall._point_collision_check(obst_map, start_pts, patch_size) & \
                        wall._point_collision_check(obst_map, goal_pts, patch_size)
            obst_map = wall._add_to_map(obst_map)
        if not np.any(obst_map>1):
            break

    obst_map = 1-obst_map # Invert values

    #save_map_image(obst_map)
    # save_map_image(obst_map,start_pts,goal_pts)

    return obst_map    