#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import sdf_2d


def one_obstacle_dataset(im_size):
  im =  np.ones((im_size, im_size), dtype=np.uint8)
  obs_size = int(0.3*im_size)
  
  obs_lb = int(0.2*im_size)
  obs_ub = int(0.8*im_size) - (obs_size + 1) 

  obs_x = np.random.randint(obs_lb, obs_ub)
  obs_y = np.random.randint(obs_lb, obs_ub)
  im[obs_y:obs_y+obs_size, obs_x:obs_x+obs_size] = 0
  return im

def one_obstacle_size_dataset(im_size):
  im =  np.ones((im_size, im_size), dtype=np.uint8)
  size_factor = np.random.uniform(0.15, 0.45)
  obs_size = int(size_factor*im_size)
  
  obs_lb = int(0.2*im_size)
  obs_ub = int(0.8*im_size)# - (obs_size + 1) 

  obs_cx = np.random.randint(obs_lb, obs_ub)
  obs_cy = np.random.randint(obs_lb, obs_ub)
  
  im[obs_cy-int(obs_size/2):obs_cy+int(obs_size/2), obs_cx-int(obs_size/2):obs_cx+int(obs_size/2)] = 0
  return im



def multi_obstacle_dataset(im_size):
  im =  np.ones((im_size, im_size), dtype=np.uint8)
  num_obstacles = np.random.randint(1, 4)
  for i in range(num_obstacles):
    if num_obstacles == 1:
      size_factor = 0.3
    else:
      size_factor = np.random.uniform(0.1, 0.3)
    obs_size = int(size_factor*im_size)
    obs_lb = int(0.1*im_size)
    obs_ub = int(0.9*im_size) - (obs_size + 1) 

    obs_x = np.random.randint(obs_lb, obs_ub)
    obs_y = np.random.randint(obs_lb, obs_ub)
    
    im[obs_y:obs_y+ obs_size, obs_x:obs_x+obs_size] = 0
  return im

def image_dataset(im_size, im_load):
  im = scipy.misc.imresize(im_load, (im_size, im_size))
  im = im.astype(np.uint8)
  return im
  
GENERATORS = { 'one_obstacle'     : one_obstacle_dataset,
               'one_obstacle_size': one_obstacle_size_dataset,
               'multi_obstacle'   : multi_obstacle_dataset,
               'image'            : image_dataset
}

def generate_dataset(out_folder, type, im_size, num_train, num_test, im_folder=None):
  train_folder = os.path.join(out_folder, "train/im_sdf")
  # valid_folder = os.path.join(out_folder, "valid/im_sdf")
  test_folder  = os.path.join(out_folder, "test/im_sdf")

  print(out_folder, train_folder, test_folder)
  if not os.path.exists(train_folder):
    os.makedirs(train_folder)
  if not os.path.exists(test_folder):
    os.makedirs(test_folder)

  for i in range(num_train):
    if type=='image':
      im_load = plt.imread(os.path.join(os.path.abspath(im_folder), "world_"+str(i+1)+".png"))
      im = GENERATORS[type](im_size, im_load)
    else: im = GENERATORS[type](im_size)
    sdf = sdf_2d(im)
    out_im  = os.path.join(train_folder, str(i) + "_im.png")
    out_sdf = os.path.join(train_folder, str(i) + "_sdf")
    print(out_im, out_sdf)
    plt.imsave(out_im, im, cmap=cm.gray)
    np.save(out_sdf, sdf)


  # for i in xrange(num_valid):
  #   im = GENERATORS[type](im_size)
  #   sdf = sdf_2d(im)
  #   out_im  = os.path.join(valid_folder, str(i) + "_im.png")
  #   out_sdf = os.path.join(valid_folder, str(i) + "_sdf")
  #   print out_im, out_sdf
  #   plt.imsave(out_im, im, cmap=cm.gray)
  #   np.save(out_sdf, sdf)

  for i in range(num_test):
    if type=='image':
      im_load = plt.imread(os.path.join(os.path.abspath(im_folder), "world_"+str(num_train+i+1)+".png"))
      im = GENERATORS[type](im_size, im_load)
    else: im = GENERATORS[type](im_size)
    sdf = sdf_2d(im)
    out_im  = os.path.join(test_folder, str(i) + "_im.png")
    out_sdf = os.path.join(test_folder, str(i) + "_sdf")
    print(out_im, out_sdf)
    plt.imsave(out_im, im, cmap=cm.gray)
    np.save(out_sdf, sdf)




if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--out_folder', type=str, required=True, default='.',  help="Relative path of output folder")
  parser.add_argument('--dataset_type', type=str, required=True, help='Type of dataset. See README.md for types available.')
  parser.add_argument('--im_size', type=int, required=True, default=128, help='Size of dataset images')
  parser.add_argument('--num_train', type=int, required=True, default=500, help='Number of training samples to generate.')
  parser.add_argument('--num_test', type=int, required=True, default=200, help='Number of training samples to generate.')
  parser.add_argument('--seed_val', type=int, default=0, help='Random seed for generating dataset')
  parser.add_argument('--im_folder', type=str, help='Only required if loading im data')
  args = parser.parse_args()
  np.random.seed(args.seed_val)
  generate_dataset(os.path.abspath(args.out_folder), args.dataset_type, args.im_size, args.num_train, args.num_test, args.im_folder)
