from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import torch
from torch.utils.data import Dataset
# from .utils import *
import diff_gpmp2.datasets.utils as utils
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class PlanningDataset(Dataset):
  """Planning dataset."""

  def __init__(self, root_dir, mode='train', num_envs=-1, num_env_probs=-1, label_subdir='opt_trajs_gpmp2'):
    """
    Args:
        root_dir (string): Directory with all the data.
        mode (string): Load train or test data
        num_envs: Number of environments to load
        num_env_probs: Number of planning probs per environment
        label_subdir: Subdirectory with optimal paths.
    """
    self.root_dir = os.path.abspath(root_dir)
    self.subdir = os.path.join(root_dir, mode)
    self.imsdf_dir = os.path.join(self.subdir, "im_sdf")
    self.label_dir = os.path.join(self.subdir, label_subdir)
    meta_file = os.path.join(self.subdir, 'meta.yaml')
    with open(meta_file) as infile:
      self.meta_data = yaml.load(infile)
    if num_envs > 0 and num_envs <= self.meta_data['num_envs'] and num_env_probs > 0 and num_env_probs <= self.meta_data['probs_per_env']:
      print('User specified %d environments and %d problems per environment'%(num_envs, num_env_probs))
      self.meta_data['num_envs'] = num_envs
      self.meta_data['probs_per_env'] = num_env_probs
    else:
      print('User did not specify amount of data or specified more data than available. \
             Using %d envs with %d problems per env'%(self.meta_data['num_envs'],
                                                      self.meta_data['probs_per_env']))
   
    self.num_files = self.meta_data['num_envs'] * self.meta_data['probs_per_env']

  def __len__(self):
    return self.num_files

  def __getitem__(self, idx):
    env_idx = int(idx/self.meta_data['probs_per_env'])
    prob_idx = int(idx%self.meta_data['probs_per_env'])
    #Load im and sdf
    imfile  = os.path.join(self.imsdf_dir, str(env_idx) + "_im.png")
    sdffile = os.path.join(self.imsdf_dir, str(env_idx) + "_sdf.npy")
    im = plt.imread(imfile)
    sdf = np.load(sdffile)
    #Do some processing
    im = utils.rgb2gray(im)
    im = np.array([im > 0.75], dtype=np.float64)
    im = torch.tensor(im)
    sdf = torch.tensor([sdf])
    #Load start, goal and optimal path/trajectory
    curr_traj_file = os.path.join(self.label_dir, "env_" + str(env_idx) +  "_prob_" + str(prob_idx) +".npz")
    npf = np.load(curr_traj_file)
    start = torch.tensor([npf['start']])
    goal = torch.tensor([npf['goal']])
    th_opt = torch.tensor(npf['th_opt'])
    #Bundle up sample and return
    sample = {'im': im, 'sdf': sdf, 'start': start, 'goal': goal, 'th_opt': th_opt}

    return sample