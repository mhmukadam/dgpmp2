from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import torch
from torch.utils.data import Dataset
from .utils import *
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class PlanningDatasetMulti(Dataset):
  """Planning dataset."""

  def __init__(self, root_dirs, mode='train', num_envs=-1, num_env_probs=-1, label_subdir='opt_trajs_gpmp2'):
    """
    Args:
      root_dir (string): Dataset directories.
      mode (string): Load train or test data
      num_envs: Number of environments to load
      num_env_probs: Number of planning probs per environment
      label_subdir: Subdirectory with optimal paths.
    """
    print(root_dirs)

    self.num_datasets = len(root_dirs)
    self.root_dirs  = [os.path.abspath(root_dir) for root_dir in root_dirs]
    self.subdirs    = [os.path.join(root_dir, mode) for root_dir in self.root_dirs]
    self.imsdf_dirs = [os.path.join(subdir, "im_sdf") for subdir in self.subdirs]
    self.label_dirs = [os.path.join(subdir, label_subdir) for subdir in self.subdirs]
    meta_files = [os.path.join(subdir, 'meta.yaml') for subdir in self.subdirs]
    self.meta_data = []
    self.num_files = []
    print(num_envs)
    
    self.envs_per_dataset = int(num_envs/self.num_datasets)


    for meta_file in meta_files:
      with open(meta_file) as infile:
        curr_meta_data = yaml.load(infile)
      if self.envs_per_dataset > 0 and self.envs_per_dataset <= curr_meta_data['num_envs'] and num_env_probs > 0 and num_env_probs <= curr_meta_data['probs_per_env']:
        print('User specified %d environments and %d problems per environment'%(self.envs_per_dataset, num_env_probs))
        curr_meta_data['num_envs'] = self.envs_per_dataset
        curr_meta_data['probs_per_env'] = num_env_probs
      else:
        print('User did not specify amount of data or specified more data than available. Using %d envs with %d problems per env'%(curr_meta_data['num_envs'],
                                                      curr_meta_data['probs_per_env']))
      self.meta_data.append(curr_meta_data)    
   
    
    self.num_files = [meta_data['num_envs'] * meta_data['probs_per_env'] for meta_data in self.meta_data]

  def __len__(self):
    total_files = reduce((lambda x, y: x + y), self.num_files)
    return  total_files

  def __getitem__(self, idx):
    dataset_num, dataset_idx = self.get_dataset(idx)
    env_idx  = int(dataset_idx/self.meta_data[dataset_num]['probs_per_env'])
    prob_idx = int(dataset_idx%self.meta_data[dataset_num]['probs_per_env'])
    #Load im and sdf
    imfile  = os.path.join(self.imsdf_dirs[dataset_num], str(env_idx) + "_im.png")
    sdffile = os.path.join(self.imsdf_dirs[dataset_num], str(env_idx) + "_sdf.npy")
    im = plt.imread(imfile)
    sdf = np.load(sdffile)
    #Do some processing
    im = utils.rgb2gray(im)
    im = np.array([im > 0.75], dtype=np.float64)
    im = torch.tensor(im)
    sdf = torch.tensor([sdf])
    #Load start, goal and optimal path/trajectory
    curr_traj_file = os.path.join(self.label_dirs[dataset_num], "env_" + str(env_idx) +  "_prob_" + str(prob_idx) +".npz")
    npf = np.load(curr_traj_file)
    start = torch.tensor([npf['start']])
    goal = torch.tensor([npf['goal']])
    th_opt = torch.tensor(npf['th_opt'])
    #Bundle up sample and return
    sample = {'im': im, 'sdf': sdf, 'start': start, 'goal': goal, 'th_opt': th_opt}

    return sample

  def get_dataset(self, idx):
    dataset_num = 0
    dataset_idx = 0
    offset = 0
    for i,n in enumerate(self.num_files):
      if idx >= offset + n:
        offset = offset + n
      else:
        dataset_num = i
        dataset_idx = idx - offset
        break
    return dataset_num, dataset_idx