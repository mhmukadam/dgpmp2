import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, "..")
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from datasets import PlanningDataset
from diff_gpmp2.env.env_2d import Env2D


plt.ion()
dataset = PlanningDataset("../datasets/dataset_files/dataset_2d_8/", mode='train', label_subdir='opt_trajs_gpmp2')
print len(dataset)

##Iterate through the dataset and visualize
for i in xrange(len(dataset)):
  print dataset[i] 
  im = dataset[i]['im'][0,:,:]
  sdf = dataset[i]['sdf'][:,:]
  start = dataset[i]['start']
  goal = dataset[i]['goal']
  th_opt = dataset[i]['th_opt']    
  env_params = dataset.meta_data['env_params']
  env = Env2D(env_params)
  env.initialize_from_image(im, sdf)
  path_f = []
  for j in xrange(th_opt.shape[0]):
    path_f.append(th_opt[j,0:2])
  print start[0][0:2], goal[0][0:2]
  env.initialize_plot(start[0][0:2], goal[0][0:2])
  env.plot_edge(path_f)
  # env.plot_signed_distance_transform()
  plt.show()
  raw_input('Curr datapoint = {}. Press enter to view next data point'.format(i))
  env.close_plot()

# def visualize_batch(sample_batch):
#   im_batch, sdf_batch, start_batch, goal_batch, th_opt_batch = \
#     sample_batch['im'], sample_batch['sdf'], sample_batch['start'], sample_batch['goal'], sample_batch['th_opt'] 
#   batch_size = len(im_batch)
#   im_size = im_batch.size(2)
#   sdf_batch = sdf_batch.unsqueeze(1)
#   grid_im = utils.make_grid(im_batch)
#   grid_sdf = utils.make_grid(sdf_batch)

#   plt.figure()
#   plt.imshow(grid_im.numpy().transpose((1,2,0)))
#   plt.figure()
#   plt.imshow(grid_sdf.numpy().transpose((1,2,0)))
  

# #Let's test batching 
# dataloader = DataLoader(dataset, batch_size=4,
#                         shuffle=True, num_workers=4)

# for i_batch, sample_batched in enumerate(dataloader):
#   print(i_batch, sample_batched['im'].size(),
#         sample_batched['sdf'].size(), sample_batched['start'].size(),
#         sample_batched['goal'].size(), sample_batched['th_opt'].size())

#   if i_batch == 3:
#     visualize_batch(sample_batched)
#     plt.axis('off')
#     plt.ioff()
#     plt.show()
#     break
