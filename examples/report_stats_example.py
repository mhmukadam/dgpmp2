#!/usr/bin/env python
import sys, os
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
import yaml
import glob

fields = ['coll_intensity', 'gp_mse', 'in_collision', 'max_penetration', 'num_iters', 'ext_cost_per_iter', 'task_loss_per_iter',
          'constraint_violation']
filetype = '.yaml'


def print_stats(all_results, best_fixed):
  global fields
  sorted_epochs = sorted(all_results.keys())
  for epoch in sorted_epochs:
    print('####################### Epoch = {} ##############'.format(epoch))
    data = all_results[epoch]
    for field in fields:
      if field not in data:
        continue
      d = data[field]
      if field == 'ext_cost_per_iter' or field == 'task_loss_per_iter':
        task_cost = []
        for iter_data in d:
          task_cost.append(iter_data[-1])
        print(('avg {} = {}').format(field, np.mean(task_cost)))
        print(('std {} = {}').format(field, np.std(task_cost)))
      else: 
        print(('avg {} = {}').format(field, np.mean(d)))
        print(('std {} = {}').format(field, np.std(d)))

  



def print_stats_succ(data, file_name):
  #Only print statistics for successful trials
  global fields
  print('####################### {} successful only ##############'.format(file_name))
  coll_results = np.array(data['in_collision'])
  succ_ids = np.where(coll_results==False)[0]
  for field in fields:
    if field not in data:      
      continue
    
    d = np.array(data[field])
    succ_data = d[succ_ids]

    if field == 'ext_cost_per_iter' or field == 'task_loss_per_iter':
      task_cost = []
      for iter_data in succ_data:
        task_cost.append(iter_data[-1])
      print(('avg {} = {}').format(field, np.mean(task_cost)))
      print(('std {} = {}').format(field, np.std(task_cost)))
    else:
      print(('avg {} = {}').format(field, np.mean(succ_data)))
      print(('std {} = {}').format(field, np.std(succ_data)))




def plot_stats(all_results, best_fixed):
  sorted_epochs = sorted(all_results.keys())
  num_unsolved = []
  num_iters = []
  gp_mse = []
  loss = []
  pos_loss = []
  vel_loss = []
  for epoch in sorted_epochs:
    data = all_results[epoch]
    num_unsolved.append(np.mean(data['in_collision']))
    num_iters.append(np.mean(data['num_iters']))
    gp_mse.append(np.mean(data['gp_mse']))
  
    mean_loss_env = []
    mean_pos_loss_env = []
    mean_vel_loss_env = []
    for d in data['loss_per_iter']:
      mean_loss_env.append(d[-1])
    loss.append(np.mean(mean_loss_env))
    for d in data['pos_loss_per_iter']:
      mean_pos_loss_env.append(d[-1])
    pos_loss.append(np.mean(mean_pos_loss_env))
    for d in data['vel_loss_per_iter']:
      mean_vel_loss_env.append(d[-1])
    vel_loss.append(np.mean(mean_vel_loss_env))

  num_unsolved_best = [np.mean(best_fixed['in_collision'])]*len(sorted_epochs)
  num_iters_best    = [np.mean(best_fixed['num_iters'])]*len(sorted_epochs)
  gp_mse_best       = [np.mean(best_fixed['gp_mse'])]*len(sorted_epochs)

  print "Best fixed covariance. In collision = {}, Avg. num iters = {}, GP_MSE = {}, Coll intensity = {}".format(np.mean(best_fixed['in_collision']), 
                                                                                            np.mean(best_fixed['num_iters']), 
                                                                                            np.mean(best_fixed['gp_mse']),
                                                                                            np.mean(best_fixed['coll_intensity']))
  fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6,ncols=1)
  ax1.plot(sorted_epochs, num_unsolved)
  ax1.plot(sorted_epochs, num_unsolved_best, 'r--')
  ax1.set_xlabel('epoch')
  ax1.set_ylabel('Mean unsolved')
  ax1.set_xlim(-1, sorted_epochs[-1])
  ax2.plot(sorted_epochs, gp_mse)
  ax2.plot(sorted_epochs, gp_mse_best, 'r--')
  ax2.set_xlabel('epoch')
  ax2.set_ylabel('Mean gp mse')
  ax2.set_xlim(-1, sorted_epochs[-1])
  ax3.plot(sorted_epochs, num_iters)
  ax3.plot(sorted_epochs, num_iters_best, 'r--')
  ax3.set_xlabel('epoch')
  ax3.set_ylabel('Mean iters')
  ax3.set_xlim(-1, sorted_epochs[-1])
  ax4.plot(sorted_epochs, loss)
  ax4.set_xlabel('epoch')
  ax4.set_ylabel('Mean regression loss')
  ax4.set_xlim(-1, sorted_epochs[-1])
  ax5.plot(sorted_epochs, pos_loss)
  ax5.set_xlabel('epoch')
  ax5.set_ylabel('Mean regression loss (position only)')
  ax5.set_xlim(-1, sorted_epochs[-1])
  ax6.plot(sorted_epochs, vel_loss)
  ax6.set_xlabel('epoch')
  ax6.set_ylabel('Mean regression loss (velocity only)')
  ax6.set_xlim(-1, sorted_epochs[-1])

  plt.show()

def main(args):
  in_folder = os.path.abspath(args.in_folder)
  if args.env_type is None:
    files = glob.glob(in_folder + "/epoch_*_valid_results.yaml")
    epoch_id = -3
    init_file = os.path.join(in_folder, 'init_planner_valid_results.yaml')
    print(files)
  else: 
    files = glob.glob(in_folder + "/epoch_*_valid_results_"+args.env_type+".yaml")
    init_file = os.path.join(in_folder, 'init_planner_valid_results' + '.yaml')
    epoch_id = -4
  
  all_results = {}
  
  with open(init_file, 'r') as fp:
    data = yaml.load(fp)
    all_results[-1] = data

  for file in files:
    epoch_num = int(file.split('_')[epoch_id])
    with open(file, 'r') as fp:
      data = yaml.load(fp)
      all_results[epoch_num] = data
  
  best_file = os.path.abspath(os.path.join(in_folder, args.best_cov_file+".yaml"))
  with open(best_file, 'r') as fp:
    best_fixed = yaml.load(fp)

  print_stats(all_results, best_fixed)
  plot_stats(all_results, best_fixed)
  plt.show()


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='PyTorch Example')
  parser.add_argument('--in_folder', type=str, required=True, help='folder with results file')
  parser.add_argument('--best_cov_file', type=str)
  parser.add_argument('--env_type', type=str)
  args = parser.parse_args()
  main(args)
