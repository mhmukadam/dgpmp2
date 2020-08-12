#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import yaml

file = 'dataset_files/dataset_2d_8/sensitivity_results.yaml'

def plot_results(results_dict):
  sigmas = [float(k) for k in results_dict.keys()]
  sigmas = sorted(sigmas)
  num_unsolved_list = []
  for sigma in sigmas:
    in_coll = results_dict[str(sigma)]['in_collision']
    num_unsolved_list.append(np.mean(in_coll))

  plt.plot(sigmas, num_unsolved_list)
  plt.plot(sigmas, num_unsolved_list, 'ko')
  plt.show()

if __name__ == "__main__":
  with open(file, 'r') as fp:
    results_dict = yaml.load(fp)
  plot_results(results_dict)
