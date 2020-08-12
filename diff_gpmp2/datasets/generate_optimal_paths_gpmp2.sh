#!/usr/bin/env bash
data_folder='dataset_files/dataset_2d_6'
num_train_envs=800
num_test_envs=200
im_size=128
probs_per_env=6
seed_val=123

python2.7 generate_optimal_paths_gpmp2.py --data_folder ${data_folder} --num_train_envs ${num_train_envs} --num_test_envs ${num_test_envs} \
          --im_size ${im_size} --probs_per_env ${probs_per_env} --seed_val ${seed_val} --train --rrt_star_init
