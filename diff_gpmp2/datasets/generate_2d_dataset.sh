#!/usr/bin/env bash
data_folder='dataset_files/dataset_2d_1'
dataset_type='tar_pit'
im_size=128
probs_per_env=1
num_train_envs=10
num_test_envs=1
seed_val=0


python generate_2d_dataset.py --out_folder ${data_folder} --dataset_type ${dataset_type} --num_train ${num_train_envs} --num_test ${num_test_envs} \
          --im_size ${im_size} --probs_per_env ${probs_per_env} --seed_val ${seed_val} #--rrtstar_init  #--render #--step
          
