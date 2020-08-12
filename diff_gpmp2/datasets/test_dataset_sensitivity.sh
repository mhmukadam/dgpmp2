#!/usr/bin/env bash
data_folder=('dataset_files/dataset_2d_14')
plan_param_files='gpmp2_params'
robot_param_files='robot'
env_param_file='env_params'
num_envs=100
obs_lambda=1.0
seed_val=123


python2.7 test_dataset_sensitivity.py --dataset_folder ${data_folder[@]}  --plan_param_file ${plan_param_files} \
                                      --robot_param_file ${robot_param_files} --env_param_file ${env_param_file} \
                                      --seed_val ${seed_val} --num_envs ${num_envs} --obs_lambda ${obs_lambda} \
                                      #--render #--step
          
