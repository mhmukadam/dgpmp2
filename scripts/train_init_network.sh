#!/usr/bin/env bash
#Specify the command line parameters
dataset_folders=('../../datasets/dataset_files/dataset_2d_8' '../../datasets/dataset_files/dataset_2d_9')
param_base_folder='../../../diff_gpmp2_new_experiments/mixed_dataset'
in_folders=(${param_base_folder}'/experiment_9_init_network')
model_files=('epoch_352')
optimizer_files=('optimizer_epoch_23_bootstrap')
start_epochs=(0)
plan_param_files=('gpmp2_params')
robot_param_files=('robot')
env_param_file='env_params'
learn_param_file='learn_params'
run_idxs=(0) #Index of the experiments to run
seed_val=123

printf "Changing directories"
echo `pwd`
cd ../diff_gpmp2/learning
echo `pwd`

for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  printf "====Dataset %s Experiment id %s\n" "${dataset_name}" "${idx}===="
  python2.7 train_initializer.py --dataset_folders ${dataset_folders[@]} --in_folder ${in_folders[idx]} --plan_param_file ${plan_param_files[idx]} \
                                 --robot_param_file ${robot_param_files[idx]} --env_param_file ${env_param_file} --learn_param_file ${learn_param_file} \
                                 --seed_val ${seed_val} --use_cuda --model_file ${model_files[idx]} --test --render #--test_overfit
done
#
