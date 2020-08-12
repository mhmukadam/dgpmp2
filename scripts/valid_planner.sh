#!/usr/bin/env bash
#Specify the command line parameters
dataset_folders=('../../datasets/dataset_files/dataset_2d_8' '../../datasets/dataset_files/dataset_2d_9')
experiment_base_folder='/home/mohak/workspace/research/diff_gpmp2_experiments/mixed_dataset'
valid_idxs_file='tarpit'
in_folder=${experiment_base_folder}'/experiment_1'
model_files=('epoch_0' 'epoch_1' 'epoch_2' 'epoch_3' 'epoch_4' 'epoch_5' 'epoch_6' 'epoch_7' 'epoch_8' 'epoch_9' 'epoch_10' 'epoch_11' 'epoch_12' 'epoch_13' 'epoch_14' 'epoch_15' 'epoch_16' 'epoch_17' 'epoch_18' 'epoch_19' 'epoch_20' 'epoch_21' 'epoch_22' 'epoch_23' 'epoch_24' 'epoch_25' 'epoch_26' 'epoch_27' 'epoch_28' 'epoch_29' 'epoch_30' 'epoch_31' 'epoch_32' 'epoch_33' 'epoch_34' 'epoch_35' 'epoch_36' 'epoch_37' 'epoch_38' 'epoch_39' 'epoch_40' 'epoch_41' 'epoch_42' 'epoch_43' 'epoch_44' 'epoch_45' 'epoch_46' 'epoch_47' 'epoch_48' 'epoch_49' 'epoch_50' 'epoch_51' 'epoch_52' 'epoch_53' 'epoch_54' 'epoch_55' 'epoch_56' 'epoch_57' 'epoch_58' 'epoch_59' 'epoch_60' 'epoch_61' 'epoch_62' 'epoch_63' 'epoch_64' 'epoch_65' 'epoch_66' 'epoch_67' 'epoch_68' 'epoch_69' 'epoch_70' 'epoch_71' 'epoch_72' 'epoch_73' 'epoch_74' 'epoch_75' 'epoch_76' 'epoch_77')
plan_param_files='gpmp2_params'
robot_param_files='robot'
env_param_file='env_params'
learn_param_file='learn_params'
run_idxs=(60 61 62 63) #Index of the experiments to run
seed_val=123

printf "Changing directories"
echo `pwd`
cd ../diff_gpmp2/learning
echo `pwd`



# #
for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  printf "====Experiment %s id %s\n" "${experiment_base_folder}" "${idx}===="
  python2.7 test_planner.py  --dataset_folder ${dataset_folders[@]} --in_folder ${in_folder} --plan_param_file ${plan_param_files} \
                             --robot_param_file ${robot_param_files} --env_param_file ${env_param_file} --learn_param_file ${learn_param_file} \
                             --model_file ${model_files[idx]}  --seed_val ${seed_val} --validation #--valid_idxs_file ${valid_idxs_file} \
			                       #--render
done

printf "Plotting results"
echo `pwd`
cd ../../examples
echo `pwd`


#python2.7 report_stats_example.py --in_folder ${in_folder}/'results/'
