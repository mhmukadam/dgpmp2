#!/usr/bin/env bash
out_folder='dataset_files/dataset_2d_7'
dataset_type='image'
im_folders=('../../lsp/graph_collision_checking_dataset/dataset_2d_4/environment_images')
im_size=128
num_train=800
num_test=200
seed_val=123

python2.7 generate_2d_im_dataset.py --out_folder ${out_folder} --dataset_type ${dataset_type} --im_size ${im_size} --num_train ${num_train}\
           --num_test ${num_test} --seed_val ${seed_val}  --im_folders ${im_folders} --render