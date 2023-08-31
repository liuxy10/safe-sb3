#!/bin/bash
# 1. cd to the folder, run ./train_js_iql_in_chunks.sh
total_timesteps=1000000
num_chunks=50
step_per_chunk = total_timesteps/num_chunks
last_timestep = 0

for (( i=0; i<$num_chunks; i++ )); do
    echo "------------------------Iteration: $i ------------------------"
    if [ $i -eq 0 ]; then
        # python run_js_iql_waymo.py -f
        python run_bc_waymo.py -f --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path /home/xinyi/src/data/metadrive/h5py/waymo_n_10000_lam_1.h5py --use_diff_action_space True --num_of_scenarios 10000 --steps 1000000
    else
        # python run_js_iql_waymo.py 
        python run_bc_waymo.py --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path /home/xinyi/src/data/metadrive/h5py/waymo_n_10000_lam_1.h5py --use_diff_action_space True --num_of_scenarios 10000 --steps 1000000
    fi
done