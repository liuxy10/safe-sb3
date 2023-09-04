#!/bin/bash
# 1. cd to the folder, run ./train_js_iql_in_chunks.sh
total_timesteps=1000000
num_chunks=50
step_per_chunk = total_timesteps/num_chunks
last_timestep = 0

for (( i=0; i<$num_chunks; i++ )); do
    echo "------------------------Iteration: $i ------------------------"
    if [ $i -eq 0 ]; then
        # DT iql, uncomment this
        python run_js_iql_waymo.py -f -dt 
        # BC iql uncomment this
        # python run_js_iql_waymo.py -f --expert_model_dir /home/xinyi/src/safe-sb3/examples/metadrive/training/tensorboard_log/bc-waymo-cost-default/BC_0/model.pt
        # BC only, uncomment this
        # python run_bc_waymo.py -f --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path /home/xinyi/src/data/metadrive/h5py/waymo_n_10000_lam_1.h5py --use_diff_action_space True --num_of_scenarios 10000 --steps 1000000
    else
        # DT iql, uncomment this
        python run_js_iql_waymo.py -dt
        # BC iql uncomment this
        # python run_js_iql_waymo.py --expert_model_dir /home/xinyi/src/safe-sb3/examples/metadrive/training/tensorboard_log/bc-waymo-cost-default/BC_0/model.pt
        # BC only, uncomment this
        # python run_bc_waymo.py --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path /home/xinyi/src/data/metadrive/h5py/waymo_n_10000_lam_1.h5py --use_diff_action_space True --num_of_scenarios 10000 --steps 1000000
    fi
done