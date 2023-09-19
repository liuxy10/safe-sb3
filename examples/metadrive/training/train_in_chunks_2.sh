#!/bin/bash
# 1. cd to the folder, run ./train_js_iql_in_chunks.sh
total_timesteps=1000000
num_chunks=100
step_per_chunk=10000

for (( i=0; i<$num_chunks; i++ )); do
    echo "------------------------Iteration: $i ------------------------"
    if [ $i -eq 0 ]; then
        # DT iql, uncomment this
        python run_js_iql_waymo_copy.py -f -dt --num_of_scenarios 10000 --steps $step_per_chunk

        # BC iql uncomment this
        # python run_js_iql_waymo.py -f --num_of_scenarios 10000 --steps $step_per_chunk --expert_model_dir /home/xinyi/src/safe-sb3/examples/metadrive/training/tensorboard_log/bc-waymo-cost-default/BC_0/model.pt
        
        # BC only, uncomment this
        # python run_bc_waymo.py -f --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path /home/xinyi/src/data/metadrive/h5py/waymo_n_10000_lam_1.h5py --use_diff_action_space True --num_of_scenarios 10000 --steps 1000000
        
        # iql only, uncomment this
        # python /home/xinyi/src/safe-sb3/examples/metadrive/training/run_iql_waymo.py -es 1 -f --num_of_scenarios 10000 --steps $step_per_chunk  
    else
        # DT iql, uncomment this
        python run_js_iql_waymo_copy.py -dt --num_of_scenarios 10000 --steps $step_per_chunk
        
        # BC iql uncomment this
        # python run_js_iql_waymo.py --num_of_scenarios 10000 --steps $step_per_chunk --expert_model_dir /home/xinyi/src/safe-sb3/examples/metadrive/training/tensorboard_log/bc-waymo-cost-default/BC_0/model.pt
       
        # BC only, uncomment this
        # python run_bc_waymo.py --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path /home/xinyi/src/data/metadrive/h5py/waymo_n_10000_lam_1.h5py --use_diff_action_space True --num_of_scenarios 10000 --steps 1000000
       
        # iql only, uncomment this
        # python /home/xinyi/src/safe-sb3/examples/metadrive/training/run_iql_waymo.py -es 1 --num_of_scenarios 10000 --steps $step_per_chunk  
    fi
done