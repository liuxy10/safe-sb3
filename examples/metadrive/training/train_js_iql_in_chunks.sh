#!/bin/bash
# 1. 
total_timesteps=1000000
num_chunks=50
step_per_chunk = total_timesteps/num_chunks
last_timestep = 0

for (( i=0; i<$num_chunks; i++ )); do
    echo "------------------------Iteration: $i ------------------------"
    if [ $i -eq 0 ]; then
        # python run_js_iql_waymo.py -f
        python run_bc_waymo.py -f
    else
        # python run_js_iql_waymo.py 
        python run_bc_waymo.py 
    fi
done