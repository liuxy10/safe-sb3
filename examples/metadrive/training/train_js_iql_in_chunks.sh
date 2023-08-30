#!/bin/bash
# 1. 
total_timesteps=1000000
num_chunks=50
step_per_chunk = total_timesteps/num_chunks

for (( i=0; i<$rounds; i++ )); do
    if [ $i -eq 0 ]; then
        python run_js_iql_waymo.py --first_round True
    else
        python run_js_iql_waymo.py --first_round False
    fi
done