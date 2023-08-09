#!/bin/bash

eps="$1"

# Define a function to run the Python file
run_collect_h5py_from_pkl() {
    local eps="$1"
    start_seed=$((eps*2000))
    
    # python examples/metadrive/collect_h5py_from_pkl.py --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path ~/src/data/metadrive/h5py/bc_9_10000_10.h5py --num_of_scenarios 10000
    python examples/metadrive/combine_pkls_for_dt.py --pkl_dir ~/src/data/metadrive/pkl_9/ --dt_data_path ~/src/data/metadrive/dt_pkl/bc_9_10000_$eps.pkl --num_of_scenarios 2000 --start_seed $start_seed
}

# Loop until the Python script runs without a segmentation fault
while :
do
    echo "collecting "
    run_collect_h5py_from_pkl $eps
    
    # Check the exit status of the Python script
    if [ $? -eq 0 ]; then
        echo "Script completed successfully."
        break
    # elif [ $? -eq 139 ]; then
    #     echo "Segmentation fault occurred. Retrying..."
    else
        
        echo "Some fault occurred. Retrying..."
        sleep 3
    fi
done
