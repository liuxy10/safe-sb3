#!/bin/bash

num_of_scenarios=10000
total_eps=10
n_per_eps=1000

# mkdir -p ~/src/data/metadrive/dt_pkl/waymo_n_10000_lam_10/
# Define a function to run the Python file

tada() {
    local eps="$1"
    start_seed=$(($eps*$n_per_eps))
    echo "-------------- start_seed = $start_seed -------------"
    # python examples/metadrive/collect_h5py_from_pkl.py --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path ~/src/data/metadrive/h5py/bc_9_10000_10.h5py --num_of_scenarios 10000
    python ~/src/safe-sb3/examples/metadrive/combine_pkls_for_dt.py --pkl_dir ~/src/data/metadrive/pkl_9/ --dt_data_path ~/src/data/metadrive/dt_pkl/waymo_n_10000_lam_10_eps_10/eps_$eps.pkl --num_of_scenarios $n_per_eps --start_seed $start_seed --lamb 10
}

# loop through all data
for ((i=0; i<total_eps; i++)); do
    echo "Iteration $i"

    # Use parallel to run three instances of the function in parallel
    tada $i

    echo "Iteration $i completed"
done

