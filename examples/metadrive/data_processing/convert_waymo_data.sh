#!/bin/bash

# 1. collect pkl env data from waymo raw data
# python examples/metadrive/data_processing/waymo_utils.py --tfrecord_dir ~/src/data/metadrive/tfrecord_9/  --pkl_dir ~/src/data/metadrive/pkl_9/ 

# 2. run pkl files in waymo env to collect RL data into a big pkl in feed in Decision Transformer
# ---------------------
num_of_scenarios=50000
total_eps=50
n_per_eps=1000

# make a folder to store waymo tfrecord dataset
mkdir -p ~/src/data/metadrive/dt_pkl/waymo_n_50000_lam_1_eps_10/
# Define a function to run the Python file in the loop
record_scenes() {
    local eps="$1"
    start_seed=$(($eps*$n_per_eps))
    echo "-------------- start_seed = $start_seed -------------"
    # python examples/metadrive/collect_h5py_from_pkl.py --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path ~/src/data/metadrive/h5py/bc_9_n_10000_lam_10.h5py --num_of_scenarios 10000
    python ~/src/safe-sb3/examples/metadrive/data_processing/combine_pkls_for_dt.py --pkl_dir ~/src/data/metadrive/pkl_9/ --dt_data_path ~/src/data/metadrive/dt_pkl/waymo_n_50000_lam_1_eps_10/eps_$eps.pkl --num_of_scenarios $n_per_eps --start_seed $start_seed --lamb 1
}

# loop through all data
for ((i=30; i<50; i++)); do
    echo "Iteration $i"

    # Use parallel to run three instances of the function in parallel
    record_scenes $i

    echo "Iteration $i completed"
done

# ------------------------------
# 3. convert data for bc as well:

# in combine_pkls_for_dt.py, uncomment the last 3 lines

