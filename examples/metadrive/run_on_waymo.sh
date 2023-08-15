# collect pkl env data from waymo raw data
# python examples/metadrive/waymo_utils.py --tfrecord_dir examples/metadrive/tfrecord_9 --pkl_dir examples/metadrive/pkl_9
# alternatively on server:
# python examples/metadrive/waymo_utils.py --tfrecord_dir ~/src/data/metadrive/tfrecord_9/  --pkl_dir ~/src/data/metadrive/pkl_9/ 

# run pkl files in waymo env to collect h5py RL data for offline RL training
python examples/metadrive/collect_h5py_from_pkl.py --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path ~/src/data/metadrive/h5py/bc_9_900.h5py --num_of_scenarios 900 

# run pkl files in waymo env to collect RL data into a big pkl in feed in Decision Transformer
python examples/metadrive/combine_pkls_for_dt.py --pkl_dir ~/src/data/metadrive/pkl_9/ --dt_data_path ~/src/data/metadrive/dt_pkl/bc_9_900.pkl --num_of_scenarios 10


# train offline BC policy 
# python examples/metadrive/run_bc_waymo.py --pkl_dir examples/metadrive/pkl_9 --h5py_path examples/metadrive/h5py/one_pack_training.h5py --output_dir examples/metadrive/saved_bc_policy

# alternatively on server:
python examples/metadrive/run_bc_waymo.py --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path /home/xinyi/src/data/metadrive/h5py/waymo_n_10000_lam_1.h5py --use_diff_action_space True --num_of_scenarios 10000 --steps 1000000
# python examples/metadrive/run_sac_waymo.py --pkl_dir ~/src/data/metadrive/pkl_9/ --use_diff_action_space True --output_dir ~/src/data/metadrive/saved_sac_policy/ --num_of_scenarios 900 --steps 1000000 --save_freq 10000

# train offline DT policy 
python /home/xinyi/src/decision-transformer/gym/experiment_waymo.py 



# train SAC with BC JumpStart (compared with sac-waymo-es0/SAC_7 and bc-waymo-es0/BC_0)
python examples/metadrive/run_js_sac_waymo.py --pkl_dir ~/src/data/metadrive/pkl_9/  --num_of_scenarios 10000 --steps 1000000 --device cuda --expert_model_dir tensorboard_log/bc-waymo-es0/BC_0/model.pt --use_diff_action=True
# train iql with BC JumpStart
python examples/metadrive/run_js_iql_waymo.py --use_transformer_expert False --pkl_dir ~/src/data/metadrive/pkl_9/  --num_of_scenarios 10000 --steps 1000000 --device cuda --expert_model_dir tensorboard_log/bc-waymo-es0/BC_0/model.pt --use_diff_action=True

python examples/metadrive/run_js_iql_waymo.py --use_transformer_expert True --pkl_dir ~/src/data/metadrive/pkl_9/  --num_of_scenarios 10000 --steps 1000000 --device cuda --expert_model_dir  --use_diff_action=True


