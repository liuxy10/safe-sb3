
# run bc training
python examples/metadrive/training/run_bc_waymo.py --pkl_dir ~/src/data/metadrive/pkl_9/ --h5py_path /home/xinyi/src/data/metadrive/h5py/waymo_n_10000_lam_1.h5py --use_diff_action_space True --num_of_scenarios 10000 --steps 1000000

# test policy: 
python examples/metadrive/training/run_bc_waymo.py --pkl_dir ~/src/data/metadrive/pkl_9/ --policy_load_dir examples/metadrive/example_policy/bc-diff-peak-10000.pt --use_diff_action_space True --num_of_scenarios 10000 --is_test True


# train offline DT policy 
python /home/xinyi/src/decision-transformer/gym/experiment_waymo.py 



# train SAC with BC JumpStart (compared with sac-waymo-es0/SAC_7 and bc-waymo-es0/BC_0)
python examples/metadrive/training/run_js_sac_waymo.py --pkl_dir ~/src/data/metadrive/pkl_9/  --num_of_scenarios 10000 --steps 1000000 --device cuda --expert_model_dir tensorboard_log/bc-waymo-es0/BC_0/model.pt --use_diff_action=True
# train iql with BC JumpStart
python examples/metadrive/training/run_js_iql_waymo.py --use_transformer_expert False --pkl_dir ~/src/data/metadrive/pkl_9/  --num_of_scenarios 10000 --steps 1000000 --device cpu   --policy_load_dir tensorboard_log/bc-waymo-es0/BC_0/model.pt --use_diff_action=True

python examples/metadrive/training/run_js_iql_waymo.py --use_transformer_expert True --pkl_dir ~/src/data/metadrive/pkl_9/  --num_of_scenarios 10000 --steps 1000000 --device cuda  --use_diff_action=True


