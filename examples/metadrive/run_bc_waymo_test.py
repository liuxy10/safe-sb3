def test(args):
    from collect_h5py_from_pkl import get_current_ego_trajectory_old

    file_list = os.listdir(args['pkl_dir'])
    if args['num_of_scenarios'] == 'ALL':
        num_scenarios = len(file_list)
    else:
        num_scenarios = int(args['num_of_scenarios'])

    print("num of scenarios: ", num_scenarios)
    env = AddCostToRewardEnv(
    {
        "manual_control": False,
        "no_traffic": False,
        "agent_policy":PMKinematicsEgoPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "start_seed": 10000, 
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
        "reactive_traffic": False,
            "vehicle_config": dict(
               # no_wheel_friction=True,
               lidar=dict(num_lasers=80, distance=50, num_others=4), # 120
               lane_line_detector=dict(num_lasers=12, distance=50), # 12
               side_detector=dict(num_lasers=20, distance=50)) # 160,
    }, 
    )

    env.seed(args["env_seed"])
    
    
    model = BC("MlpPolicy", env)
    model_dir = args["policy_load_dir"]
    model.set_parameters(model_dir)
    # eval_callback = EvalCallback(env) # don't know how to use it to eval during training, but it seems unnecessary to do so in the first place

    mean_reward, std_reward, mean_success_rate=evaluate_policy(model, env, n_eval_episodes=50, deterministic=True, render=False)
    print("mean_reward, std_reward, mean_success_rate = ", mean_reward, std_reward, mean_success_rate )
    # for seed in range(0, num_scenarios):
    #     plot_waymo_vs_pred(env, model, seed, 'bc', savefig_dir = "examples/metadrive/figs/bc_vs_waymo/diff_action")
      
    del model
    env.close()