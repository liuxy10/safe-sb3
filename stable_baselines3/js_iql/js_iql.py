from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union
import io
import pathlib

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.iql.iql import IQL
from stable_baselines3.iql.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy, ValueNet
from stable_baselines3.js_sac import utils as js_utils

SelfJumpStartIQL = TypeVar("SelfJumpStartIQL", bound="JumpStartIQL")


class JumpStartIQL(IQL):
    """
    Jump Start Implicit Q Learning (js-IQL).
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic
    value: ValueNet

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        expert_policy: Any,
        # data_collection_env: GymEnv,
        use_transformer_expert: bool,
        target_return: Optional[float] = None,
        reward_scale: Optional[float] = None,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        guidance_timesteps: int = 500_000,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            _init_setup_model=_init_setup_model,
        )
        # self.data_collection_env = data_collection_env
        self.guidance_timesteps = guidance_timesteps
        js_utils.check_expert_policy(expert_policy, self)
        self.expert_policy = expert_policy
        self.use_transformer_expert = use_transformer_expert
        if self.use_transformer_expert:
            assert target_return is not None
            assert reward_scale is not None
            assert obs_mean is not None
            assert obs_std is not None
            self.obs_mean = th.from_numpy(obs_mean).to(device=device)
            self.obs_std = th.from_numpy(obs_std).to(device=device)
        self.target_return_init = target_return
        self.reward_scale = reward_scale
        self.num_steps_per_save = 100



    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        
        # Select action randomly or according to policy
        self.hist_ac  = np.concatenate(
            [self.hist_ac, np.zeros((1, self.ac_dim))], axis=0
        )
        self.hist_re = np.concatenate([self.hist_re, np.zeros(1)])
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            guide_prob = self.get_guide_probability()
            # print("num_timesteps, guide_prob = ", self.num_timesteps, guide_prob)
            use_guide = np.random.choice([False, True], p=[1-guide_prob, guide_prob])
            if use_guide:
                if self.use_transformer_expert:
                    hist_obs = th.tensor(
                        self.hist_obs, dtype=th.float32, device=self.device
                    )
                    hist_ac = th.tensor(
                        self.hist_ac, dtype=th.float32, device=self.device
                    )
                    hist_re = th.tensor(
                        self.hist_re, dtype=th.float32, device=self.device
                    )
                    target_return = th.tensor(
                        self.target_return, dtype=th.float32, device=self.device
                    )
                    timesteps = th.tensor(
                        self.timesteps, dtype=th.long, device=self.device
                    )
                    unscaled_action = self.expert_policy.get_action(
                        (hist_obs - self.obs_mean) / self.obs_std,
                        hist_ac,
                        hist_re,
                        target_return,
                        timesteps,
                    )
                    unscaled_action = unscaled_action.detach().cpu().numpy()
                    unscaled_action = unscaled_action.reshape(1,2)
                    # print("[use_guide] unscaled_action.shape", unscaled_action.shape)
                else:
                    unscaled_action, _ = self.expert_policy.predict(
                        self._last_obs, deterministic=False
                    )
                    
            else:
                # Note: when using continuous actions,
                # we assume that the policy uses tanh to scale the action
                # We use non-deterministic action in the case of SAC, for TD3, it does not matter
                unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        self.hist_ac[-1] = unscaled_action
        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            # import pdb; pdb.set_trace()
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.hist_obs = np.concatenate([self.hist_obs, new_obs], axis=0)
            assert len(self.hist_obs.shape) == 2
            self.hist_re[-1] = rewards
            # TODO: delete this when updated env 
            self.target_return = np.array([[400]])
            ##############################
            pred_return = self.target_return[0,-1] - (rewards/self.reward_scale) # question: what is prediction return?
            # import pdb; pdb.set_trace() # 
            self.target_return = np.concatenate(
                [self.target_return, pred_return.reshape(1, 1)], axis=1)
            t = self.timesteps[0, -1] + 1
            self.timesteps = np.concatenate(
                [self.timesteps, np.ones((1, 1)) * t], axis=1)

            assert dones.shape == (1, )
            if dones:
                
                self.hist_obs = self._last_obs.reshape(1, self.obs_dim)
                self.hist_ac= np.zeros((0, self.ac_dim))
                self.hist_re = np.zeros(0)
                if self.use_transformer_expert:
                    self.target_return = np.array([[self.target_return_init]])
                self.timesteps = np.zeros((1, 1))

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            from stable_baselines3.bc.policies import BCPolicy
            if not isinstance(self.policy, BCPolicy):
                self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def get_guide_probability(self):
        if self.num_timesteps > self.guidance_timesteps:
            return 0.
        prob_start = 0.9
        prob = prob_start * np.exp(-5. * self.num_timesteps / self.guidance_timesteps)
        return prob


    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfJumpStartIQL],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfJumpStartIQL:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError(
                "The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            # pytype: disable=unsupported-operands
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(
                env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # pytype: disable=not-instantiable,wrong-keyword-args
        
        if cls == JumpStartIQL:
            kwargs = kwargs['kwargs']
            model = cls(
                policy=data["policy_class"],
                env=env, 
                expert_policy = kwargs["expert_policy"],
                use_transformer_expert = kwargs["use_transformer_expert"],
                target_return= kwargs["target_return"],
                reward_scale= kwargs ["reward_scale"],
                obs_mean = kwargs["obs_mean"],
                obs_std = kwargs["obs_std"],
                tensorboard_log= kwargs["tensorboard_log"],
                verbose=1,
                device = device
            )
        else:
            model = cls(
                policy=data["policy_class"],
                env=env,
                device=device,
                _init_setup_model=False,  # type: ignore[call-arg]
            )


        # pytype: enable=not-instantiable,wrong-keyword-args

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(
                    model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            # type: ignore[operator]  # pytype: disable=attribute-error
            model.policy.reset_noise()
        return model
