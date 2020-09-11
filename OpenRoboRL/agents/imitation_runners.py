# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv


def traj_segment_generator(policy, env, horizon, reward_giver=None, gail=False, callback=None):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
    :param gail: (bool) Whether we are using this generator for standard trpo or with gail
    :param callback: (BaseCallback)
    :return: (dict) generator that returns a dict with the following keys:
        - observations: (np.ndarray) observations
        - rewards: (numpy float) rewards (if gail is used it is the predicted reward)
        - true_rewards: (numpy float) if gail is used it is the original reward
        - vpred: (numpy float) action logits
        - dones: (numpy bool) dones (is end of episode, used for logging)
        - episode_starts: (numpy bool)
            True if first timestep of an episode, used for GAE
        - actions: (np.ndarray) actions
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
        - ep_true_rets: (float) the real environment reward
        - continue_training: (bool) Whether to continue training
            or stop early (triggered by the callback)
    """
    # Check when using GAIL
    assert not (
        gail and reward_giver is None), "You must pass a reward giver when using GAIL"

    num_robot = env.num_robot
    if horizon % num_robot != 0:
        horizon += (num_robot - horizon % num_robot)

    # Initialize state variables
    step = 0
    observation = env.reset()
    # not used, just so we have the datatype
    action = [env.action_space.sample() for _ in range(num_robot)]
    vpred = [0 for _ in range(num_robot)]
    info = [0 for _ in range(num_robot)]
    reward = [0 for _ in range(num_robot)]
    done = [False for _ in range(num_robot)]
    state = [policy.initial_state for _ in range(num_robot)]

    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    current_ep_len = 0  # len of current episode
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    # Initialize history arrays
    observations = np.array([observation[0] for _ in range(horizon)])
    true_rewards = np.zeros(horizon, 'float32')
    rewards = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    nextvpreds = np.zeros(horizon, 'float32')
    episode_starts = np.zeros(horizon, 'bool')
    dones = np.zeros(horizon, 'bool')
    actions = np.array([action[0] for _ in range(horizon)])
    # marks if we're on first timestep of an episode
    episode_start = [True for _ in range(num_robot)]

    callback.on_rollout_start()

    while True:
        for j in range(num_robot):
            observation[j] = observation[j].reshape(-1, *observation[j].shape)
            act, vpred[j], state[j], info[j] = policy.step(
                observation[j], state[j], done[j])
            action[j] = act
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
            last_vpred = 0.0
            for j in range(num_robot):
                nextvpreds[i+j] = last_vpred

            callback.on_rollout_end()
            yield {
                "observations": observations,
                "rewards": rewards,
                "dones": dones,
                "episode_starts": episode_starts,
                "true_rewards": true_rewards,
                "vpred": vpreds,
                "nextvpreds": nextvpreds,
                "actions": actions,
                "ep_rets": ep_rets,
                "ep_lens": ep_lens,
                "ep_true_rets": ep_true_rets,
                "total_timestep": current_it_len,
                'continue_training': True
            }
            for j in range(num_robot):
                _, vpred[j], _, info[j] = policy.step(observation[j])
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            # Reset current iteration length
            current_it_len = 0
            callback.on_rollout_start()

        i = step % horizon
        for j in range(num_robot):
            observations[i+j] = observation[j]
            vpreds[i+j] = vpred[j][0]
            actions[i+j] = action[j][0]
            episode_starts[i+j] = episode_start[j]

            if (not episode_start[j]) and (i > 0):
                nextvpreds[i-num_robot+j] = vpred[j][0]

        clipped_action = [0 for _ in range(num_robot)]
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            for j in range(num_robot):
                clipped_action[j] = np.clip(
                    action[j], env.action_space.low, env.action_space.high)[0]

        # if gail:
        #     reward = reward_giver.get_reward(observation, clipped_action[0])
        #     observation, true_reward, done, info = env.step(clipped_action[0])
        # else:
        #     observation, reward, done, info = env.step(clipped_action[0])
        #     true_reward = reward
        observation, reward, done, info = env.step(clipped_action)
        true_reward = reward

        if callback is not None:
            if callback.on_step() is False:
                # We have to return everything so pytype does not complain
                yield {
                    "observations": observations,
                    "rewards": rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "true_rewards": true_rewards,
                    "vpred": vpreds,
                    "nextvpreds": nextvpreds,
                    "actions": actions,
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "total_timestep": current_it_len,
                    'continue_training': False
                }
                return

        for j in range(num_robot):
            rewards[i+j] = reward[j]
            true_rewards[i+j] = true_reward[j]
            dones[i+j] = done[j]
            episode_start[j] = done[j]

            cur_ep_ret += reward[j]
            cur_ep_true_ret += true_reward[j]
            current_it_len += 1
            current_ep_len += 1

        if np.sum(done) > 0:    # at least one True, then reset
            last_vpred = 0.0
            for j in range(num_robot):
                nextvpreds[i+j] = last_vpred

                # Retrieve unnormalized reward if using Monitor wrapper
                maybe_ep_info = info[j].get('episode')
                if maybe_ep_info is not None:
                    if not gail:
                        cur_ep_ret = maybe_ep_info['r']
                    cur_ep_true_ret = maybe_ep_info['r']

            ep_rets.append(cur_ep_ret / num_robot)
            ep_true_rets.append(cur_ep_true_ret / num_robot)
            ep_lens.append(current_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            current_ep_len = 0

            if not isinstance(env, VecEnv):
                observation = env.reset()

        step += num_robot
    return
