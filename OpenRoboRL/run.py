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

import yaml
import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time

import agents.imitation_policies as imitation_policies
import agents.ppo_imitation as ppo_imitation
from envs.quadruped_robot import quadruped_gym_env
from envs.quadruped_robot import imitation_wrapper_env
from envs.quadruped_robot.robots import minitaur
from envs.quadruped_robot.task import imitation_task

from stable_baselines.common.callbacks import CheckpointCallback


def set_rand_seed(seed=None):
    if seed is None:
        seed = int(time.time())

    seed += 97 * MPI.COMM_WORLD.Get_rank()

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return


def build_env(name_robot, motion_files, num_robot, num_parallel_envs, mode,
              enable_randomizer):
    assert len(motion_files) > 0

    curriculum_episode_length_start = 20
    curriculum_episode_length_end = 600

    robot = [minitaur.Minitaur(name_robot=name_robot, robot_index=i, enable_randomizer = enable_randomizer)
             for i in range(num_robot)]
    task = [imitation_task.ImitationTask(ref_motion_filenames=motion_files,
                                         enable_cycle_sync=True,
                                         tar_frame_steps=[1, 2, 10, 30],
                                         ref_state_init_prob=0.9,
                                         warmup_time=0.25)
            for _ in range(num_robot)]

    env = quadruped_gym_env.LocomotionGymEnv(robot, task)

    if mode == "test":
        curriculum_episode_length_start = curriculum_episode_length_end

    env = imitation_wrapper_env.ImitationWrapperEnv(env,
                                                    episode_length_start=curriculum_episode_length_start,
                                                    episode_length_end=curriculum_episode_length_end,
                                                    curriculum_steps=30000000,
                                                    num_parallel_envs=num_parallel_envs)
    return env


def build_agent(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
    policy_kwargs = {
        "net_arch": [{"pi": [512, 256],
                      "vf": [512, 256]}],
        "act_fun": tf.nn.relu
    }

    timesteps_per_actorbatch = int(
        np.ceil(float(timesteps_per_actorbatch) / num_procs))
    optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

    agent = ppo_imitation.PPOImitation(
        policy=imitation_policies.ImitationPolicy,
        env=env,
        gamma=0.95,
        timesteps_per_actorbatch=timesteps_per_actorbatch,
        clip_param=0.2,
        optim_epochs=1,
        optim_stepsize=1e-5,
        optim_batchsize=optim_batchsize,
        lam=0.95,
        adam_epsilon=1e-5,
        schedule='constant',
        policy_kwargs=policy_kwargs,
        tensorboard_log=output_dir,
        verbose=1)
    return agent


def train(agent, env, total_timesteps, output_dir="", int_save_freq=0):
    if (output_dir == ""):
        save_path = None
    else:
        save_path = os.path.join(output_dir, "model.zip")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    callbacks = []
    # Save a checkpoint every n steps
    if (output_dir != ""):
        if (int_save_freq > 0):
            int_dir = os.path.join(output_dir, "intermedate")
            callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
                                                name_prefix='model'))

    agent.learn(total_timesteps=total_timesteps,
                save_path=save_path, callback=callbacks)

    return


def test(agent, env, num_procs, num_episodes=None):
    curr_return = 0
    sum_return = 0
    episode_count = 0
    a = [0 for i in range(env.num_robot)]

    if num_episodes is not None:
        num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
    else:
        num_local_episodes = np.inf

    o = env.reset()
    while episode_count < num_local_episodes:
        a, _ = agent.predict(np.array(o), deterministic=True)
        o, r, done, _ = env.step(a)
        for i in range(env.num_robot):
            curr_return += r[i]

        if all(done):
            o = env.reset()
            sum_return += curr_return
            episode_count += 1

    sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
    episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

    mean_return = sum_return / episode_count

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Mean Return: " + str(mean_return))
        print("Episode Count: " + str(episode_count))

    return


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--task", dest="task",
                            type=str, default="imitation_learning_laikago")
    # arg_parser.add_argument("--task", dest="task",
    #                         type=str, default="imitation_learning_minicheetah")
    args = arg_parser.parse_args()

    with open('OpenRoboRL/config/training_param.yaml') as f:
        training_params_dict = yaml.safe_load(f)
        if args.task in list(training_params_dict.keys()):
            training_params = training_params_dict[args.task]
        else:
            raise ValueError(
                "task not found for pybullet_sim_config.yaml")

    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    enable_env_rand = training_params["enable_env_randomizer"] and (
        training_params["mode"] != "test")
    env = build_env(name_robot=training_params["robot"],
                    motion_files=[training_params["motion_file"]],
                    num_robot=training_params["num_robot"],
                    num_parallel_envs=num_procs,
                    mode=training_params["mode"],
                    enable_randomizer=enable_env_rand)

    agent = build_agent(env=env,
                        num_procs=num_procs,
                        timesteps_per_actorbatch=training_params["timestep_per_actorbach"],
                        optim_batchsize=training_params["optim_batchsize"],
                        output_dir=training_params["output_dir"])

    if training_params["model_file"] != "":
        agent.load_parameters(training_params["model_file"])

    if training_params["mode"] == "train":
        train(agent=agent,
              env=env,
              total_timesteps=training_params["total_timesteps"],
              output_dir=training_params["output_dir"],
              int_save_freq=training_params["int_save_freq"])
    elif training_params["mode"] == "test":
        test(agent=agent,
             env=env,
             num_procs=num_procs,
             num_episodes=training_params["num_test_episodes"])
    else:
        assert False, "Unsupported mode: " + training_params["mode"]

    return


if __name__ == '__main__':
    main()
