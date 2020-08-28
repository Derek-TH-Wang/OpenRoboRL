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

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time

from envs.env_wrappers import imitation_wrapper_env
from envs.env_wrappers import observation_dictionary_to_array_wrapper
from task import imitation_task
import learning.imitation_policies as imitation_policies
import learning.ppo_imitation as ppo_imitation

from stable_baselines.common.callbacks import CheckpointCallback

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = False

ROBOT = "laikago"


def set_rand_seed(seed=None):
    if seed is None:
        seed = int(time.time())

    seed += 97 * MPI.COMM_WORLD.Get_rank()

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return


def build_env(robot, motion_files, num_parallel_envs, mode, enable_rendering):
    assert len(motion_files) > 0

    curriculum_episode_length_start = 20
    curriculum_episode_length_end = 600

    env = imitation_task.ImitationTask(robot,
                                       ref_motion_filenames=motion_files,
                                       enable_cycle_sync=True,
                                       tar_frame_steps=[1, 2, 10, 30],
                                       ref_state_init_prob=0.9,
                                       warmup_time=0.25,)

    env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(
        env)

    if mode == "test":
        curriculum_episode_length_start = curriculum_episode_length_end

    env = imitation_wrapper_env.ImitationWrapperEnv(env,
                                                    episode_length_start=curriculum_episode_length_start,
                                                    episode_length_end=curriculum_episode_length_end,
                                                    curriculum_steps=30000000,
                                                    num_parallel_envs=num_parallel_envs)
    return env


def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
    policy_kwargs = {
        "net_arch": [{"pi": [512, 256],
                      "vf": [512, 256]}],
        "act_fun": tf.nn.relu
    }

    timesteps_per_actorbatch = int(
        np.ceil(float(timesteps_per_actorbatch) / num_procs))
    optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

    model = ppo_imitation.PPOImitation(
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
    return model


def train(model, env, total_timesteps, output_dir="", int_save_freq=0):
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

    model.learn(total_timesteps=total_timesteps,
                save_path=save_path, callback=callbacks)

    return


def test(model, env, num_procs, num_episodes=None):
    curr_return = 0
    sum_return = 0
    episode_count = 0

    if num_episodes is not None:
        num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
    else:
        num_local_episodes = np.inf
    # s = time.time()
    o = env.reset()
    while episode_count < num_local_episodes:
        a, _ = model.predict(o, deterministic=True)
        o, r, done, info = env.step(a)
        curr_return += r
        # print(time.time()-s)
        # s = time.time()
        if done:
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
    if ROBOT == "mini_cheetah":
        arg_parser.add_argument("--robot", dest="robot",
                                type=str, default="mini_cheetah")
        arg_parser.add_argument("--motion_file", dest="motion_file", type=str,
                                default="OpenRoboRL/learning/data/motions/dog_trot_xr3.txt")
        arg_parser.add_argument("--model_file", dest="model_file", type=str,
                                default="OpenRoboRL/learning/data/policies/xr3_trot.zip")
        # arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
    elif ROBOT == "laikago":
        arg_parser.add_argument("--robot", dest="robot",
                                type=str, default="laikago")
        arg_parser.add_argument("--motion_file", dest="motion_file", type=str,
                                default="OpenRoboRL/learning/data/motions/dog_pace.txt")
        # arg_parser.add_argument("--model_file", dest="model_file", type=str,
        #                         default="OpenRoboRL/learning/data/policies/dog_pace.zip")
        arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--visualize", dest="visualize",
                            action="store_true", default=True)
    arg_parser.add_argument(
        "--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--num_test_episodes",
                            dest="num_test_episodes", type=int, default=None)
    arg_parser.add_argument("--total_timesteps",
                            dest="total_timesteps", type=int, default=2e8)
    # save intermediate model every n policy steps
    arg_parser.add_argument(
        "--int_save_freq", dest="int_save_freq", type=int, default=0)

    args = arg_parser.parse_args()

    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    # enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test") derektodo
    env = build_env(robot=args.robot,
                    motion_files=[args.motion_file],
                    num_parallel_envs=num_procs,
                    mode=args.mode,
                    enable_rendering=args.visualize)

    model = build_model(env=env,
                        num_procs=num_procs,
                        timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
                        optim_batchsize=OPTIM_BATCHSIZE,
                        output_dir=args.output_dir)

    if args.model_file != "":
        model.load_parameters(args.model_file)

    if args.mode == "train":
        train(model=model,
              env=env,
              total_timesteps=args.total_timesteps,
              output_dir=args.output_dir,
              int_save_freq=args.int_save_freq)
    elif args.mode == "test":
        test(model=model,
             env=env,
             num_procs=num_procs,
             num_episodes=args.num_test_episodes)
    else:
        assert False, "Unsupported mode: " + args.mode

    return


if __name__ == '__main__':
    main()
