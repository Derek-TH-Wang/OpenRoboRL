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

from envs.quadruped_robot import locomotion_gym_env
from envs.quadruped_robot import locomotion_gym_config
from envs.quadruped_robot import imitation_wrapper_env
from envs.quadruped_robot import observation_dictionary_to_array_wrapper
from envs.quadruped_robot.task import imitation_task
from envs.utilities.randomizer import controllable_env_randomizer_from_config
from envs.quadruped_robot.robots import laikago


def build_imitation_env(robot, motion_files, num_robot, num_parallel_envs, mode,
                        enable_randomizer, enable_rendering):
  assert len(motion_files) > 0

  curriculum_episode_length_start = 20
  curriculum_episode_length_end = 600
  
  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering

  task = [imitation_task.ImitationTask(ref_motion_filenames=motion_files,
                                      enable_cycle_sync=True,
                                      tar_frame_steps=[1, 2, 10, 30],
                                      ref_state_init_prob=0.9,
                                      warmup_time=0.25)
          for _ in range(num_robot)]

  randomizers = []
  if enable_randomizer:
    randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
    randomizers.append(randomizer)

  env = locomotion_gym_env.LocomotionGymEnv(sim_params=sim_params, name_robot=robot, num_robot = num_robot,
                                            env_randomizers=randomizers, task=task)

  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)

  if mode == "test":
      curriculum_episode_length_start = curriculum_episode_length_end

  env = imitation_wrapper_env.ImitationWrapperEnv(env,
                                                  episode_length_start=curriculum_episode_length_start,
                                                  episode_length_end=curriculum_episode_length_end,
                                                  curriculum_steps=30000000,
                                                  num_parallel_envs=num_parallel_envs)
  return env
