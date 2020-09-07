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

"""This file implements the locomotion gym env."""
import yaml
import collections
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd


class LocomotionGymEnv(gym.Env):
    """The gym environment for the locomotion tasks."""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self, robot=None, task=None, env_randomizers=None):
        """Initializes the locomotion gym environment.

        Args:
          robot: A class of a robot.
          task: A callable function/class to calculate the reward and termination
            condition. Takes the gym env as the argument when calling.
          env_randomizers: A list of EnvRandomizer(s). An EnvRandomizer may
            randomize the physical property of minitaur, change the terrrain during
            reset(), or add perturbation forces during step().

        Raises:
          ValueError: If the num_action_repeat is less than 1.

        """

        self.seed()
        if robot == None or task == None:
            raise ValueError("no input robot or task")
        self._robot = robot
        self._task = task
        self._env_randomizers = env_randomizers if env_randomizers else []
        self.num_robot = len(robot)
        self._init()

        return

    def seed(self, seed=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]

    def reset(self, reset_visualization_camera=False):
        """Resets the robot's position in the world or rebuild the sim world.

        The simulation world will be rebuilt if self._hard_reset is True.

        Args:
          initial_motor_angles: A list of Floats. The desired joint angles after
            reset. If None, the robot will use its built-in value.
          reset_duration: Float. The time (in seconds) needed to rotate all motors
            to the desired initial values.
          reset_visualization_camera: Whether to reset debug visualization camera on
            reset.

        Returns:
          A numpy array contains the initial observation after reset.
        """
        if self._is_render:
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_RENDERING, 0)

        # Reset the pose of the robot.
        for i in range(self.num_robot):
            self._robot[i].Reset()

        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        self._env_step_counter = 0
        if reset_visualization_camera:
            self._pybullet_client.resetDebugVisualizerCamera(
                3.0, 0, -30, [0, 0, 0])

        if self._is_render:
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_RENDERING, 1)

            if self._task[i] and hasattr(self._task[i], 'reset'):
                self._task[i].reset(self._robot[i], self)

        # Loop over all env randomizers.
        for i in range(self.num_robot):
            for env_randomizer in self._env_randomizers:
                env_randomizer.randomize_env(self._robot[i])

        obs = [self._robot[i]._get_observation()
               for i in range(self.num_robot)]
        obs = self._flatten_observation(obs)

        return obs

    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
          action: Can be a list of desired motor angles for all motors when the
            robot is in position control mode; A list of desired motor torques. Or a
            list of tuples (q, qdot, kp, kd, tau) for hybrid control mode. The
            action must be compatible with the robot's motor control mode. Also, we
            are not going to use the leg space (swing/extension) definition at the
            gym level, since they are specific to Minitaur.

        Returns:
          observations: The observation dictionary. The keys are the sensor names
            and the values are the sensor readings.
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.

        Raises:
          ValueError: The action dimension is not the same as the number of motors.
          ValueError: The magnitude of actions is out of bounds.
        """
        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._env_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            # Also keep the previous orientation of the camera set by the user.
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

            delay = self._pybullet_client.readUserDebugParameter(
                self._delay_id)
            if (delay > 0):
                time.sleep(delay)
        for i in range(self.num_robot):
            for env_randomizer in self._env_randomizers:
                env_randomizer.randomize_step(self._robot[i])

        obs, reward, done = self._step(action)

        return obs, reward, done, {}

    def close(self):
        if hasattr(self, '_robot') and self._robot:
            for i in range(self.num_robot):
                self._robot[i].Terminate()

    def get_ground(self):
        """Get simulation ground model."""
        return self._world_dict['ground']

    def _init(self):
        with open('OpenRoboRL/config/pybullet_sim_param.yaml') as f:
            sim_params_dict = yaml.safe_load(f)
            if "quadruped_robot" in list(sim_params_dict.keys()):
                self._sim_params = sim_params_dict["quadruped_robot"]
            else:
                raise ValueError(
                    "Hyperparameters not found for pybullet_sim_config.yaml")

        self.action_space = self._robot[0].action_space
        self.observation_space = self._flatten_observation_spaces(
            self.robot[0].observation_space)

        # Simulation related parameters.
        self._num_action_repeat = self._robot[0].action_repeat
        self._sim_time_step = self._sim_params["sim_time_step_s"]
        self._env_time_step = self._num_action_repeat * self._sim_time_step
        self._env_step_counter = 0

        self._num_bullet_solver_iterations = int(self._sim_params["num_sim_iter_step"] /
                                                 self._num_action_repeat)
        self._is_render = self._sim_params["enable_rendering"]

        # The wall-clock time at which the last frame is rendered.
        self._last_frame_time = 0.0
        self._show_reference_id = -1

        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(
                connection_mode=pybullet.GUI)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI,
                self._sim_params["enable_rendering_gui"])
            self._delay_id = pybullet.addUserDebugParameter("delay", 0, 0.3, 0)
        else:
            self._pybullet_client = bullet_client.BulletClient(
                connection_mode=pybullet.DIRECT)
        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(
            numSolverIterations=self._num_bullet_solver_iterations)
        self._pybullet_client.setTimeStep(self._sim_time_step)
        self._pybullet_client.setGravity(0, 0, -10)

        # Rebuild the world.
        self._world_dict = {
            "ground": self._pybullet_client.loadURDF("plane_implicit.urdf")
        }

        for i in range(self.num_robot):
            self._robot[i].set_sim_handler(self._pybullet_client)
            self._robot[i].init_robot()

        self.reset()

    def _step(self, action):
        done = [False for _ in range(self.num_robot)]
        obs = [0 for _ in range(self.num_robot)]
        reward = [0 for _ in range(self.num_robot)]

        for i in range(self.num_robot):
            self._robot[i].SetAct(action[i])
        for i in range(self._robot[0].action_repeat):
            for j in range(self.num_robot):
                self._robot[j].RobotStep(i)
            self._pybullet_client.stepSimulation()
            for j in range(self.num_robot):
                self._robot[j].ReceiveObservation()
        for i in range(self.num_robot):
            obs[i] = self._robot[i].GetObs()
        obs = self._flatten_observation(obs)

        for i in range(self.num_robot):
            reward[i] = self._reward(self._task[i])
            self._task[i].update()
            done[i] = self._termination(self._task[i], self._robot[i])

            self._env_step_counter += 1

            if done[i]:
                self._robot[i].Terminate()
        return obs, reward, done

    def _termination(self, task, robot):
        if not robot.is_safe:
            return True
        if task and hasattr(task, 'done'):
            return task.done()
        return False

    def _reward(self, task):
        if task:
            return task()
        return 0

    def _flatten_observation_spaces(self, observation_spaces, observation_excluded=()):
        """Flattens the dictionary observation spaces to gym.spaces.Box.

        If observation_excluded is passed in, it will still return a dictionary,
        which includes all the (key, observation_spaces[key]) in observation_excluded,
        and ('other': the flattened Box space).

        Args:
          observation_spaces: A dictionary of all the observation spaces.
          observation_excluded: A list/tuple of all the keys of the observations to be
            ignored during flattening.

        Returns:
          A box space or a dictionary of observation spaces based on whether
            observation_excluded is empty.
        """
        if not isinstance(observation_excluded, (list, tuple)):
            observation_excluded = [observation_excluded]
        lower_bound = []
        upper_bound = []
        for key, value in observation_spaces.spaces.items():
            if key not in observation_excluded:
                lower_bound.append(np.asarray(value.low).flatten())
                upper_bound.append(np.asarray(value.high).flatten())
        lower_bound = np.concatenate(lower_bound)
        upper_bound = np.concatenate(upper_bound)
        observation_space = spaces.Box(
            np.array(lower_bound), np.array(upper_bound), dtype=np.float32)
        if not observation_excluded:
            return observation_space
        else:
            observation_spaces_after_flatten = {"other": observation_space}
            for key in observation_excluded:
                observation_spaces_after_flatten[key] = observation_spaces[key]
            return spaces.Dict(observation_spaces_after_flatten)

    def _flatten_observation(self, observation_dict, observation_excluded=()):
        """Flattens the observation dictionary to an array.

        If observation_excluded is passed in, it will still return a dictionary,
        which includes all the (key, observation_dict[key]) in observation_excluded,
        and ('other': the flattened array).

        Args:
          observation_dict: A dictionary of all the observations.
          observation_excluded: A list/tuple of all the keys of the observations to be
            ignored during flattening.

        Returns:
          An array or a dictionary of observations based on whether
            observation_excluded is empty.
        """
        if not isinstance(observation_excluded, (list, tuple)):
            observation_excluded = [observation_excluded]

        num_robot = len(observation_dict)
        flat_observations = [0 for _ in range(num_robot)]
        for i in range(num_robot):
            observations = []
            for key, value in observation_dict[i].items():
                if key not in observation_excluded:
                    observations.append(np.asarray(value).flatten())
            flat_observations[i] = np.concatenate(observations)
        if not observation_excluded:
            return flat_observations
        else:
            raise ValueError(
                'flatten_observations observation_excluded is not none')

    @property
    def rendering_enabled(self):
        return self._is_render

    @property
    def world_dict(self):
        return self._world_dict.copy()

    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def env_step_counter(self):
        return self._env_step_counter

    @property
    def env_time_step(self):
        return self._env_time_step

    @property
    def task(self):
        return self._task

    @property
    def robot(self):
        return self._robot
