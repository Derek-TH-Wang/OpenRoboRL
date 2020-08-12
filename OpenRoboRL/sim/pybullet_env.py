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
import collections
import time
import gym
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd
from sim import sim_config
from envs.utilities import controllable_env_randomizer_from_config


_ACTION_EPS = 0.01
_NUM_SIMULATION_ITERATION_STEPS = 300
_LOG_BUFFER_LENGTH = 5000


class PybulletEnv(gym.Env):
  """The gym environment for the pybullet simulator."""
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 100
  }

  def __init__(self,
               time_step=0.001,
               action_repeat=33,
               on_rack=False,
               enable_randomizer=True):
    """Initializes the locomotion gym environment.

    Args:
      gym_config: An instance of LocomotionGymConfig.
      env_randomizers: A list of EnvRandomizer(s). An EnvRandomizer may
        randomize the physical property of minitaur, change the terrrain during
        reset(), or add perturbation forces during step().

    Raises:
      ValueError: If the num_action_repeat is less than 1.

    """

    self._env_randomizers = []
    if enable_randomizer:
      randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
      self._env_randomizers.append(randomizer)

    self.seed()
    self._sim_config = sim_config.SimulationParameters()

    self._world_dict = {}

    # Simulation related parameters.
    self.num_action_repeat = action_repeat
    self._on_rack = on_rack
    if self.num_action_repeat < 1:
      raise ValueError('number of action repeats should be at least 1.')
    self.sim_time_step = time_step
    self._env_time_step = self.num_action_repeat * self.sim_time_step
    self._env_step_counter = 0

    self._num_bullet_solver_iterations = int(_NUM_SIMULATION_ITERATION_STEPS /
                                             self.num_action_repeat)
    self._is_render = self._sim_config.enable_rendering

    # The wall-clock time at which the last frame is rendered.
    self._last_frame_time = 0.0
    self._show_reference_id = -1
    
    if self._is_render:
      self.pybullet_client = bullet_client.BulletClient(
          connection_mode=pybullet.GUI)
      pybullet.configureDebugVisualizer(
          pybullet.COV_ENABLE_GUI,
          self._sim_config.enable_rendering_gui)
      self._show_reference_id = pybullet.addUserDebugParameter("show reference",0,1,
        self._sim_config.draw_ref_model_alpha)
      self._delay_id = pybullet.addUserDebugParameter("delay",0,0.3,0)
    else:
      self.pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    self.pybullet_client.setAdditionalSearchPath(pd.getDataPath())
    if self._sim_config.egl_rendering:
      self.pybullet_client.loadPlugin('eglRendererPlugin')

    # Set the default render options.
    self._camera_dist = self._sim_config.camera_distance
    self._camera_yaw = self._sim_config.camera_yaw
    self._camera_pitch = self._sim_config.camera_pitch
    self._render_width = self._sim_config.render_width
    self._render_height = self._sim_config.render_height

    self._hard_reset = True
    self.reset()
    self._hard_reset = self._sim_config.enable_hard_reset

  def close(self):
    raise NotImplementedError()

  def seed(self, seed=None):
    self.np_random, self.np_random_seed = seeding.np_random(seed)
    return [self.np_random_seed]

  def reset(self,
            initial_motor_angles=None,
            reset_duration=0.0,
            reset_visualization_camera=True):
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
      self.pybullet_client.configureDebugVisualizer(
          self.pybullet_client.COV_ENABLE_RENDERING, 0)

    # Clear the simulation world and rebuild the robot interface.
    if self._hard_reset:
      self.pybullet_client.resetSimulation()
      self.pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=self._num_bullet_solver_iterations)
      self.pybullet_client.setTimeStep(self.sim_time_step)
      self.pybullet_client.setGravity(0, 0, -10)

      # Rebuild the world.
      self._world_dict = {
          "ground": self.pybullet_client.loadURDF("plane_implicit.urdf")
      }

    # Reset the pose of the robot.
    obs = self.reset_robot(reload_urdf=self._hard_reset, 
                            default_motor_angles=initial_motor_angles, 
                            reset_time=reset_duration)

    self.pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
    self._env_step_counter = 0
    if reset_visualization_camera:
      self.pybullet_client.resetDebugVisualizerCamera(self._camera_dist,
                                                       self._camera_yaw,
                                                       self._camera_pitch,
                                                       [0, 0, 0])

    if self._is_render:
      self.pybullet_client.configureDebugVisualizer(
          self.pybullet_client.COV_ENABLE_RENDERING, 1)

    self.reset_task()

    # Loop over all env randomizers.
    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_env(self)

    return obs

  def reset_robot(self, reload_urdf, default_motor_angles, reset_time):
    raise NotImplementedError()

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
      observations: The observation dictionary.
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
      base_pos = self.GetBasePosition()

      # Also keep the previous orientation of the camera set by the user.
      [yaw, pitch,
       dist] = self.pybullet_client.getDebugVisualizerCamera()[8:11]
      self.pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch,
                                                       base_pos)
      self.pybullet_client.configureDebugVisualizer(
        self.pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING,1)
      
      # alpha = self.pybullet_client.readUserDebugParameter(self._show_reference_id)
      # ref_col = [1, 1, 1, alpha]
      # self.pybullet_client.changeVisualShape(self._task._ref_model, -1, rgbaColor=ref_col)
      # for l in range (self.pybullet_client.getNumJoints(self._task._ref_model)):
      # 	self.pybullet_client.changeVisualShape(self._task._ref_model, l, rgbaColor=ref_col)
    
      delay = self.pybullet_client.readUserDebugParameter(self._delay_id)
      if (delay>0):
        time.sleep(delay)
    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_step(self)

    obs = self.set_action(action)
    reward = self.compute_reward()
    done = self.is_done()

    self._env_step_counter += 1

    return obs, reward, done, {}

  def compute_reward(self):
    raise NotImplementedError()

  def GetBasePosition(self):
    raise NotImplementedError()

  def set_action(self, action):
    raise NotImplementedError()

  def is_done(self):
    raise NotImplementedError()

  def render(self, base_pos, mode='rgb_array'):
    if mode != 'rgb_array':
      raise ValueError('Unsupported render mode:{}'.format(mode))
    view_matrix = self.pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._camera_dist,
        yaw=self._camera_yaw,
        pitch=self._camera_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self.pybullet_client.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(self._render_width) / self._render_height,
        nearVal=0.1,
        farVal=100.0)
    (_, _, px, _, _) = self.pybullet_client.getCameraImage(
        width=self._render_width,
        height=self._render_height,
        renderer=self.pybullet_client.ER_BULLET_HARDWARE_OPENGL,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def get_ground(self):
    """Get simulation ground model."""
    return self._world_dict['ground']

  def set_ground(self, ground_id):
    """Set simulation ground model."""
    self._world_dict['ground'] = ground_id

  @property
  def rendering_enabled(self):
    return self._is_render

  @property
  def last_base_position(self):
    return self._last_base_position

  @property
  def world_dict(self):
    return self._world_dict.copy()

  @world_dict.setter
  def world_dict(self, new_dict):
    self._world_dict = new_dict.copy()

  def set_time_step(self, num_action_repeat, sim_step=0.001):
    """Sets the time step of the environment.

    Args:
      num_action_repeat: The number of simulation steps/action repeats to be
        executed when calling env.step().
      sim_step: The simulation time step in PyBullet. By default, the simulation
        step is 0.001s, which is a good trade-off between simulation speed and
        accuracy.

    Raises:
      ValueError: If the num_action_repeat is less than 1.
    """
    if num_action_repeat < 1:
      raise ValueError('number of action repeats should be at least 1.')
    self.sim_time_step = sim_step
    self.num_action_repeat = num_action_repeat
    self._env_time_step = sim_step * num_action_repeat
    self._num_bullet_solver_iterations = (
        _NUM_SIMULATION_ITERATION_STEPS / self.num_action_repeat)
    self.pybullet_client.setPhysicsEngineParameter(
        numSolverIterations=int(np.round(self._num_bullet_solver_iterations)))
    self.pybullet_client.setTimeStep(self.sim_time_step)

  # @property
  # def pybullet_client(self):
  #   return self.pybullet_client

  @property
  def env_step_counter(self):
    return self._env_step_counter

  @property
  def hard_reset(self):
    return self._hard_reset

  @property
  def env_time_step(self):
    return self._env_time_step
