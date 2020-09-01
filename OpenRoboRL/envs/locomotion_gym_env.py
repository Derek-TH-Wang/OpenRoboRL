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
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd

from envs.sensors import sensor
from envs.sensors import space_utils


_ACTION_EPS = 0.01
_NUM_SIMULATION_ITERATION_STEPS = 300
_LOG_BUFFER_LENGTH = 5000

class LocomotionGymEnv(gym.Env):
  """The gym environment for the locomotion tasks."""
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 100
  }

  def __init__(self,
               gym_config,
               robot_class=None,
               num_robot=1,
               env_sensors=None,
               robot_sensors=None,
               task=None,
               env_randomizers=None):
    """Initializes the locomotion gym environment.

    Args:
      gym_config: An instance of LocomotionGymConfig.
      robot_class: A class of a robot. We provide a class rather than an
        instance due to hard_reset functionality. Parameters are expected to be
        configured with gin.
      sensors: A list of environmental sensors for observation.
      task: A callable function/class to calculate the reward and termination
        condition. Takes the gym env as the argument when calling.
      env_randomizers: A list of EnvRandomizer(s). An EnvRandomizer may
        randomize the physical property of minitaur, change the terrrain during
        reset(), or add perturbation forces during step().

    Raises:
      ValueError: If the num_action_repeat is less than 1.

    """

    self.seed()
    self._gym_config = gym_config
    self._robot_class = robot_class
    self.num_robot = num_robot
    self._robot_sensors = robot_sensors

    self._sensors = env_sensors if env_sensors is not None else list()
    if self._robot_class is None:
      raise ValueError('robot_class cannot be None.')

    # A dictionary containing the objects in the world other than the robot.
    self._world_dict = {}
    self._task = task

    self._env_randomizers = env_randomizers if env_randomizers else []

    # This is a workaround due to the issue in b/130128505#comment5
    # if isinstance(self._task, sensor.Sensor):
    #   self._sensors.append(self._task)

    # Simulation related parameters.
    self._num_action_repeat = gym_config.simulation_parameters.num_action_repeat
    self._on_rack = gym_config.simulation_parameters.robot_on_rack
    if self._num_action_repeat < 1:
      raise ValueError('number of action repeats should be at least 1.')
    self._sim_time_step = gym_config.simulation_parameters.sim_time_step_s
    self._env_time_step = self._num_action_repeat * self._sim_time_step
    self._env_step_counter = 0

    self._num_bullet_solver_iterations = int(_NUM_SIMULATION_ITERATION_STEPS /
                                             self._num_action_repeat)
    self._is_render = gym_config.simulation_parameters.enable_rendering

    # The wall-clock time at which the last frame is rendered.
    self._last_frame_time = 0.0
    self._show_reference_id = -1
    
    if self._is_render:
      self._pybullet_client = bullet_client.BulletClient(
          connection_mode=pybullet.GUI)
      pybullet.configureDebugVisualizer(
          pybullet.COV_ENABLE_GUI,
          gym_config.simulation_parameters.enable_rendering_gui)
      self._show_reference_id = pybullet.addUserDebugParameter("show reference",0,1,0.5)
      self._delay_id = pybullet.addUserDebugParameter("delay",0,0.3,0)
    else:
      self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
    if gym_config.simulation_parameters.egl_rendering:
      self._pybullet_client.loadPlugin('eglRendererPlugin')

    # The action list contains the name of all actions.
    self._action_list = []
    action_upper_bound = []
    action_lower_bound = []
    action_config = robot_class[0].ACTION_CONFIG
    for action in action_config:
      self._action_list.append(action.name)
      action_upper_bound.append(action.upper_bound)
      action_lower_bound.append(action.lower_bound)

    self.action_space = spaces.Box(
        np.array(action_lower_bound),
        np.array(action_upper_bound),
        dtype=np.float32)

    # Set the default render options.
    self._camera_dist = gym_config.simulation_parameters.camera_distance
    self._camera_yaw = gym_config.simulation_parameters.camera_yaw
    self._camera_pitch = gym_config.simulation_parameters.camera_pitch
    self._render_width = gym_config.simulation_parameters.render_width
    self._render_height = gym_config.simulation_parameters.render_height

    self._hard_reset = True
    self.reset()

    self._hard_reset = gym_config.simulation_parameters.enable_hard_reset

    # Construct the observation space from the list of sensors. Note that we
    # will reconstruct the observation_space after the robot is created.
    self.observation_space = (
        space_utils.convert_sensors_to_gym_space_dictionary(self.all_sensors(self._robot[0])))

  def close(self):
    if hasattr(self, '_robot') and self._robot:
      for i in range(self.num_robot):
        self._robot[i].Terminate()

  def seed(self, seed=None):
    self.np_random, self.np_random_seed = seeding.np_random(seed)
    return [self.np_random_seed]

  def all_sensors(self, robot):
    """Returns all robot and environmental sensors."""
    return robot.GetAllSensors() + self._sensors

  def sensor_by_name(self, robot, name):
    """Returns the sensor with the given name, or None if not exist."""
    for sensor_ in self.all_sensors(robot):
      if sensor_.get_name() == name:
        return sensor_
    return None

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
      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_RENDERING, 0)

    # Clear the simulation world and rebuild the robot interface.
    if self._hard_reset:
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=self._num_bullet_solver_iterations)
      self._pybullet_client.setTimeStep(self._sim_time_step)
      self._pybullet_client.setGravity(0, 0, -10)

      # Rebuild the world.
      self._world_dict = {
          "ground": self._pybullet_client.loadURDF("plane_implicit.urdf")
      }

      # Rebuild the robot
      self._robot = [self._robot_class[i](
          pybullet_client=self._pybullet_client,
          robot_index = i,
          sensors=self._robot_sensors[i],
          on_rack=self._on_rack) for i in range(self.num_robot)]

    # Reset the pose of the robot.
    for i in range(self.num_robot):
      self._robot[i].Reset(
          reload_urdf=False,
          default_motor_angles=initial_motor_angles,
          reset_time=reset_duration)

    self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
    self._env_step_counter = 0
    # if reset_visualization_camera:
      # self._pybullet_client.resetDebugVisualizerCamera(self._camera_dist,
      #                                                  self._camera_yaw,
      #                                                  self._camera_pitch,
      #                                                  [0, 0, 0])

    if self._is_render:
      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_RENDERING, 1)

    for i in range(self.num_robot):
      for s in self.all_sensors(self._robot[i]):
        s.on_reset(self._robot[i])

      if self._task[i] and hasattr(self._task[i], 'reset'):
        self._task[i].reset(self._robot[i], self)

    # Loop over all env randomizers.
    for i in range(self.num_robot):
      for env_randomizer in self._env_randomizers:
        env_randomizer.randomize_env(self._robot[i])

    obs = [self._get_observation(self._robot[i]) for i in range(self.num_robot)]

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
        self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING,1)
      
      # alpha = self._pybullet_client.readUserDebugParameter(self._show_reference_id)
      # ref_col = [1, 1, 1, alpha]
      # self._pybullet_client.changeVisualShape(self._task._ref_model, -1, rgbaColor=ref_col)
      # for l in range (self._pybullet_client.getNumJoints(self._task._ref_model)):
      # 	self._pybullet_client.changeVisualShape(self._task._ref_model, l, rgbaColor=ref_col)
    
      delay = self._pybullet_client.readUserDebugParameter(self._delay_id)
      if (delay>0):
        time.sleep(delay)
    for i in range(self.num_robot):
      for env_randomizer in self._env_randomizers:
        env_randomizer.randomize_step(self._robot[i])

    # robot class and put the logics here.
    done = [False for _ in range(self.num_robot)]
    obs = [0 for _ in range(self.num_robot)]
    reward = [0 for _ in range(self.num_robot)]


    for i in range(self.num_robot):
      self._robot[i].SetAct(action[i])
    for i in range(self._robot[i].action_repeat):
      for j in range(self.num_robot):
        self._robot[j].RobotStep(i)
      self._pybullet_client.stepSimulation()
      for j in range(self.num_robot):
        self._robot[j].ReceiveObservation()

    # for i in range(self.num_robot):
    #   self._robot[i].Step(action[i])


    for i in range(self.num_robot):
      reward[i] = self._reward(self._task[i])

      for s in self.all_sensors(self._robot[i]):
        s.on_step()
      
      if self._task[i] and hasattr(self._task[i], 'update'):
        self._task[i].update()
      
      done[i] = self._termination(self._task[i], self._robot[i])

      self._env_step_counter += 1

      if done[i]:
        self._robot[i].Terminate()
      
      obs[i] = self._get_observation(self._robot[i])

    return obs, reward, done, {}

  def render(self, mode='rgb_array'):
    if mode != 'rgb_array':
      raise ValueError('Unsupported render mode:{}'.format(mode))
    base_pos = self._robot[0].GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._camera_dist,
        yaw=self._camera_yaw,
        pitch=self._camera_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(self._render_width) / self._render_height,
        nearVal=0.1,
        farVal=100.0)
    (_, _, px, _, _) = self._pybullet_client.getCameraImage(
        width=self._render_width,
        height=self._render_height,
        renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
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
  def world_dict(self):
    return self._world_dict.copy()

  @world_dict.setter
  def world_dict(self, new_dict):
    self._world_dict = new_dict.copy()

  def _termination(self, task, robot):
    if not robot.is_safe:
      return True

    if task and hasattr(task, 'done'):
      return task.done()

    for s in self.all_sensors(robot):
      s.on_terminate()

    return False

  def _reward(self, task):
    if task:
      return task()
    return 0

  def _get_observation(self, robot):
    """Get observation of this environment from a list of sensors.

    Returns:
      observations: sensory observation in the numpy array format
    """
    sensors_dict = {}
    for s in self.all_sensors(robot):
      sensors_dict[s.get_name()] = s.get_observation()

    observations = collections.OrderedDict(sorted(list(sensors_dict.items())))
    return observations

  @property
  def pybullet_client(self):
    return self._pybullet_client

  @property
  def robot(self):
    return self._robot

  @property
  def env_step_counter(self):
    return self._env_step_counter

  @property
  def hard_reset(self):
    return self._hard_reset

  @property
  def env_time_step(self):
    return self._env_time_step

  @property
  def task(self):
    return self._task

  @property
  def robot_class(self):
    return self._robot_class
