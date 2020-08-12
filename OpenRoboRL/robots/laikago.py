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

"""Pybullet simulation of a Laikago robot."""
import math
import typing
import os
import re
import numpy as np
from gym import spaces
from utils import transformations

from robots import robot_motor
from robots import quadruped
from robots.sensors import environment_sensors
from robots.sensors import sensor_wrappers
from robots.sensors import robot_sensors
from robots.sensors import space_utils


NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_motor_2_chassis_joint",
    "FR_upper_leg_2_hip_motor_joint",
    "FR_lower_leg_2_upper_leg_joint",
    "FL_hip_motor_2_chassis_joint",
    "FL_upper_leg_2_hip_motor_joint",
    "FL_lower_leg_2_upper_leg_joint",
    "RR_hip_motor_2_chassis_joint",
    "RR_upper_leg_2_hip_motor_joint",
    "RR_lower_leg_2_upper_leg_joint",
    "RL_hip_motor_2_chassis_joint",
    "RL_upper_leg_2_hip_motor_joint",
    "RL_lower_leg_2_upper_leg_joint",
]
INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.48]
JOINT_DIRECTIONS = np.array([-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1])
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = -0.6
KNEE_JOINT_OFFSET = 0.66
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

_DEFAULT_HIP_POSITIONS = (
    (0.21, -0.1157, 0),
    (0.21, 0.1157, 0),
    (-0.21, -0.1157, 0),
    (-0.21, 0.1157, 0),
)

ABDUCTION_P_GAIN = 220.0
ABDUCTION_D_GAIN = 0.3
HIP_P_GAIN = 220.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 220.0
KNEE_D_GAIN = 2.0

LAIKAGO_DEFAULT_ABDUCTION_ANGLE = 0
LAIKAGO_DEFAULT_HIP_ANGLE = 0.67
LAIKAGO_DEFAULT_KNEE_ANGLE = -1.25

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([
    LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
    LAIKAGO_DEFAULT_HIP_ANGLE,
    LAIKAGO_DEFAULT_KNEE_ANGLE
] * NUM_LEGS)

_CHASSIS_NAME_PATTERN = re.compile(r"\w+_chassis_\w+")
_MOTOR_NAME_PATTERN = re.compile(r"\w+_hip_motor_\w+")
_KNEE_NAME_PATTERN = re.compile(r"\w+_lower_leg_\w+")
_TOE_NAME_PATTERN = re.compile(r"jtoe\d*")

URDF_FILENAME = "laikago/laikago_toes_limits.urdf"

UPPER_BOUND = 6.28318548203
LOWER_BOUND = -6.28318548203


class Laikago(quadruped.Quadruped):
  """A simulation for the Laikago robot."""
  
  ACTION_CONFIG = [[UPPER_BOUND]*12, [LOWER_BOUND]*12]

  def __init__(self,
      urdf_filename=URDF_FILENAME,
      time_step=0.001,
      action_repeat=33,
      control_latency=0.002,
      on_rack=False,
      enable_action_interpolation=True,
      enable_action_filter=True,
      enable_randomizer=True
  ):
    self._urdf_filename = urdf_filename

    sensors = [
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=NUM_MOTORS), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=environment_sensors.LastActionSensor(num_actions=NUM_MOTORS), num_history=3)
    ]
    # Construct the observation space from the list of sensors. Note that we
    # will reconstruct the observation_space after the robot is created.
    self.observation_space = (
        space_utils.convert_sensors_to_gym_space_dictionary(sensors))
    self.action_space = spaces.Box(
        np.array([LOWER_BOUND]*12),
        np.array([UPPER_BOUND]*12),
        dtype=np.float32)
  
    motor_kp = [ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN]
    motor_kd = [ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN]

    motor_torque_limits = None # jp hack
    
    super(Laikago, self).__init__(
        urdf_path=self._urdf_filename,
        time_step=time_step,
        action_repeat=action_repeat,
        num_motors=NUM_MOTORS,
        name_motor=MOTOR_NAMES,
        dofs_per_leg=DOFS_PER_LEG,
        init_motor_angle = INIT_MOTOR_ANGLES,
        motor_direction=JOINT_DIRECTIONS,
        motor_offset=JOINT_OFFSETS,
        motor_overheat_protection=False,
        motor_model_class=robot_motor.RobotMotorModel,
        sensors=sensors,
        motor_kp=motor_kp,
        motor_kd=motor_kd,
        control_latency=control_latency,
        on_rack=on_rack,
        enable_action_interpolation=enable_action_interpolation,
        enable_action_filter=enable_action_filter,
        enable_randomizer=enable_randomizer)

    return

  def GetHipPositionsInBaseFrame(self):
    return _DEFAULT_HIP_POSITIONS

  def CalFootPositionsInBaseFrame(self):
    """Get the robot's foot position in the base frame."""
    assert len(self._foot_link_ids) == self.num_legs
    foot_positions = []
    for foot_id in self.GetFootLinkIDs():
      foot_positions.append(
          self.GetLinkPos(
              robot=self,
              link_id=foot_id,
          ))
    return np.array(foot_positions)

  def CalJointF2Tau(self, leg_id, contact_force):
    """Maps the foot contact force to the leg joint torques."""
    jv = self.CalJacobian(leg_id)
    all_motor_torques = np.matmul(contact_force, jv)
    motor_torques = {}
    motors_per_leg = self.num_motors // self.num_legs
    com_dof = 6
    for joint_id in range(leg_id * motors_per_leg,
                          (leg_id + 1) * motors_per_leg):
      motor_torques[joint_id] = all_motor_torques[
          com_dof + joint_id] * self._motor_direction[joint_id]

    return motor_torques

  def CalIK(self, leg_id,
                                              foot_local_position):
    """Use IK to compute the motor angles, given the foot link's local position.

    Args:
      leg_id: The leg index.
      foot_local_position: The foot link's position in the base frame.

    Returns:
      A tuple. The position indices and the angles for all joints along the
      leg. The position indices is consistent with the joint orders as returned
      by GetMotorAngles API.
    """
    pass

  def _BuildUrdfIds(self):
    """Build the link Ids from its name in the URDF file.

    Raises:
      ValueError: Unknown category of the joint name.
    """
    num_joints = self.pybullet_client.getNumJoints(self.quadruped)
    self._chassis_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._knee_link_ids = []
    self._foot_link_ids = []

    for i in range(num_joints):
      joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if _CHASSIS_NAME_PATTERN.match(joint_name):
        self._chassis_link_ids.append(joint_id)
      elif _MOTOR_NAME_PATTERN.match(joint_name):
        self._motor_link_ids.append(joint_id)
      # We either treat the lower leg or the toe as the foot link, depending on
      # the urdf version used.
      elif _KNEE_NAME_PATTERN.match(joint_name):
        self._knee_link_ids.append(joint_id)
      elif _TOE_NAME_PATTERN.match(joint_name):
        self._foot_link_ids.append(joint_id)
      else:
        raise ValueError("Unknown category of joint %s" % joint_name)

    self._leg_link_ids.extend(self._knee_link_ids)
    self._leg_link_ids.extend(self._foot_link_ids)
    self._foot_link_ids.extend(self._knee_link_ids)

    self._chassis_link_ids.sort()
    self._motor_link_ids.sort()
    self._foot_link_ids.sort()
    self._leg_link_ids.sort()

    return

  def _GetDefaultInitPosition(self):
    if self._on_rack:
      return INIT_RACK_POSITION
    else:
      return INIT_POSITION

  def _GetDefaultInitOrientation(self):
    # The Laikago URDF assumes the initial pose of heading towards z axis,
    # and belly towards y axis. The following transformation is to transform
    # the Laikago initial orientation to our commonly used orientation: heading
    # towards -x direction, and z axis is the up direction.
    init_orientation = transformations.quaternion_from_euler(math.pi / 2.0, 0, math.pi / 2.0)
    return init_orientation

  def GetDefaultInitPosition(self):
    """Get default initial base position."""
    return self._GetDefaultInitPosition()

  def GetDefaultInitOrientation(self):
    """Get default initial base orientation."""
    return self._GetDefaultInitOrientation()

  def GetDefaultInitJointPose(self):
    """Get default initial joint pose."""
    joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
    return joint_pose
