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

"""Pybullet simulation of a xr3 robot."""
import math
import typing
import os
import re
import numpy as np
from utils import transformations

from robots import robot_motor
from sim import pybullet_quadruped
from robots.sensors import environment_sensors
from robots.sensors import sensor_wrappers
from robots.sensors import robot_sensors
from robots.sensors import sensor
from robots.sensors import space_utils


NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "abduct_fl",
    "thigh_fl",
    "knee_fl",
    "abduct_hl",
    "thigh_hl",
    "knee_hl",
    "abduct_fr",
    "thigh_fr",
    "knee_fr",
    "abduct_hr",
    "thigh_hr",
    "knee_hr",
]
INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.28]
JOINT_DIRECTIONS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) #to be done
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0 #to be done
KNEE_JOINT_OFFSET = 0.0 #to be done
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

_DEFAULT_HIP_POSITIONS = (
    (0.38, -0.1161, 0),
    (0.38, 0.1161, 0),
    (-0.38, -0.1161, 0),
    (-0.38, 0.1161, 0),
)

ABDUCTION_P_GAIN = 80.0
ABDUCTION_D_GAIN = 0.1
HIP_P_GAIN = 80.0
HIP_D_GAIN = 1.0
KNEE_P_GAIN = 80.0
KNEE_D_GAIN = 1.0

XR3_DEFAULT_ABDUCTION_ANGLE = 0
XR3_DEFAULT_HIP_ANGLE = -0.78
XR3_DEFAULT_KNEE_ANGLE = 1.74

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([
    XR3_DEFAULT_ABDUCTION_ANGLE,
    XR3_DEFAULT_HIP_ANGLE,
    XR3_DEFAULT_KNEE_ANGLE
] * NUM_LEGS)

_CHASSIS_NAME_PATTERN = re.compile(r"abduct_")
_MOTOR_NAME_PATTERN = re.compile(r"thigh_")
_KNEE_NAME_PATTERN = re.compile(r"knee_")
_TOE_NAME_PATTERN = re.compile(r"toe_")

# URDF_FILENAME = "/home/derek/RL/algorithm/OpenRoboRL/OpenRoboRL/robots/quadruped_robot.urdf"
URDF_FILENAME = "/home/derek/RL/algorithm/OpenRoboRL/OpenRoboRL/robots/mini_cheetah.urdf"
# URDF_FILENAME = "mini_cheetah/mini_cheetah.urdf"
# URDF_FILENAME = "laikago/laikago_toes_limits.urdf"

UPPER_BOUND = 6.28318548203
LOWER_BOUND = -6.28318548203

class XR3(pybullet_quadruped.Quadruped):
  """A simulation for the Laikago robot."""
  
  ACTION_CONFIG = [[UPPER_BOUND]*12, [LOWER_BOUND]*12]

  def __init__(self,
      pybullet_client,
      urdf_filename=URDF_FILENAME,
      time_step=0.001,
      action_repeat=33,
      control_latency=0.002,
      on_rack=False,
      enable_action_interpolation=True,
      enable_action_filter=True
  ):
    self.robot_name = "xr3"
    self._urdf_filename = urdf_filename

    sensors = [
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=NUM_MOTORS), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=environment_sensors.LastActionSensor(num_actions=NUM_MOTORS), num_history=3)
    ]

    motor_kp = [ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN]
    motor_kd = [ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN]

    motor_torque_limits = None # jp hack
    
    super(XR3, self).__init__(
        pybullet_client=pybullet_client,
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
        enable_action_filter=enable_action_filter)
    
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

  def CalIK(self, leg_id, foot_local_position):
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
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._chassis_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._knee_link_ids = []
    self._foot_link_ids = []

    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
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
    init_orientation = transformations.quaternion_from_euler(0.0, 0.0, 0.0)
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

