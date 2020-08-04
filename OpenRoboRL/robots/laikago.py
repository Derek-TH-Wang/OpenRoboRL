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
from utils import transformations

from robots import robot_motor
from sim import pybullet_quadruped
from robots.sensors import environment_sensors
from robots.sensors import sensor_wrappers
from robots.sensors import robot_sensors


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

class Laikago(pybullet_quadruped.Quadruped):
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
    
    super(Laikago, self).__init__(
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
          self._cal_link_pos(
              robot=self,
              link_id=foot_id,
          ))
    return np.array(foot_positions)

  def CalJacobian(self, leg_id):
    """Compute the Jacobian for a given leg."""
    # Does not work for Minitaur which has the four bar mechanism for now.
    assert len(self._foot_link_ids) == self.num_legs
    return self._cal_jacobian(
        robot=self,
        link_id=self._foot_link_ids[leg_id],
    )

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
    assert len(self._foot_link_ids) == self.num_legs
    toe_id = self._foot_link_ids[leg_id]

    motors_per_leg = self.num_motors // self.num_legs
    joint_position_idxs = [
        i for i in range(leg_id * motors_per_leg, leg_id * motors_per_leg +
                         motors_per_leg)
    ]

    joint_angles = self._ik(
        robot=self,
        link_position=foot_local_position,
        link_id=toe_id,
        joint_ids=joint_position_idxs,
    )

    # Joint offset is necessary for Laikago.
    joint_angles = np.multiply(
        np.asarray(joint_angles) -
        np.asarray(self._motor_offset)[joint_position_idxs],
        self._motor_direction[joint_position_idxs])

    # Return the joing index (the same as when calling GetMotorAngles) as well
    # as the angles.
    return joint_position_idxs, joint_angles.tolist()

  def _ik(
      self,
      robot: typing.Any,
      link_position: typing.Sequence[float],
      link_id: int,
      joint_ids: typing.Sequence[int],
      base_translation: typing.Sequence[float] = (0, 0, 0),
      base_rotation: typing.Sequence[float] = (0, 0, 0, 1)):
    """Uses Inverse Kinematics to calculate joint angles.

    Args:
      robot: A robot instance.
      link_position: The (x, y, z) of the link in the body frame. This local frame
        is transformed relative to the COM frame using a given translation and
        rotation.
      link_id: The link id as returned from loadURDF.
      joint_ids: The positional index of the joints. This can be different from
        the joint unique ids.
      base_translation: Additional base translation.
      base_rotation: Additional base rotation.

    Returns:
      A list of joint angles.
    """
    # Projects to local frame.
    base_position, base_orientation = robot.GetBasePosition(
    ), robot.GetBaseOrientation()
    base_position, base_orientation = robot.pybullet_client.multiplyTransforms(
        base_position, base_orientation, base_translation, base_rotation)

    # Projects to world space.
    world_link_pos, _ = robot.pybullet_client.multiplyTransforms(
        base_position, base_orientation, link_position, (0, 0, 0, 1))
    ik_solver = 0
    all_joint_angles = robot.pybullet_client.calculateInverseKinematics(
        robot.quadruped, link_id, world_link_pos, solver=ik_solver)

    # Extract the relevant joint angles.
    joint_angles = [all_joint_angles[i] for i in joint_ids]
    return joint_angles


  def _cal_link_pos(
      self, 
      robot: typing.Any,
      link_id: int,
  ):
    """Computes the link's local position in the robot frame.

    Args:
      robot: A robot instance.
      link_id: The link to calculate its relative position.

    Returns:
      The relative position of the link.
    """
    base_position, base_orientation = robot.GetBasePosition(
    ), robot.GetBaseOrientation()
    inverse_translation, inverse_rotation = robot.pybullet_client.invertTransform(
        base_position, base_orientation)

    link_state = robot.pybullet_client.getLinkState(robot.quadruped, link_id)
    link_position = link_state[0]
    link_local_position, _ = robot.pybullet_client.multiplyTransforms(
        inverse_translation, inverse_rotation, link_position, (0, 0, 0, 1))

    return np.array(link_local_position)

  def _cal_jacobian(
      self,
      robot: typing.Any,
      link_id: int,
  ):
    """Computes the Jacobian matrix for the given link.

    Args:
      robot: A robot instance.
      link_id: The link id as returned from loadURDF.

    Returns:
      The 3 x N transposed Jacobian matrix. where N is the total DoFs of the
      robot. For a quadruped, the first 6 columns of the matrix corresponds to
      the CoM translation and rotation. The columns corresponds to a leg can be
      extracted with indices [6 + leg_id * 3: 6 + leg_id * 3 + 3].
    """

    all_joint_angles = [state[0] for state in robot.joint_states]
    zero_vec = [0] * len(all_joint_angles)
    jv, _ = robot.pybullet_client.calculateJacobian(robot.quadruped, link_id,
                                                    (0, 0, 0), all_joint_angles,
                                                    zero_vec, zero_vec)
    jacobian = np.array(jv)
    assert jacobian.shape[0] == 3
    return jacobian

  def CalJacobian(self, leg_id):
    """Compute the Jacobian for a given leg."""
    # Does not work for Minitaur which has the four bar mechanism for now.
    assert len(self._foot_link_ids) == self.num_legs
    return self._cal_jacobian(
        robot=self,
        link_id=self._foot_link_ids[leg_id],
    )[(2, 0, 1), :]

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
