import math
import os
import re
import numpy as np
import pybullet as pyb

from robots import robot_motor
from robots import quadruped
from envs import locomotion_gym_config

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

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
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

# URDF_FILENAME = "/home/derek/RL/algorithm/motion_imitation/motion_imitation/robots/quadruped_robot.urdf"
URDF_FILENAME = "/home/derek/RL/algorithm/motion_imitation/motion_imitation/robots/mini_cheetah.urdf"
# URDF_FILENAME = "mini_cheetah/mini_cheetah.urdf"
# URDF_FILENAME = "laikago/laikago_toes_limits.urdf"
_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

UPPER_BOUND = 6.28318548203
LOWER_BOUND = -6.28318548203

class xr3(quadruped.Quadruped):
  """A simulation for the Laikago robot."""
  
  ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(name="motor_angle_0", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_1", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_2", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_3", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_4", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_5", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_6", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_7", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_8", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_9", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_10", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_11", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND)
  ]

  def __init__(self,
      pybullet_client,
      urdf_filename=URDF_FILENAME,
      enable_clip_motor_commands=True,
      time_step=0.001,
      action_repeat=33,
      sensors=None,
      control_latency=0.002,
      on_rack=False,
      enable_action_interpolation=True,
      enable_action_filter=True
  ):
    self.robot_name = "xr3"
    self._urdf_filename = urdf_filename

    self._enable_clip_motor_commands = enable_clip_motor_commands

    motor_kp = [ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN]
    motor_kd = [ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN]

    motor_torque_limits = None # jp hack
    
    super(xr3, self).__init__(
        pybullet_client=pybullet_client,
        time_step=time_step,
        action_repeat=action_repeat,
        num_motors=NUM_MOTORS,
        dofs_per_leg=DOFS_PER_LEG,
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

  def _LoadRobotURDF(self):
    print("-----------------_LoadRobotURDF")
    laikago_urdf_path = self.GetURDFFile()
    if self._self_collision_enabled:
      self.quadruped = self._pybullet_client.loadURDF(
          laikago_urdf_path,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation(),
          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
    else:
      self.quadruped = self._pybullet_client.loadURDF(
          laikago_urdf_path, self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation())
    

  def _SettleDownForReset(self, default_motor_angles, reset_time):
    self.ReceiveObservation()

    if reset_time <= 0:
      return

    for _ in range(500):
      self._StepInternal(
          INIT_MOTOR_ANGLES,
          motor_control_mode=robot_motor.POSITION)
    if default_motor_angles is not None:
      num_steps_to_reset = int(reset_time / self.time_step)
      for _ in range(num_steps_to_reset):
        self._StepInternal(
            default_motor_angles,
            motor_control_mode=robot_motor.POSITION)

  def GetHipPositionsInBaseFrame(self):
    return _DEFAULT_HIP_POSITIONS

  def GetFootContacts(self):
    all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

    contacts = [False, False, False, False]
    for contact in all_contacts:
      # Ignore self contacts
      if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
        continue
      try:
        toe_link_index = self._foot_link_ids.index(
            contact[_LINK_A_FIELD_NUMBER])
        contacts[toe_link_index] = True
      except ValueError:
        continue

    return contacts

  def ComputeJacobian(self, leg_id):
    """Compute the Jacobian for a given leg."""
    # Because of the default rotation in the Laikago URDF, we need to reorder
    # the rows in the Jacobian matrix.
    return super(xr3, self).ComputeJacobian(leg_id)[(2, 0, 1), :]

  def ResetPose(self, add_constraint):
    del add_constraint
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)
    for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
      if "abduct" in name:
        angle = INIT_MOTOR_ANGLES[i] + HIP_JOINT_OFFSET
      elif "thigh" in name:
        angle = INIT_MOTOR_ANGLES[i] + UPPER_LEG_JOINT_OFFSET
      elif "knee" in name:
        angle = INIT_MOTOR_ANGLES[i] + KNEE_JOINT_OFFSET
      else:
        raise ValueError("The name %s is not recognized as a motor joint." %
                         name)
      self._pybullet_client.resetJointState(
          self.quadruped, self._joint_name_to_id[name], angle, targetVelocity=0)

  def GetURDFFile(self):
    return self._urdf_filename

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

  def _GetMotorNames(self):
    return MOTOR_NAMES

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
    init_orientation = pyb.getQuaternionFromEuler([0.0, 0, 0.0])
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

  def ApplyAction(self, motor_commands, motor_control_mode=None):
    """Clips and then apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).N
      motor_control_mode: A MotorControlMode enum.
    """
    if self._enable_clip_motor_commands:
      motor_commands = self._ClipMotorCommands(motor_commands)

    super(xr3, self).ApplyAction(motor_commands, motor_control_mode)
    return

  def _ClipMotorCommands(self, motor_commands):
    """Clips motor commands.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).

    Returns:
      Clipped motor commands.
    """

    # clamp the motor command by the joint limit, in case weired things happens
    max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
    current_motor_angles = self.GetMotorAngles()
    motor_commands = np.clip(motor_commands,
                             current_motor_angles - max_angle_change,
                             current_motor_angles + max_angle_change)
    return motor_commands

