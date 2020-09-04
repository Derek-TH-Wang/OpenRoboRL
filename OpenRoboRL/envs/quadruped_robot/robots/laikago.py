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
import re
import numpy as np
from gym import spaces



URDF_FILENAME = "laikago/laikago_toes_limits.urdf"

T_STEP = 0.001
NUM_ACTION_REPEAT = 33
CTRL_LATENCY = 0.002
ENABLE_ENV_RANDOMIZER = True

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
PATTERN = [re.compile(r"\w+_chassis_\w+"), re.compile(r"\w+_hip_motor_\w+"),
           re.compile(r"\w+_lower_leg_\w+"), re.compile(r"jtoe\d*")]

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.48]
INIT_QUAT = [0.5, 0.5, 0.5, 0.5]
JOINT_DIRECTIONS = np.array([-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1])
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array([0.0, -0.6, 0.66] * 4)

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

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([0, 0.67, -1.25] * NUM_LEGS)


UPPER_BOUND = 6.28318548203
LOWER_BOUND = -6.28318548203

action_space = spaces.Box(
    np.array([LOWER_BOUND]*12),
    np.array([UPPER_BOUND]*12),
    dtype=np.float32)

motor_kp = [ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN] * NUM_LEGS
motor_kd = [ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN] * NUM_LEGS


OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2