# coding=utf-8
# Copyright 2020 The Cloudminds Authors.
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

"""Pybullet simulation of a Mini_Cheetah robot."""
import math
import re
import numpy as np
from gym import spaces


URDF_FILENAME = "mini_cheetah/mini_cheetah.urdf"

T_STEP = 0.001
NUM_ACTION_REPEAT = 33
CTRL_LATENCY = 0.002

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "torso_to_abduct_fl_j",
    "abduct_fl_to_thigh_fl_j",
    "thigh_fl_to_knee_fl_j",
    "torso_to_abduct_hl_j",
    "abduct_hl_to_thigh_hl_j",
    "thigh_hl_to_knee_hl_j",
    "torso_to_abduct_fr_j",
    "abduct_fr_to_thigh_fr_j",
    "thigh_fr_to_knee_fr_j",
    "torso_to_abduct_hr_j",
    "abduct_hr_to_thigh_hr_j",
    "thigh_hr_to_knee_hr_j",
]
PATTERN = [re.compile(r"torso_"), re.compile(r"abduct_"),
           re.compile(r"thigh_"), re.compile(r"toe_")]

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.28]
INIT_QUAT = [0.0, 0.0, 0.0, 1.0]
JOINT_DIRECTIONS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # to be done
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array([0.0, 0.0, 0.0] * 4)

_DEFAULT_HIP_POSITIONS = (
    (0.38, -0.1161, 0),
    (0.38, 0.1161, 0),
    (-0.38, -0.1161, 0),
    (-0.38, 0.1161, 0),
)

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([0, -0.78, 1.74] * NUM_LEGS)


motor_kp = [80.0, 80.0, 80.0] * NUM_LEGS
motor_kd = [0.1, 1.0, 1.0] * NUM_LEGS


OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
