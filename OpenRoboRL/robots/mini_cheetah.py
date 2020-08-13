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

from robots import quadruped
from robots.sensors import environment_sensors
from robots.sensors import sensor_wrappers
from robots.sensors import robot_sensors
from robots.sensors import space_utils


URDF_FILENAME = "mini_cheetah/mini_cheetah.urdf"

T_STEP = 0.001
NUM_ACTION_REPEAT = 33
CTRL_LATENCY = 0.002
ENABLE_ENV_RANDOMIZER = False

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
INIT_EUL = [0.0, 0.0, 0.0, 1.0]
JOINT_DIRECTIONS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # to be done
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array([0.0, 0.0, 0.0] * 4)

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

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([0, -0.78, 1.74] * NUM_LEGS)

sensors = [
    sensor_wrappers.HistoricSensorWrapper(
        wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=NUM_MOTORS), num_history=3),
    sensor_wrappers.HistoricSensorWrapper(
        wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
    sensor_wrappers.HistoricSensorWrapper(
        wrapped_sensor=environment_sensors.LastActionSensor(num_actions=NUM_MOTORS), num_history=3)
]

UPPER_BOUND = 6.28318548203
LOWER_BOUND = -6.28318548203

observation_space = (
    space_utils.convert_sensors_to_gym_space_dictionary(sensors))
action_space = spaces.Box(
    np.array([LOWER_BOUND]*12),
    np.array([UPPER_BOUND]*12),
    dtype=np.float32)

motor_kp = [ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN] * NUM_LEGS
motor_kd = [ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN] * NUM_LEGS
