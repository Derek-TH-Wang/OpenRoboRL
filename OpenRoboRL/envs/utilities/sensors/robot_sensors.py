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

"""Simple sensors related to the robot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import typing

from envs.utilities.sensors import sensor

_ARRAY = typing.Iterable[float]
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY]
_DATATYPE_LIST = typing.Iterable[typing.Any]


class MotorAngleSensor(sensor.BoxSpaceSensor):
  """A sensor that reads motor angles from the robot."""

  def __init__(self,
               num_motors: int,
               noisy_reading: bool = True,
               observe_sine_cosine: bool = False,
               lower_bound: _FLOAT_OR_ARRAY = -np.pi,
               upper_bound: _FLOAT_OR_ARRAY = np.pi,
               name: typing.Text = "MotorAngle",
               dtype: typing.Type[typing.Any] = np.float64) -> None:
    """Constructs MotorAngleSensor.

    Args:
      num_motors: the number of motors in the robot
      noisy_reading: whether values are true observations
      observe_sine_cosine: whether to convert readings to sine/cosine values for
        continuity
      lower_bound: the lower bound of the motor angle
      upper_bound: the upper bound of the motor angle
      name: the name of the sensor
      dtype: data type of sensor value
    """
    self._num_motors = num_motors
    self._noisy_reading = noisy_reading
    self._observe_sine_cosine = observe_sine_cosine

    if observe_sine_cosine:
      super(MotorAngleSensor, self).__init__(
          name=name,
          shape=(self._num_motors * 2,),
          lower_bound=-np.ones(self._num_motors * 2),
          upper_bound=np.ones(self._num_motors * 2),
          dtype=dtype)
    else:
      super(MotorAngleSensor, self).__init__(
          name=name,
          shape=(self._num_motors,),
          lower_bound=lower_bound,
          upper_bound=upper_bound,
          dtype=dtype)

  def _get_observation(self) -> _ARRAY:
    if self._noisy_reading:
      motor_angles = self._robot.get_motor_angles()
    else:
      motor_angles = self._robot.get_true_motor_angles()

    if self._observe_sine_cosine:
      return np.hstack((np.cos(motor_angles), np.sin(motor_angles)))
    else:
      return motor_angles

class IMUSensor(sensor.BoxSpaceSensor):
  """An IMU sensor that reads orientations and angular velocities."""

  def __init__(self,
               channels: typing.Iterable[typing.Text] = None,
               noisy_reading: bool = True,
               lower_bound: _FLOAT_OR_ARRAY = None,
               upper_bound: _FLOAT_OR_ARRAY = None,
               name: typing.Text = "IMU",
               dtype: typing.Type[typing.Any] = np.float64) -> None:
    """Constructs IMUSensor.

    It generates separate IMU value channels, e.g. IMU_R, IMU_P, IMU_dR, ...

    Args:
      channels: value channels wants to subscribe. A upper letter represents
        orientation and a lower letter represents angular velocity. (e.g. ['R',
        'P', 'Y', 'dR', 'dP', 'dY'] or ['R', 'P', 'dR', 'dP'])
      noisy_reading: whether values are true observations
      lower_bound: the lower bound IMU values
        (default: [-2pi, -2pi, -2000pi, -2000pi])
      upper_bound: the lower bound IMU values
        (default: [2pi, 2pi, 2000pi, 2000pi])
      name: the name of the sensor
      dtype: data type of sensor value
    """
    self._channels = channels if channels else ["R", "P", "dR", "dP"]
    self._num_channels = len(self._channels)
    self._noisy_reading = noisy_reading

    # Compute the default lower and upper bounds
    if lower_bound is None and upper_bound is None:
      lower_bound = []
      upper_bound = []
      for channel in self._channels:
        if channel in ["R", "P", "Y"]:
          lower_bound.append(-2.0 * np.pi)
          upper_bound.append(2.0 * np.pi)
        elif channel in ["Rcos", "Rsin", "Pcos", "Psin", "Ycos", "Ysin"]:
          lower_bound.append(-1.)
          upper_bound.append(1.)
        elif channel in ["dR", "dP", "dY"]:
          lower_bound.append(-2000.0 * np.pi)
          upper_bound.append(2000.0 * np.pi)

    super(IMUSensor, self).__init__(
        name=name,
        shape=(self._num_channels,),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        dtype=dtype)

    # Compute the observation_datatype
    datatype = [("{}_{}".format(name, channel), self._dtype)
                for channel in self._channels]

    self._datatype = datatype

  def get_channels(self) -> typing.Iterable[typing.Text]:
    return self._channels

  def get_num_channels(self) -> int:
    return self._num_channels

  def get_observation_datatype(self) -> _DATATYPE_LIST:
    """Returns box-shape data type."""
    return self._datatype

  def _get_observation(self) -> _ARRAY:
    if self._noisy_reading:
      rpy = self._robot.get_base_rpy()
      drpy = self._robot.get_base_rpy_rate()
    else:
      rpy = self._robot.get_true_base_rpy()
      drpy = self._robot.get_true_base_rpy_rate()

    assert len(rpy) >= 3, rpy
    assert len(drpy) >= 3, drpy

    observations = np.zeros(self._num_channels)
    for i, channel in enumerate(self._channels):
      if channel == "R":
        observations[i] = rpy[0]
      if channel == "Rcos":
        observations[i] = np.cos(rpy[0])
      if channel == "Rsin":
        observations[i] = np.sin(rpy[0])
      if channel == "P":
        observations[i] = rpy[1]
      if channel == "Pcos":
        observations[i] = np.cos(rpy[1])
      if channel == "Psin":
        observations[i] = np.sin(rpy[1])
      if channel == "Y":
        observations[i] = rpy[2]
      if channel == "Ycos":
        observations[i] = np.cos(rpy[2])
      if channel == "Ysin":
        observations[i] = np.sin(rpy[2])
      if channel == "dR":
        observations[i] = drpy[0]
      if channel == "dP":
        observations[i] = drpy[1]
      if channel == "dY":
        observations[i] = drpy[2]
    return observations
