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

"""
This should be identical to sim_config.proto.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr
import typing

@attr.s
class SimulationParameters(object):
  """Parameters specific for the pyBullet simulation."""
  enable_hard_reset = attr.ib(type=bool, default=False)
  enable_rendering = attr.ib(type=bool, default=False)
  enable_rendering_gui = attr.ib(type=bool, default=True)
  camera_distance = attr.ib(type=float, default=1.0)
  camera_yaw = attr.ib(type=float, default=0)
  camera_pitch = attr.ib(type=float, default=-30)
  render_width = attr.ib(type=int, default=480)
  render_height = attr.ib(type=int, default=360)
  egl_rendering = attr.ib(type=bool, default=False)

  draw_ref_model_alpha = attr.ib(type=float, default=0.5)

