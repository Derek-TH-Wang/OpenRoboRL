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

"""This file implements the interface of the pybullet simulator."""
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd

class PybulletInterface(object):
    def init_pybullet(self, is_render):
        # if is_render:
        #     self.pybullet_client = bullet_client.BulletClient(
        #         connection_mode=pybullet.GUI)
        #         pybullet.configureDebugVisualizer(
        #         pybullet.COV_ENABLE_GUI,
        #         sim_config.enable_rendering_gui)
        #     self._show_reference_id = pybullet.addUserDebugParameter("show reference",0,1,
        #         self._task._draw_ref_model_alpha)
        #     self._delay_id = pybullet.addUserDebugParameter("delay",0,0.3,0)
        # else:
        #     self.pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        #     self.pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        # if sim_config.egl_rendering:
        #     self.pybullet_client.loadPlugin('eglRendererPlugin')
        return