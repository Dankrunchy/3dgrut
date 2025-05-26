# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Edited by Daniel Kerezsy, 24.04.2025
# Simple Pinhole Camera based on Renderer and NeRFDataset

import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import Tensor

from threedgrut.datasets import NeRFDataset, ColmapDataset, ScannetppDataset
from threedgrut.model.model import MixtureOfGaussians

from .datasets.protocols import Batch


class BasicCamera:
    def __init__(
        self, model, conf, 
        height: int, width: int, pos: Tensor, look_at: Tensor, FOV: int
    ) -> None:

        self.model: MixtureOfGaussians = model
        self.conf = conf
        try:
            self.dataset, self.dataloader = self.create_test_dataloader(conf)
        except:
            pass

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

        # intrinsics
        self.height = height
        self.width = width
        self.fov = FOV
        
        U, V, W             = self.uvw_frame( pos, look_at )
        _x_dirs             = torch.linspace( -1.0, 1.0, self.width, device="cuda" )
        _y_dirs             = -torch.linspace( -1.0, 1.0, self.height, device="cuda" )
        d                   = torch.stack( torch.meshgrid( _y_dirs, _x_dirs, indexing='ij' ) )
        directions          = self._normalize( d[0].unsqueeze(-1) * V + d[1].unsqueeze(-1) * U + W )
        
        self.intrinsics     = [FOV, FOV, width / 2, height / 2]
        
        with torch.no_grad():
            self.rays_o_cam     = torch.zeros((1, height, width, 3), dtype=torch.float32, device="cuda")
            self.rays_o_cam[0]  = pos.cuda()[...]
            self.rays_d_cam     = directions.reshape((1, height, width, 3)).contiguous()

    @torch.no_grad
    def uvw_frame(self, position, look_at):
        """Constructs camera UVW space from data given in constructor"""
        W = (look_at - position)
        wlen = torch.sqrt( (W*W).sum() )
        # make sure that there is a target to look at (no 360° camera)
        assert wlen > 0, (position, look_at)

        # we do normalize W -- since it implies focal length and for now we want that only via the field of view
        W /= wlen
        wlen = 1.0
        # FIXME assumes that W is never exactly along z axis
        U = self._normalize( torch.linalg.cross( W, torch.tensor( [0, 0, 1], dtype=torch.float32, device="cuda" ) ) )
        V = self._normalize( torch.linalg.cross( U, W ) )

        ulen = wlen * np.tan(0.5 * np.deg2rad(self.fov))
        U *= ulen

        vlen = ulen * (self.height / self.width)  # aspect ratio
        V *= vlen

        return ( U, V, W )
    
    @staticmethod
    def _normalize(vec: Tensor):
        """small helper function that should probably be in it's own utility file"""
        return vec / torch.sqrt( (vec*vec).sum(axis=-1) ).unsqueeze(-1)
    
    def get_batch(self, ground_truth: Tensor = None) -> Batch:
        """returns 'batch' with information for renderer (camera rays)"""
        T_to_world = torch.eye(4, dtype=self.rays_o_cam.dtype, device=self.rays_o_cam.device)[None]
        sample = {
            "rgb_gt":       ground_truth,
            "rays_ori":     self.rays_o_cam,
            "rays_dir":     self.rays_d_cam,
            "T_to_world":   T_to_world,
            "intrinsics":   self.intrinsics,
        }

        return Batch(**sample)
        
    def render(self, ground_truth: Tensor = None) -> Tensor:
        """renders single image attached to autograd tree"""
        gpu_batch = self.get_batch( 
            ground_truth.reshape(1, *ground_truth.shape) if ground_truth is not None else None
        )
        
        self.model.build_acc(True)
        outputs = self.model(gpu_batch, train=True, frame_id=1000)
        
        return outputs["pred_rgb"].squeeze()
