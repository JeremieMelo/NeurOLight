"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-26 00:36:10
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-02 19:24:43
"""

from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import torch
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn
from torch.types import Device

from .layers.fno_conv2d import FNOConv2d

__all__ = ["PDE_NN_BASE"]


class PDE_NN_BASE(nn.Module):
    _conv = (FNOConv2d,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """
        The overall network.
        1. Lift the input to the desire channel dimension.
        2. integral operators u' = f(u).
        3. Project from the channel space to the output space.
        """

    def reset_parameters(self, random_state: Optional[int] = None):
        for name, m in self.named_modules():
            if isinstance(m, self._conv):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()

    @lru_cache(maxsize=8)
    def _get_linear_pos_enc(self, shape, device) -> Tensor:
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.arange(0, size_x, device=device)
        gridy = torch.arange(0, size_y, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
        return mesh

    def get_grid(self, shape, device: Device, mode: str = "linear", epsilon=None, wavelength=None, grid_step=None):
        # epsilon must be real permittivity without normalization
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        if mode == "linear":
            gridx = torch.linspace(0, 1, size_x, device=device)
            gridy = torch.linspace(0, 1, size_y, device=device)
            gridx, gridy = torch.meshgrid(gridx, gridy)
            return torch.stack([gridy, gridx], dim=0).unsqueeze(0).expand(batchsize, -1, -1, -1)
        elif mode in {"exp", "exp_noeps"}:  # exp in the complex domain

            mesh = self._get_linear_pos_enc(shape, device)
            # mesh [1 ,2 ,h, w] real
            # grid_step [bs, 2, 1, 1] real
            # wavelength [bs, 1, 1, 1] real
            # epsilon [bs, 1, h, w] complex
            mesh = torch.view_as_real(
                torch.exp(
                    mesh.mul(grid_step.div(wavelength).mul(1j * 2 * np.pi)[..., None, None]).mul(epsilon.data.sqrt())
                )
            )  # [bs, 2, h, w, 2] real
            return mesh.permute(0, 1, 4, 2, 3).flatten(1, 2)
        elif mode == "exp3":  # exp in the complex domain
            gridx = torch.arange(0, size_x, device=device)
            gridy = torch.arange(0, size_y, device=device)
            gridx, gridy = torch.meshgrid(gridx, gridy)
            mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
            mesh = torch.exp(
                mesh.mul(grid_step[..., None, None]).mul(
                    1j * 2 * np.pi / wavelength[..., None, None] * epsilon.data.sqrt()
                )
            )  # [bs, 2, h, w] complex
            mesh = torch.view_as_real(torch.cat([mesh, mesh[:, 0:1].add(mesh[:, 1:])], dim=1))  # [bs, 3, h, w, 2] real
            return mesh.permute(0, 1, 4, 2, 3).flatten(1, 2)
        elif mode == "exp4":  # exp in the complex domain
            gridx = torch.arange(0, size_x, device=device)
            gridy = torch.arange(0, size_y, device=device)
            gridx, gridy = torch.meshgrid(gridx, gridy)
            mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
            mesh = torch.exp(
                mesh.mul(grid_step[..., None, None]).mul(
                    1j * 2 * np.pi / wavelength[..., None, None] * epsilon.data.sqrt()
                )
            )  # [bs, 2, h, w] complex
            mesh = torch.view_as_real(
                torch.cat([mesh, mesh[:, 0:1].mul(mesh[:, 1:]), mesh[:, 0:1].div(mesh[:, 1:])], dim=1)
            )  # [bs, 4, h, w, 2] real
            return mesh.permute(0, 1, 4, 2, 3).flatten(1, 2)
        elif mode == "exp_full":
            gridx = torch.arange(0, size_x, device=device)
            gridy = torch.arange(0, size_y, device=device)
            gridx, gridy = torch.meshgrid(gridx, gridy)
            mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
            mesh = torch.exp(
                mesh.mul(grid_step[..., None, None]).mul(
                    1j * 2 * np.pi / wavelength[..., None, None] * epsilon.data.sqrt()
                )
            )  # [bs, 2, h, w] complex
            mesh = (
                torch.view_as_real(mesh).permute(0, 1, 4, 2, 3).flatten(1, 2)
            )  # [bs, 2, h, w, 2] real -> [bs, 4, h, w] real
            wavelength_map = wavelength[..., None, None].expand(
                mesh.shape[0], 1, mesh.shape[2], mesh.shape[3]
            )  # [bs, 1, h, w] real
            grid_step_mesh = (
                grid_step[..., None, None].expand(mesh.shape[0], 2, mesh.shape[2], mesh.shape[3]) * 10
            )  # 0.05 um -> 0.5 for statistical stability # [bs, 2, h, w] real
            return torch.cat([mesh, wavelength_map, grid_step_mesh], dim=1)  # [bs, 7, h, w] real
        elif mode == "exp_full_r":
            gridx = torch.arange(0, size_x, device=device)
            gridy = torch.arange(0, size_y, device=device)
            gridx, gridy = torch.meshgrid(gridx, gridy)
            mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
            mesh = torch.exp(
                mesh.mul(grid_step[..., None, None]).mul(
                    1j * 2 * np.pi / wavelength[..., None, None] * epsilon.data.sqrt()
                )
            )  # [bs, 2, h, w] complex
            mesh = (
                torch.view_as_real(mesh).permute(0, 1, 4, 2, 3).flatten(1, 2)
            )  # [bs, 2, h, w, 2] real -> [bs, 4, h, w] real
            wavelength_map = (1 / wavelength)[..., None, None].expand(
                mesh.shape[0], 1, mesh.shape[2], mesh.shape[3]
            )  # [bs, 1, h, w] real
            grid_step_mesh = (
                grid_step[..., None, None].expand(mesh.shape[0], 2, mesh.shape[2], mesh.shape[3]) * 10
            )  # 0.05 um -> 0.5 for statistical stability # [bs, 2, h, w] real
            return torch.cat([mesh, wavelength_map, grid_step_mesh], dim=1)  # [bs, 7, h, w] real
        elif mode == "raw":
            wavelength_map = wavelength[..., None, None].expand(batchsize, 1, size_x, size_y)  # [bs, 1, h, w] real
            grid_step_mesh = (
                grid_step[..., None, None].expand(batchsize, 2, size_x, size_y) * 10
            )  # 0.05 um -> 0.5 for statistical stability # [bs, 2, h, w] real
            return torch.cat([wavelength_map, grid_step_mesh], dim=1)  # [bs, 3, h, w] real

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def reset_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def set_linear_probing_mode(self, mode: bool = True):
        self.linear_probing_mode = mode
