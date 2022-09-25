"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-10 01:25:27
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-10 01:37:06
"""


from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.functional import Tensor
from torch.types import Device

from .constant import *
from .layers import PermittivityEncoder
from .pde_base import PDE_NN_BASE

__all__ = ["UNet"]


def double_conv(in_channels, hidden, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
    )


class UNet(PDE_NN_BASE):
    """
    Frequency-domain scattered electric field envelop predictor
    Assumption:
    (1) TE10 mode, i.e., Ey(r, omega) = Ez(r, omega) = 0
    (2) Fixed wavelength. wavelength currently not being modeled
    (3) Only predict Ex_scatter(r, omega)

    Args:
        PDE_NN_BASE ([type]): [description]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dim: int = 16,
        act_func: Optional[str] = "GELU",
        domain_size: Tuple[float] = [20, 100],  # computation domain in unit of um
        grid_step: float = 1.550 / 20,  # grid step size in unit of um, typically 1/20 or 1/30 of the wavelength
        pml_width: float = 0,
        pml_permittivity: complex = 0 + 0j,
        buffer_width: float = 0.5,
        buffer_permittivity: complex = -1e-10 + 0j,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        eps_min: float = 2.085136,
        eps_max: float = 12.3,
        aux_head: bool = False,
        aux_head_idx: int = 1,
        pos_encoding: str = "exp",
        with_cp=False,
    ):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % 2 == 0, f"The output channels must be even number larger than 2, but got {out_channels}"
        self.dim = dim
        self.act_func = act_func
        self.domain_size = domain_size
        self.grid_step = grid_step
        self.domain_size_pixel = [round(i / grid_step) for i in domain_size]
        self.buffer_width = buffer_width
        self.buffer_permittivity = buffer_permittivity
        self.pml_width = pml_width
        self.pml_permittivity = pml_permittivity
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.aux_head = aux_head
        self.aux_head_idx = aux_head_idx
        self.pos_encoding = pos_encoding
        self.with_cp = with_cp
        if pos_encoding == "none":
            pass
        elif pos_encoding == "linear":
            self.in_channels += 2
        elif pos_encoding == "exp":
            self.in_channels += 4
        elif pos_encoding == "exp3":
            self.in_channels += 6
        elif pos_encoding == "exp4":
            self.in_channels += 8
        elif pos_encoding in {"exp_full", "exp_full_r"}:
            self.in_channels += 7
        else:
            raise ValueError(f"pos_encoding only supports linear and exp, but got {pos_encoding}")

        self.device = device

        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()
        self.reset_parameters()

        self.permittivity_encoder = None
        self.set_trainable_permittivity(False)

    def build_layers(self):
        dim = self.dim
        self.dconv_down1 = double_conv(self.in_channels, dim, dim)
        self.dconv_down2 = double_conv(dim, dim * 2, dim * 2)
        self.dconv_down3 = double_conv(dim * 2, dim * 4, dim * 4)
        self.dconv_down4 = double_conv(dim * 4, dim * 8, dim * 8)

        # self.maxpool = nn.MaxPool2d(2)
        self.maxpool = nn.AvgPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample1 = nn.ConvTranspose2d(dim * 8, dim * 8, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(dim * 4, dim * 4, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(dim * 2, dim * 2, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(dim * 12, dim * 8, dim * 4)
        self.dconv_up2 = double_conv(dim * 6, dim * 4, dim * 2)
        self.dconv_up1 = double_conv(dim * 3, dim * 2, dim)
        self.drop_out = nn.Dropout2d(self.dropout_rate)
        self.conv_last = nn.Conv2d(dim, self.out_channels, 1)

    def set_trainable_permittivity(self, mode: bool = True) -> None:
        self.trainable_permittivity = mode

    def init_trainable_permittivity(
        self,
        regions: Tensor,
        valid_range: Tuple[int],
    ):
        self.permittivity_encoder = PermittivityEncoder(
            size=self.domain_size_pixel,
            regions=regions,
            valid_range=valid_range,
            device=self.device,
        )

    def requires_network_params_grad(self, mode: float = True) -> None:
        params = self.parameters()
        for p in params:
            p.requires_grad_(mode)

    def forward(self, x: Tensor, wavelength: Tensor, grid_step: Tensor) -> Tensor:
        # x [bs, inc, h, w] complex
        # wavelength [bs, 1] real
        # grid_step [bs, 2] real
        epsilon = x[:, 0:1] * (self.eps_max - self.eps_min) + self.eps_min  # this is de-normalized permittivity

        # convert complex permittivity/mode to real numbers
        x = torch.view_as_real(x).permute(0, 1, 4, 2, 3).flatten(1, 2)  # [bs, inc*2, h, w] real

        # positional encoding
        grid = self.get_grid(
            x.shape,
            x.device,
            mode=self.pos_encoding,
            epsilon=epsilon,
            wavelength=wavelength,
            grid_step=grid_step,
        )  # [bs, 2 or 4 or 8, h, w] real
        if grid is not None:
            x = torch.cat((x, grid), dim=1)  # [bs, inc*2+4, h, w] real

        # DNN-based electric field envelop prediction
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)  # [bs, outc, h, w] real
        # convert to complex frequency-domain electric field envelops
        x = torch.view_as_complex(
            x.view(x.size(0), -1, 2, x.size(-2), x.size(-1)).permute(0, 1, 3, 4, 2).contiguous()
        )  # [bs, outc/2, h, w] complex

        return x
