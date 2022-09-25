"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-26 19:57:31
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-01-15 15:49:13
"""

import operator
from functools import reduce
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torch.types import Device


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    return c


class ComplexMSELoss(torch.nn.MSELoss):
    def __init__(self, norm=False) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """Complex MSE between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        # factors = torch.linspace(0.1, 1, x.size(-1), device=x.device).view(1,1,1,-1)
        # return F.mse_loss(torch.view_as_real(x.mul(factors)), torch.view_as_real(target.mul(factors)))
        if self.norm:
            diff = torch.view_as_real(x - target)
            return (
                diff.square()
                .sum(dim=[1, 2, 3, 4])
                .div(torch.view_as_real(target).square().sum(dim=[1, 2, 3, 4]))
                .mean()
            )
        return F.mse_loss(torch.view_as_real(x), torch.view_as_real(target))


class ComplexL1Loss(torch.nn.MSELoss):
    def __init__(self, norm=False) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """Complex L1 loss between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        if self.norm:
            diff = torch.view_as_real(x - target)
            return diff.norm(p=1, dim=[1, 2, 3, 4]).div(torch.view_as_real(target).norm(p=1, dim=[1, 2, 3, 4])).mean()
        return F.l1_loss(torch.view_as_real(x), torch.view_as_real(target))


class ComplexTVLoss(torch.nn.MSELoss):
    def __init__(self, norm=False) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """TV loss between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        target_deriv_x = torch.view_as_real(target[..., 1:, :] - target[..., :-1, :])
        target_deriv_y = torch.view_as_real(target[..., 1:] - target[..., :-1])
        pred_deriv_x = torch.view_as_real(x[..., 1:, :] - x[..., :-1, :])
        pred_deriv_y = torch.view_as_real(x[..., 1:] - x[..., :-1])
        if self.norm:
            return (
                (pred_deriv_x - target_deriv_x)
                .square()
                .sum(dim=[1, 2, 3, 4])
                .add((pred_deriv_y - target_deriv_y).square().sum(dim=[1, 2, 3, 4]))
                .div(target_deriv_x.square().sum(dim=[1, 2, 3, 4]).add(target_deriv_y.square().sum(dim=[1, 2, 3, 4])))
                .mean()
            )
        return F.mse_loss(pred_deriv_x, target_deriv_x) + F.mse_loss(pred_deriv_y, target_deriv_y)


def normalize(x):
    if isinstance(x, np.ndarray):
        x_min, x_max = np.percentile(x, 5), np.percentile(x, 95)
        x = np.clip(x, x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    else:
        x_min, x_max = torch.quantile(x, 0.05), torch.quantile(x, 0.95)
        x = x.clamp(x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    return x


def plot_compare(
    wavelength: Tensor,
    grid_step: Tensor,
    epsilon: Tensor,
    pred_fields: Tensor,
    target_fields: Tensor,
    filepath: str,
    pol: str = "Hz",
    norm: bool = True,
) -> None:
    if epsilon.is_complex():
        epsilon = epsilon.real
    # simulation = Simulation(omega, eps_r=epsilon.data.cpu().numpy(), dl=grid_step, NPML=[1,1], pol='Hz')
    field_val = pred_fields.data.cpu().numpy()
    target_field_val = target_fields.data.cpu().numpy()
    # eps_r = simulation.eps_r
    eps_r = epsilon.data.cpu().numpy()
    err_field_val = field_val - target_field_val
    # field_val = np.abs(field_val)
    field_val = field_val.real
    # target_field_val = np.abs(target_field_val)
    target_field_val = target_field_val.real
    err_field_val = np.abs(err_field_val)
    outline_val = np.abs(eps_r)

    # vmax = field_val.max()
    vmin = 0.0
    b = field_val.shape[0]
    fig, axes = plt.subplots(3, b, constrained_layout=True, figsize=(5 * b, 3.1))
    if b == 1:
        axes = axes[..., np.newaxis]
    cmap = "magma"
    # print(field_val.shape, target_field_val.shape, outline_val.shape)
    for i in range(b):
        vmax = np.max(target_field_val[i])
        if norm:
            h1 = axes[0, i].imshow(normalize(field_val[i]), cmap=cmap, origin="lower")
            h2 = axes[1, i].imshow(normalize(target_field_val[i]), cmap=cmap, origin="lower")
        else:
            h1 = axes[0, i].imshow(field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")
            h2 = axes[1, i].imshow(target_field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")
        h3 = axes[2, i].imshow(err_field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")
        for j in range(3):
            divider = make_axes_locatable(axes[j, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            fig.colorbar([h1, h2, h3][j], label=pol, ax=axes[j, i], cax=cax)
        axes[0, i].title.set_text(
            f"{wavelength[i,0].item():.2f} um, dh=({grid_step[i,0].item()*1000:.1f} nm x {grid_step[i,1].item()*1000:.1f} nm)"
        )
        # fig.colorbar(h2, label=pol, ax=axes[1,i])
        # fig.colorbar(h3, label=pol, ax=axes[2,i])

    # Do black and white so we can see on both magma and RdBu
    for ax in axes.flatten():
        ax.contour(outline_val[0], levels=2, linewidths=1.0, colors="w")
        ax.contour(outline_val[0], levels=2, linewidths=0.5, colors="k")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_dynamics(
    wavelength: Tensor,
    grid_step: Tensor,
    epsilon: Tensor,
    pred_fields: Tensor,
    target_fields: Tensor,
    filepath: str,
    eps_list,
    eps_text_loc_list,
    region_list,
    box_id,
    ref_eps,
    norm: bool = True,
    wl_text_pos=None,
    time=None,
    fps=None,
) -> None:
    if epsilon.is_complex():
        epsilon = epsilon.real
        ref_eps = ref_eps.real
    # simulation = Simulation(omega, eps_r=epsilon.data.cpu().numpy(), dl=grid_step, NPML=[1,1], pol='Hz')
    field_val = pred_fields.data.cpu().numpy()
    target_field_val = target_fields.data.cpu().numpy()
    # eps_r = simulation.eps_r
    eps_r = epsilon.data.cpu().numpy()
    err_field_val = field_val - target_field_val
    # field_val = np.abs(field_val)
    field_val = field_val.real
    # target_field_val = np.abs(target_field_val)
    target_field_val = target_field_val.real
    err_field_val = np.abs(err_field_val)
    outline_val = np.abs(ref_eps.data.cpu().numpy())

    # vmax = field_val.max()
    vmin = 0.0
    b = field_val.shape[0]
    fig, ax = plt.subplots(1, b, constrained_layout=True, figsize=(5.1, 1.15))
    # fig, ax = plt.subplots(1, b, constrained_layout=True, figsize=(5.2, 1))
    cmap = "magma"
    # print(field_val.shape, target_field_val.shape, outline_val.shape)
    i = 0
    vmax = np.max(target_field_val[i])
    if norm:
        h1 = ax.imshow(normalize(field_val[i]), cmap=cmap, origin="lower")
    else:
        h1 = ax.imshow(field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h1, label="Mag.", ax=ax, cax=cax)
    for i, (eps, eps_pos, box) in enumerate(zip(eps_list, eps_text_loc_list, region_list)):
        xl, xh, yl, yh = box
        if i == box_id:
            color = "yellow"
        else:
            color = "white"
        ax.annotate(r"$\epsilon_r$" + f" = {eps:.3f}", xy=eps_pos, xytext=eps_pos, color=color)
        ax.plot((xl, xh), (yl, yl), linewidth=0.5, color=color)
        ax.plot((xl, xh), (yh, yh), linewidth=0.5, color=color)
        ax.plot((xl, xl), (yl, yh), linewidth=0.5, color=color)
        ax.plot((xh, xh), (yl, yh), linewidth=0.5, color=color)
    if wl_text_pos is not None:
        if box_id == len(region_list):
            color = "yellow"
        else:
            color = "white"
        ax.annotate(r"$\lambda$" + f" = {wavelength.item():.3f}", xy=wl_text_pos, xytext=wl_text_pos, color=color)
        # fig.colorbar(h2, label=pol, ax=axes[1,i])
        # fig.colorbar(h3, label=pol, ax=axes[2,i])
    # ax.annotate(f"Runtime = {time:.3f} s", xy=(field_val.shape[-1]//2-30, field_val.shape[-2]), xytext=(field_val.shape[-1]//2-40, field_val.shape[-2]+1), color="black", annotation_clip=False)
    if time is not None:
        ax.annotate(
            f"Runtime = {time:.3f} s, FPS = {fps:.1f}",
            xy=(field_val.shape[-1] // 2 - 110, field_val.shape[-2] + 3),
            xytext=(field_val.shape[-1] // 2 - 110, field_val.shape[-2] + 3),
            color="black",
            annotation_clip=False,
        )

    if box_id == len(region_list) + 1:
        color = "blue"
    else:
        color = "black"
    ax.annotate(
        r"$l_z$" + f" = {grid_step[..., 0].item()*field_val.shape[-1]:.2f} " + r"$\mu m$",
        xy=(field_val.shape[-1] // 2 - 30, -15),
        xytext=(field_val.shape[-1] // 2 - 30, -15),
        color=color,
        annotation_clip=False,
    )
    ax.annotate(
        r"$l_x$" + f" = {grid_step[..., 1].item()*field_val.shape[-2]:.2f} " + r"$\mu m$",
        xy=(-18, field_val.shape[-2] // 2 - 44),
        xytext=(-18, field_val.shape[-2] // 2 - 44),
        color=color,
        annotation_clip=False,
        rotation=90,
    )

    # Do black and white so we can see on both magma and RdBu

    ax.contour(outline_val[0], levels=1, linewidths=1.0, colors="w")
    ax.contour(outline_val[0], levels=1, linewidths=0.5, colors="k")
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()
