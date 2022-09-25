"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-15 19:28:45
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-15 19:28:45
"""
from typing import Optional, Tuple
import numpy as np
from angler.structures import get_grid
import torch
from pyutils.compute import gen_gaussian_filter2d_cpu
from itertools import product

eps_sio2 = 1.44 ** 2
eps_si = 3.48 ** 2

__all__ = ["MMI_NxM", "mmi_2x2", "mmi_3x3", "mmi_4x4", "mmi_6x6", "mmi_8x8"]


def apply_regions(reg_list, xs, ys, eps_r_list, eps_bg):
    # feed this function a list of regions and some coordinates and it spits out a permittivity
    if isinstance(eps_r_list, (int, float)):
        eps_r_list = [eps_r_list] * len(reg_list)
    # if it's not a list, make it one
    if not isinstance(reg_list, (list, tuple)):
        reg_list = [reg_list]

    # initialize permittivity
    eps_r = np.zeros(xs.shape) + eps_bg

    # loop through lambdas and apply masks
    for e, reg in zip(eps_r_list, reg_list):
        reg_vec = np.vectorize(reg)
        material_mask = reg_vec(xs, ys)
        eps_r[material_mask] = e

    return eps_r


def gaussian_blurring(x):
    # return x
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
    size = 3
    std = 0.4
    ax = torch.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    kernel = torch.exp(-0.5 / std ** 2 * (xx ** 2 + yy ** 2))
    kernel = kernel.div(kernel.sum()).unsqueeze(0).unsqueeze(0).float()
    return torch.nn.functional.conv2d(x, kernel, padding=size // 2).squeeze(0).squeeze(0).numpy()


class MMI_NxM(object):
    def __init__(
        self,
        num_in_ports: int,
        num_out_ports: int,
        box_size: Tuple[float, float],  # box [length, width], um
        wg_width: Tuple[float, float] = (0.4, 0.4),  # in/out wavelength width, um
        port_diff: Tuple[float, float] = (4, 4),  # distance between in/out waveguides. um
        port_len: float = 10,  # length of in/out waveguide from PML to box. um
        border_width: float = 3,  # space between box and PML. um
        grid_step: float = 0.1,  # isotropic grid step um
        NPML: Tuple[int, int] = (20, 20),  # PML pixel width. pixel
        eps_r: float = eps_si,  # relative refractive index
        eps_bg: float = eps_sio2,  # background refractive index
    ):
        super().__init__()
        self.num_in_ports = num_in_ports
        self.num_out_ports = num_out_ports
        self.box_size = box_size
        self.wg_width = wg_width
        self.port_diff = port_diff
        self.port_len = port_len
        self.border_width = border_width
        self.grid_step = grid_step
        self.NPML = list(NPML)
        self.eps_r = eps_r
        self.eps_bg = eps_bg
        # geometric parameters
        Nx = 2 * NPML[0] + int(round((port_len * 2 + box_size[0]) / grid_step))  # num. grids in horizontal
        Ny = 2 * NPML[1] + int(round((box_size[1] + 2 * border_width) / grid_step))  # num. grids in vertical

        self.shape = (Nx, Ny)  # shape of domain (in num. grids)

        y_mid = 0
        # self.wg_width_px = [int(round(i / grid_step)) for i in wg_width]

        # x and y coordinate arrays
        self.xs, self.ys = get_grid(self.shape, grid_step)

        # define the regions
        box = lambda x, y: (np.abs(x) < box_size[0] / 2) * (np.abs(y - y_mid) < box_size[1] / 2)

        in_ports = []
        out_ports = []
        for i in range(num_in_ports):
            y_i = (i - (num_in_ports - 1) / 2) * port_diff[0]
            # wg_i = lambda x, y, y_i=y_i: (x < 0) * (abs(y - y_i) < grid_step * self.wg_width_px[0] / 2)
            wg_i = lambda x, y, y_i=y_i: (x < 0) * (abs(y - y_i) < self.wg_width[0] / 2)
            in_ports.append(wg_i)

        for i in range(num_out_ports):
            y_i = (i - (num_out_ports - 1) / 2.0) * port_diff[1]
            wg_i = lambda x, y, y_i=y_i: (x > 0) * (abs(y - y_i) < self.wg_width[1] / 2)
            out_ports.append(wg_i)
        reg_list = in_ports + out_ports + [box]

        self.epsilon_map = apply_regions(reg_list, self.xs, self.ys, eps_r_list=eps_r, eps_bg=eps_bg)

        self.design_region = apply_regions([box], self.xs, self.ys, eps_r_list=1, eps_bg=0)
        self.pad_regions = None

        self.in_port_centers = [
            (-box_size[0] / 2 - 0.98 * port_len, (i - (num_in_ports - 1) / 2) * port_diff[0])
            for i in range(num_in_ports)
        ]  # centers

        # 001110011100
        # 01001010010 -> 1,4,6,9 -> 3, 8
        # 00111100111100
        # 0100010100010 -> 1,5,7,11 -> 4, 10
        cut = self.epsilon_map[0] > eps_bg
        # print(cut)
        edges = cut[1:] ^ cut[:-1]
        indices = np.nonzero(edges)[0]
        self.in_port_width_px = (indices[1::2] - indices[::2]).tolist()
        # print(indices)
        # print(self.in_port_width_px)
        centers = (indices[::2] + indices[1::2]) // 2 + 1
        self.in_port_centers_px = [
            (NPML[0] + int(round(box_size[0] / 2 + port_len - np.abs(x))), y)
            for (x, _), y in zip(self.in_port_centers, centers)
        ]
        # print(self.in_port_centers)
        # print(self.in_port_centers_px)
        # print(
        #     self.epsilon_map[
        #         self.in_port_centers_px[0][0],
        #         self.in_port_centers_px[0][1] - 5 : self.in_port_centers_px[0][1] + 5,
        #     ]
        # )

        cut = self.epsilon_map[-1] > eps_bg
        # print(cut)
        edges = cut[1:] ^ cut[:-1]
        indices = np.nonzero(edges)[0]
        self.out_port_width_px = (indices[1::2] - indices[::2]).tolist()
        # print(indices)
        # print(self.out_port_width_px)
        centers = (indices[::2] + indices[1::2]) // 2 + 1
        self.out_port_centers = [
            (box_size[0] / 2 + 0.98 * port_len, (float(i) - float(num_out_ports - 1) / 2.0) * port_diff[1])
            for i in range(num_out_ports)
        ]  # centers
        self.out_port_pixel_centers = [
            (
                Nx - 1 - NPML[0] - int(round(box_size[0] / 2 + port_len - np.abs(x))),
                NPML[1] + int(round((border_width + box_size[1] / 2 + y) / grid_step)),
            )
            for x, y in self.out_port_centers
        ]

        self.epsilon_map = gaussian_blurring(self.epsilon_map)

    def set_pad_region(self, pad_regions):
        # pad_regions = [[xl, xh, yl, yh], [xl, xh, yl, yh], ...] rectanglar pads bounding box
        # (0,0) is the center of the entire region
        # default argument in lambda can avoid lazy evaluation in python!
        self.pad_regions = [
            lambda x, y, xl=xl, xh=xh, yl=yl, yh=yh: (xl < x < xh) and (yl < y < yh)
            for xl, xh, yl, yh in pad_regions
        ]
        self.pad_region_mask = apply_regions(
            self.pad_regions, self.xs, self.ys, eps_r_list=1, eps_bg=0
        ).astype(np.bool)

    def set_pad_eps(self, pad_eps) -> np.ndarray:
        assert self.pad_regions is not None and len(pad_eps) == len(self.pad_regions)
        epsilon_map = apply_regions(self.pad_regions, self.xs, self.ys, eps_r_list=pad_eps, eps_bg=0)
        return np.where(self.pad_region_mask, epsilon_map, self.epsilon_map)

    def trim_pml(self, epsilon_map: Optional[np.ndarray] = None):
        epsilon_map = epsilon_map if epsilon_map is not None else self.epsilon_map
        return epsilon_map[
            ...,
            self.NPML[0] : epsilon_map.shape[-2] - self.NPML[-2],
            self.NPML[1] : epsilon_map.shape[-1] - self.NPML[-1],
        ]

    def resize(self, x, size, mode="bilinear"):
        if not isinstance(x, torch.Tensor):
            y = torch.from_numpy(x)
        else:
            y = x
        y = y.view(-1, 1, x.shape[-2], x.shape[-1])
        old_grid_step = (self.grid_step, self.grid_step)
        old_size = y.shape[-2:]
        new_grid_step = [old_size[0] / size[0] * old_grid_step[0], old_size[1] / size[1] * old_grid_step[1]]
        if y.is_complex():
            y = torch.complex(
                torch.nn.functional.interpolate(y.real, size=size, mode=mode),
                torch.nn.functional.interpolate(y.imag, size=size, mode=mode),
            )
        else:
            y = torch.nn.functional.interpolate(y, size=size, mode=mode)
        y = y.view(list(x.shape[:-2]) + list(size))
        if isinstance(x, np.ndarray):
            y = y.numpy()
        return y, new_grid_step

    def __repr__(self) -> str:
        str = f"MMI{self.num_in_ports}x{self.num_out_ports}("
        str += f"size = {self.box_size[0]} um x {self.box_size[1]} um)"
        return str


def mmi_2x2():
    N = 2
    wl = 1.55
    index_si = 3.48
    size = (12, 3)
    port_len = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wl / index_si / 2, wl / index_si / 2),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (10, 0.8)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_2x2_L_random():
    N = 2
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(15, 20), np.random.uniform(3, 5))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (size[0] * np.random.uniform(0.7, 0.9), size[1] / N * np.random.uniform(0.4, 0.65))
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi



def mmi_3x3():
    N = 3
    wl = 1.55
    index_si = 3.48
    size = (13.5, 3.5)
    port_len = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wl / index_si / 2, wl / index_si / 2),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (10, 0.8)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_3x3_L():
    N = 3
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    size = (25.9, 6.1)
    port_len = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        # wg_width=(wl / index_si / 2, wl / index_si / 2),  # in/out wavelength width, um
        wg_width=(1.1, 1.1),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (20, 1.2)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_3x3_L_random():
    N = 3
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(20, 30), np.random.uniform(5.5, 7))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (size[0] * np.random.uniform(0.7, 0.9), size[1] / N * np.random.uniform(0.4, 0.65))
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi



def mmi_3x3_L_random_slots():
    ## random rectangular SiO2 slots
    N = 3
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(20, 30), np.random.uniform(5.5, 7))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    n_slots = (30, 7)
    total_slots = n_slots[0] * n_slots[1]
    n_sampled_slots = int(np.random.uniform(0.05, 0.1) * total_slots)
    w, h = size[0] / n_slots[0] * 0.8, size[1] / n_slots[1] * 0.8  # do not remove materials on the boundary
    slot_centers_x = np.linspace(-(n_slots[0] / 2 - 0.5) * w, (n_slots[0] / 2 - 0.5) * w, n_slots[0])
    slot_centers_y = np.linspace(-(n_slots[1] / 2 - 0.5) * h, (n_slots[1] / 2 - 0.5) * h, n_slots[1])

    centers_x = np.random.choice(slot_centers_x, size=n_sampled_slots, replace=True)
    centers_y = slot_centers_y[
        (np.round(np.random.choice(len(slot_centers_y), size=n_sampled_slots, replace=True) / 2) * 2).astype(
            np.int32
        )
    ]  # a trick to generate slots along the prop direction
    pad_centers = np.stack([centers_x, centers_y], axis=-1)
    # pad_centers = np.array(list(product(slot_centers_x, slot_centers_y)))[np.random.choice(total_slots, size=n_sampled_slots, replace=False)]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_4x4():
    N = 4
    wl = 1.55
    index_si = 3.48
    size = (16, 4)
    port_len = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        # wg_width=(wl / index_si / 2, wl / index_si / 2),  # in/out wavelength width, um
        wg_width=(0.8, 0.8),
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (11, 0.7)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_4x4_L():
    N = 4
    wl = 1.55
    index_si = 3.48
    size = (31.5, 6.1)
    port_len = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        # wg_width=(wl / index_si / 2, wl / index_si / 2),  # in/out wavelength width, um
        wg_width=(1.0, 1.0),
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (20, 1)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_4x4_L_random():
    N = 4
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(20, 30), np.random.uniform(5.5, 7))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (size[0] * np.random.uniform(0.7, 0.9), size[1] / N * np.random.uniform(0.4, 0.65))
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi

def mmi_4x4_L_random_3pads():
    N = 4
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(20, 30), np.random.uniform(5.5, 7))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    mmi3 = MMI_NxM(
        3,
        3,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (size[0] * np.random.uniform(0.7, 0.9), size[1] / N * np.random.uniform(0.4, 0.65))
    pad_centers = [(0, y) for _, y in mmi3.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_5x5_L_random():
    N = 5
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(25, 30), np.random.uniform(7, 9))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (size[0] * np.random.uniform(0.7, 0.9), size[1] / N * np.random.uniform(0.4, 0.65))
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_6x6_L_random():
    N = 6
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(40, 50), np.random.uniform(10, 14))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (size[0] * np.random.uniform(0.7, 0.9), size[1] / N * np.random.uniform(0.4, 0.65))
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_6x6(
    *args,
    **kwargs,
):
    return MMI_NxM(6, 6, *args, **kwargs)


def mmi_8x8(
    *args,
    **kwargs,
):
    return MMI_NxM(8, 8, *args, **kwargs)
