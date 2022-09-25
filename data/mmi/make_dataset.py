"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-01 14:27:33
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-01 15:13:21
"""
# add angler to path (not necessary if pip installed)
import sys

import matplotlib.pylab as plt
import numpy as np
import torch
from tqdm import tqdm

from device_shape import (
    mmi_2x2_L_random,
    mmi_3x3_L,
    mmi_3x3_L_random,
    mmi_3x3_L_random_slots,
    mmi_4x4_L,
    mmi_4x4_L_random,
    mmi_4x4_L_random_3pads,
    mmi_5x5_L_random,
    mmi_6x6_L_random,
)

sys.path.append("..")

from itertools import product

# import the main simulation and optimization classes
from angler import Optimization, Simulation

from device_shape import *

# import some structure generators


def generate_mmi_random_data(configs, name):
    # each epsilon combination randomly sample an MMI box size, treat them as unified permittivies distribution
    c0 = 299792458  # speed of light in vacuum (m/s)
    source_amp = 1e-9  # amplitude of modal source (make around 1 for nonlinear effects)
    neff_si = 3.48
    epsilon_list = []
    field_list = []
    grid_step_list = []
    wavelength_list = []
    input_len_list = []

    for idx, config in enumerate(configs):
        print(f"Generating data with config:\n\t{config} ({idx:4d}/{len(configs):4d})")
        pol, device_fn, eps_range, n_points, wavelengths, size, random_state = config
        eps_min, eps_max = eps_range
        device = device_fn()
        n_eps = [(np.linspace(0, 1, n_points) * (eps_max - eps_min) + eps_min).tolist()] * len(device.pad_regions)
        device_id = 0
        for eps in tqdm(product(*n_eps)):
            np.random.seed(random_state + device_id)
            device = device_fn()  # re-sample device shape
            device_id += 1
            for wavelength in wavelengths:
                lambda0 = wavelength / 1e6  # free space wavelength (m)
                omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
                eps_map = device.set_pad_eps(eps)
                epsilon_list_tmp = []
                field_list_tmp = []
                grid_step_list_tmp = []
                wavelength_list_tmp = []
                input_len_list_tmp = []
                for i in range(device.num_in_ports):
                    simulation = Simulation(omega, eps_map, device.grid_step, device.NPML, pol)
                    simulation.add_mode(
                        neff=neff_si,
                        direction_normal="x",
                        center=device.in_port_centers_px[i],
                        width=int(2 * device.in_port_width_px[i]),
                        scale=source_amp,
                    )
                    simulation.setup_modes()
                    (Ex, Ey, Hz) = simulation.solve_fields()

                    if pol == "Hz":
                        field = np.stack(
                            [
                                simulation.fields["Ex"],
                                simulation.fields["Ey"],
                                simulation.fields["Hz"],
                            ],
                            axis=0,
                        )
                    else:
                        field = np.stack(
                            [
                                simulation.fields["Hx"],
                                simulation.fields["Hy"],
                                simulation.fields["Ez"],
                            ],
                            axis=0,
                        )

                    eps_map_resize, grid_step = device.resize(device.trim_pml(eps_map), size=size, mode="bilinear")
                    # simulation.eps_r = eps_map_resize
                    # simulation.plt_eps(outline=False)
                    # plt.savefig(f"angler_gen_smmi_eps.png", dpi=300)
                    # exit(0)
                    epsilon_list_tmp.append(eps_map_resize)
                    field, _ = device.resize(device.trim_pml(field), size=size, mode="bilinear")
                    # simulation.fields["Hz"] = field[2]
                    # simulation.plt_re(outline=False)
                    # plt.savefig(f"angler_gen_mmi_simu.png", dpi=300)
                    # exit(0)
                    field_list_tmp.append(field)
                    wavelength_list_tmp.append(np.array([wavelength]))
                    grid_step_list_tmp.append(np.array(grid_step))
                    input_len_list_tmp.append(np.array([int(device.port_len / grid_step[0])]))
                 
                epsilon_list.append(np.stack(epsilon_list_tmp, axis=0))
                field_list.append(np.stack(field_list_tmp, axis=0))
                wavelength_list.append(np.stack(wavelength_list_tmp, axis=0))
                grid_step_list.append(np.stack(grid_step_list_tmp, axis=0))
                input_len_list.append(np.stack(input_len_list_tmp, axis=0))

    epsilon_list = torch.from_numpy(
        np.stack(epsilon_list, axis=0).astype(np.complex64)[:, :, np.newaxis, :, :]
    ).transpose(-1, -2)
    field_list = torch.from_numpy(np.stack(field_list, axis=0).astype(np.complex64)).transpose(-1, -2)
    grid_step_list = torch.from_numpy(np.stack(grid_step_list, axis=0).astype(np.float32))
    wavelength_list = torch.from_numpy(np.stack(wavelength_list, axis=0).astype(np.float32))
    input_len_list = torch.from_numpy(np.stack(input_len_list, axis=0).astype(np.int32))
    print(
        epsilon_list.shape,
        field_list.shape,
        grid_step_list.shape,
        wavelength_list.shape,
        input_len_list.shape,
    )
    import os

    if not os.path.isdir("./raw"):
        os.mkdir("./raw")
    torch.save(
        {
            "eps": epsilon_list,
            "fields": field_list,
            "wavelength": wavelength_list,
            "grid_step": grid_step_list,
            "input_len": input_len_list,
        },
        f"./raw/{name}.pt",
    )
    print(f"Saved simulation data ./raw/{name}.pt")


def generate_mmi_random_spectra_data(configs, name):
    # each epsilon combination randomly sample an MMI box size, treat them as unified permittivies distribution
    c0 = 299792458  # speed of light in vacuum (m/s)
    source_amp = 1e-9  # amplitude of modal source (make around 1 for nonlinear effects)
    neff_si = 3.48
    epsilon_list = []
    field_list = []
    grid_step_list = []
    wavelength_list = []
    input_len_list = []

    for idx, config in enumerate(configs):
        print(f"Generating data with config:\n\t{config} ({idx:4d}/{len(configs):4d})")
        pol, device_fn, eps_range, n_points, wavelengths, size, random_state = config
        eps_min, eps_max = eps_range
        device = device_fn()

        device_id = 0
        for _ in tqdm(range(n_points)):
            np.random.seed(random_state + device_id)
            device = device_fn()  # re-sample device shape
            device_id += 1
            eps = (
                np.round(np.random.uniform(0, 7, size=[len(device.pad_regions)])) / 7 * (eps_max - eps_min) + eps_min
            ).tolist()
            for wavelength in wavelengths:
                lambda0 = wavelength / 1e6  # free space wavelength (m)
                omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
                eps_map = device.set_pad_eps(eps)
                epsilon_list_tmp = []
                field_list_tmp = []
                grid_step_list_tmp = []
                wavelength_list_tmp = []
                input_len_list_tmp = []
                for i in range(device.num_in_ports):
                    simulation = Simulation(omega, eps_map, device.grid_step, device.NPML, pol)
                    simulation.add_mode(
                        neff=neff_si,
                        direction_normal="x",
                        center=device.in_port_centers_px[i],
                        width=int(2 * device.in_port_width_px[i]),
                        scale=source_amp,
                    )
                    simulation.setup_modes()
                    (Ex, Ey, Hz) = simulation.solve_fields()

                    if pol == "Hz":
                        field = np.stack(
                            [
                                simulation.fields["Ex"],
                                simulation.fields["Ey"],
                                simulation.fields["Hz"],
                            ],
                            axis=0,
                        )
                    else:
                        field = np.stack(
                            [
                                simulation.fields["Hx"],
                                simulation.fields["Hy"],
                                simulation.fields["Ez"],
                            ],
                            axis=0,
                        )

                    eps_map_resize, grid_step = device.resize(device.trim_pml(eps_map), size=size, mode="bilinear")
                    # simulation.eps_r = eps_map_resize
                    # simulation.plt_eps(outline=False)
                    # plt.savefig(f"angler_gen_smmi_eps.png", dpi=300)
                    # exit(0)
                    epsilon_list_tmp.append(eps_map_resize)
                    field, _ = device.resize(device.trim_pml(field), size=size, mode="bilinear")
                    ## draw fields
                    # simulation.fields["Hz"] = field[2]
                    # simulation.plt_re(outline=False)
                    # plt.savefig(f"angler_gen_mmi{device.num_in_ports}x{device.num_in_ports}_simu.png", dpi=400)
                    # exit(0)
                    field_list_tmp.append(field)
                    wavelength_list_tmp.append(np.array([wavelength]))
                    grid_step_list_tmp.append(np.array(grid_step))
                    input_len_list_tmp.append(np.array([int(device.port_len / grid_step[0])]))
                 
                epsilon_list.append(np.stack(epsilon_list_tmp, axis=0))
                field_list.append(np.stack(field_list_tmp, axis=0))
                wavelength_list.append(np.stack(wavelength_list_tmp, axis=0))
                grid_step_list.append(np.stack(grid_step_list_tmp, axis=0))
                input_len_list.append(np.stack(input_len_list_tmp, axis=0))

    epsilon_list = torch.from_numpy(
        np.stack(epsilon_list, axis=0).astype(np.complex64)[:, :, np.newaxis, :, :]
    ).transpose(-1, -2)
    field_list = torch.from_numpy(np.stack(field_list, axis=0).astype(np.complex64)).transpose(-1, -2)
    grid_step_list = torch.from_numpy(np.stack(grid_step_list, axis=0).astype(np.float32))
    wavelength_list = torch.from_numpy(np.stack(wavelength_list, axis=0).astype(np.float32))
    input_len_list = torch.from_numpy(np.stack(input_len_list, axis=0).astype(np.int32))
    print(
        epsilon_list.shape,
        field_list.shape,
        grid_step_list.shape,
        wavelength_list.shape,
        input_len_list.shape,
    )
    import os

    if not os.path.isdir("./raw"):
        os.mkdir("./raw")
    torch.save(
        {
            "eps": epsilon_list,
            "fields": field_list,
            "wavelength": wavelength_list,
            "grid_step": grid_step_list,
            "input_len": input_len_list,
        },
        f"./raw/{name}.pt",
    )
    print(f"Saved simulation data ./raw/{name}.pt")


def generate_slot_mmi_random_data(configs, name):
    # each epsilon combination randomly sample an MMI box size, treat them as unified permittivies distribution
    c0 = 299792458  # speed of light in vacuum (m/s)
    source_amp = 1e-9  # amplitude of modal source (make around 1 for nonlinear effects)
    neff_si = 3.48
    epsilon_list = []
    field_list = []
    grid_step_list = []
    wavelength_list = []
    input_len_list = []

    for idx, config in enumerate(configs):
        print(f"Generating data with config:\n\t{config} ({idx:4d}/{len(configs):4d})")
        pol, device_fn, eps_val, n_points, wavelengths, size, random_state = config
        # n_points: how many samples for each wavelength

        device_id = 0
        for _ in tqdm(range(n_points)):
            np.random.seed(random_state + device_id)
            device = device_fn()  # re-sample device shape
            device_id += 1
            for wavelength in wavelengths:
                lambda0 = wavelength / 1e6  # free space wavelength (m)
                omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
                eps_map = device.set_pad_eps(np.zeros([len(device.pad_regions)]) + eps_val)
                epsilon_list_tmp = []
                field_list_tmp = []
                grid_step_list_tmp = []
                wavelength_list_tmp = []
                input_len_list_tmp = []
                for i in range(device.num_in_ports):
                    simulation = Simulation(omega, eps_map, device.grid_step, device.NPML, pol)
                    simulation.add_mode(
                        neff=neff_si,
                        direction_normal="x",
                        center=device.in_port_centers_px[i],
                        width=int(2 * device.in_port_width_px[i]),
                        scale=source_amp,
                    )
                    simulation.setup_modes()
                    (Ex, Ey, Hz) = simulation.solve_fields()

                    if pol == "Hz":
                        field = np.stack(
                            [
                                simulation.fields["Ex"],
                                simulation.fields["Ey"],
                                simulation.fields["Hz"],
                            ],
                            axis=0,
                        )
                    else:
                        field = np.stack(
                            [
                                simulation.fields["Hx"],
                                simulation.fields["Hy"],
                                simulation.fields["Ez"],
                            ],
                            axis=0,
                        )

                    eps_map_resize, grid_step = device.resize(device.trim_pml(eps_map), size=size, mode="bilinear")
                    # simulation.eps_r = eps_map_resize
                    # simulation.plt_eps(outline=False)
                    # plt.savefig(f"angler_gen_smmi_eps.png", dpi=300)
                    # exit(0)
                    epsilon_list_tmp.append(eps_map_resize)
                    field, _ = device.resize(device.trim_pml(field), size=size, mode="bilinear")
                    # simulation.fields["Hz"] = field[2]
                    # simulation.plt_re(outline=False)
                    # plt.savefig(f"angler_gen_smmi_simu.png", dpi=300)
                    # exit(0)
                    field_list_tmp.append(field)
                    wavelength_list_tmp.append(np.array([wavelength]))
                    grid_step_list_tmp.append(np.array(grid_step))
                    input_len_list_tmp.append(np.array([int(device.port_len / grid_step[0])]))
               
                epsilon_list.append(np.stack(epsilon_list_tmp, axis=0))
                field_list.append(np.stack(field_list_tmp, axis=0))
                wavelength_list.append(np.stack(wavelength_list_tmp, axis=0))
                grid_step_list.append(np.stack(grid_step_list_tmp, axis=0))
                input_len_list.append(np.stack(input_len_list_tmp, axis=0))

    epsilon_list = torch.from_numpy(
        np.stack(epsilon_list, axis=0).astype(np.complex64)[:, :, np.newaxis, :, :]
    ).transpose(-1, -2)
    field_list = torch.from_numpy(np.stack(field_list, axis=0).astype(np.complex64)).transpose(-1, -2)
    grid_step_list = torch.from_numpy(np.stack(grid_step_list, axis=0).astype(np.float32))
    wavelength_list = torch.from_numpy(np.stack(wavelength_list, axis=0).astype(np.float32))
    input_len_list = torch.from_numpy(np.stack(input_len_list, axis=0).astype(np.int32))
    print(
        epsilon_list.shape,
        field_list.shape,
        grid_step_list.shape,
        wavelength_list.shape,
        input_len_list.shape,
    )
    import os

    if not os.path.isdir("./raw"):
        os.mkdir("./raw")
    torch.save(
        {
            "eps": epsilon_list,
            "fields": field_list,
            "wavelength": wavelength_list,
            "grid_step": grid_step_list,
            "input_len": input_len_list,
        },
        f"./raw/{name}.pt",
    )
    print(f"Saved simulation data ./raw/{name}.pt")


def postprocess(name, epsilon_min=1, epsilon_max=12.3):
    data = torch.load(f"./raw/{name}.pt")
    epsilon = data["eps"]  # [bs, N, h, w] complex
    fields = data["fields"]  # [bs, N, 3, h, w] complex
    wavelengths = data["wavelength"]  # [bs, N, 1] real
    grid_steps = data["grid_step"]  # [bs, N, 2] real
    input_lens = data["input_len"]  # [bs, N, 1] int

    # normalize epsilon
    epsilon = (epsilon - 1) / (epsilon_max - 1)
    print(epsilon.shape, epsilon_min, epsilon_max)

    # normalize fields
    fields = fields[:, :, 2:3]  # Hz or Ez only
    mag = fields.abs()
    mag_mean = mag.mean(dim=(0, 1, 3, 4), keepdim=True)
    for i in range(mag_mean.shape[2]):
        if mag_mean[:, :, i].mean() > 1e-18:
            mag_std = mag[:, :, i : i + 1].std()
            fields[:, :, i : i + 1] /= mag_std * 2
    print(fields.shape, fields.abs().max(), fields.abs().std())

    # append input mode
    input_mode = fields.clone()
    input_mask = torch.zeros(
        input_mode.shape[0], input_mode.shape[1], 1, 1, input_mode.shape[-1]
    )  # [bs, N, 1, 1, 1, w]
    for i in range(input_mode.shape[0]):
        input_mask[i, ..., : int(input_lens[i, 0, 0])].fill_(1)
    input_mode.mul_(input_mask)

    # make data
    data = torch.cat([epsilon, input_mode], dim=2)
    print(data.shape, fields.shape, wavelengths.shape, grid_steps.shape)
    print(f"postprocessed {name}")
    torch.save(
        {
            "eps": data,
            "fields": fields,
            "wavelength": wavelengths,
            "grid_step": grid_steps,
            "eps_min": torch.tensor([1.0]),
            "eps_max": torch.tensor([epsilon_max]),
        },
        f"./raw/{name}_mode.pt",
    )


def append_input_mode(pol_list):
    pol_list = sorted(pol_list)
    data = torch.load(f"./raw/{'_'.join(pol_list)}_fields_epsilon.pt")
    epsilon = data["eps"][:, :, 0:1]
    epsilon_min, epsilon_max = epsilon.abs().min().item(), epsilon.abs().max().item()
    epsilon = (epsilon - 1) / (epsilon_max - 1)
    print(epsilon.shape, epsilon_min, epsilon_max)

    wavelength = data["wavelength"]
    if pol_list == ["Hz"]:
        fields = data["fields"][:, :, 0:2]  # Ex, Ey, Ez only
    elif pol_list == ["Ez"]:
        fields = data["fields"][:, :, 2:3]  # Ex, Ey, Ez only
    else:
        fields = data["fields"][:, :, 0:3]  # Ex, Ey, Ez only
    print(fields.shape)

    mag = fields.abs()
    mag_mean = mag.mean(dim=(0, 1, 3, 4), keepdim=True)
    for i in range(mag_mean.shape[2]):
        if mag_mean[:, :, i].mean() > 1e-18:
            mag_std = mag[:, :, i : i + 1].std()
            fields[:, :, i : i + 1] /= mag_std * 2

    print(fields.abs().max(), fields.abs().std())
    input_mode = fields.clone()
    input_taper = 9.9
    dl = 0.1
    input_mode[..., int(input_taper / dl) :].fill_(0)  # only know input mode
    epsilon = torch.cat([epsilon, input_mode], dim=2)
    grid_step = data["grid_step"]
    print(epsilon.shape, fields.shape, wavelength.shape, grid_step.shape)


def launch_rHz_data_mmi2x2_generation():
    pol = "Hz"
    device_list = [mmi_2x2_L_random]
    points_per_wavelength = [128]
    eps_range = [11.9, 12.3]

    size = (384, 80)

    wavelengths = np.arange(1.53, 1.571, 0.01).tolist()
    tasks = list(enumerate(wavelengths))

    for i, wavelength in tasks:
        name = f"rHz_mmi2x2_{i}_fields_epsilon"
        configs = [
            (pol, device, eps_range, n_points, [wavelength], size, 30000 + 2000 * i)
            for device, n_points in zip(device_list, points_per_wavelength)
        ]
        generate_mmi_random_spectra_data(configs, name=name)
        postprocess(name)


def launch_rHz_data_mmi3x3_generation():
    pol = "Hz"
    device_list = [mmi_3x3_L_random]
    points_per_port = [8]
    eps_range = [11.9, 12.3]

    size = (384, 80)
    # np.random.seed(42)  # set random seed
    wavelengths = np.arange(1.53, 1.571, 0.01).tolist()
    tasks = [
        (0, wavelengths[0]),
        (1, wavelengths[1]),
        (2, wavelengths[2]),
        (3, wavelengths[3]),
        (4, wavelengths[4]),
    ]

    # wavelengths = np.arange(1.535, 1.576, 0.01).tolist()
    # tasks = [
    #     (5, wavelengths[0]),
    #     (6, wavelengths[1]),
    #     (7, wavelengths[2]),
    #     (8, wavelengths[3]),
    #     (9, wavelengths[4]),
    # ]

    for i, wavelength in tasks:
        name = f"rHz_{i}_fields_epsilon"
        configs = [
            (pol, device, eps_range, n_points, [wavelength], size, int(10000 + i * 2000))
            for device, n_points in zip(device_list, points_per_port)
        ]
        generate_mmi_random_data(configs, name=name)
        postprocess(name)


def launch_rHz_data_mmi4x4_generation():
    pol = "Hz"
    device_list = [mmi_4x4_L_random]
    points_per_wavelength = [256]
    eps_range = [11.9, 12.3]

    size = (384, 80)

    wavelengths = np.arange(1.53, 1.571, 0.01).tolist()
    tasks = list(enumerate(wavelengths))
    
    for i, wavelength in tasks:
        name = f"rHz_mmi4x4_{i}_fields_epsilon"
        configs = [
            (pol, device, eps_range, n_points, [wavelength], size, 30000 + 2000 * i)
            for device, n_points in zip(device_list, points_per_wavelength)
        ]
        generate_mmi_random_spectra_data(configs, name=name)
        postprocess(name)


def launch_rHz_data_mmi5x5_generation():
    pol = "Hz"
    device_list = [mmi_5x5_L_random]
    points_per_wavelength = [256]
    eps_range = [11.9, 12.3]

    size = (384, 80)

    wavelengths = np.arange(1.53, 1.571, 0.01).tolist()
    tasks = list(enumerate(wavelengths))

    for i, wavelength in tasks:
        name = f"rHz_mmi5x5_{i}_fields_epsilon"
        configs = [
            (pol, device, eps_range, n_points, [wavelength], size, 30000 + 2000 * i)
            for device, n_points in zip(device_list, points_per_wavelength)
        ]
        generate_mmi_random_spectra_data(configs, name=name)
        postprocess(name)


def launch_slot_rHz_data_mmi3x3_generation():
    pol = "Hz"
    device_list = [mmi_3x3_L_random_slots]
    points_per_port = [512]
    eps_val = 1.44**2
    wavelengths = np.arange(1.53, 1.571, 0.01).tolist()
    size = (384, 80)
    # np.random.seed(42)  # set random seed
    tasks = [
        (0, wavelengths[0]),
        (1, wavelengths[1]),
        (2, wavelengths[2]),  
        (3, wavelengths[3]),
        (4, wavelengths[4]),  
    ]

    wavelengths = np.arange(1.535, 1.576, 0.01).tolist()
    tasks = [
        (5, wavelengths[0]),
        (6, wavelengths[1]), 
        (7, wavelengths[2]), 
        (8, wavelengths[3]),
        (9, wavelengths[4]), 
    ]

    for i, wavelength in tasks:
        name = f"slot_rHz_{i}_fields_epsilon"
        configs = [
            (pol, device, eps_val, n_points, [wavelength], size, int(10000 + i * 2000))
            for device, n_points in zip(device_list, points_per_port)
        ]
        generate_slot_mmi_random_data(configs, name=name)
        postprocess(name)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1]

    if mode == "mmi2x2":
        launch_rHz_data_mmi2x2_generation()
    if mode == "mmi3x3":
        launch_rHz_data_mmi3x3_generation()
    elif mode == "mmi4x4":
        launch_rHz_data_mmi4x4_generation()
    elif mode == "mmi5x5":
        launch_rHz_data_mmi5x5_generation()
    elif mode == "mmi3x3_etched":
        launch_slot_rHz_data_mmi3x3_generation()
    else:
        print(f"Not supported mode: {mode}")
