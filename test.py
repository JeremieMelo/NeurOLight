"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-17 18:52:06
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-17 19:42:37
"""
#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.torch_train import count_parameters, load_model, set_torch_deterministic
from pyutils.typing import Criterion, DataLoader

from core import builder
from core.datasets.mixup import MixupAll
from core.utils import plot_compare, plot_dynamics


def multiport_train_multiport_test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
    test_split: str = "test",
    test_random_state: int = 0,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    mse_vec = []
    with torch.no_grad():
        for i, (wavelength, grid_step, data, target) in enumerate(test_loader):
            wavelength = wavelength.to(device, non_blocking=True)
            grid_step = grid_step.to(device, non_blocking=True)
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(data, target, random_state=i + test_random_state, vflip=False)

            wavelength, grid_step, data, target = [x.flatten(0, 1) for x in [wavelength, grid_step, data, target]]
            output = model(data, wavelength, grid_step)

            val_loss = criterion(output, target)
            mse_meter.update(val_loss.item())
            mse_vec.append(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\n(MM) Test set: Average loss: {:.4e} std: {:.4e}\n".format(mse_meter.avg, np.std(mse_vec)))
    mlflow.log_metrics({"test_loss": mse_meter.avg}, step=epoch)

    if plot and (epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"mm_{test_split}.png")
        plot_compare(
            wavelength[0:3],
            grid_step=grid_step[0:3],
            epsilon=data[0:3, 0],
            pred_fields=output[0:3, -1],
            target_fields=target[0:3, -1],
            filepath=filepath,
            pol="Hz",
            norm=False,
        )


def observe_features(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
    test_split: str = "test",
    test_random_state: int = 0,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    mse_vec = []
    with torch.no_grad():
        for i, (wavelength, grid_step, data, target) in enumerate(test_loader):
            wavelength = wavelength.to(device, non_blocking=True)
            grid_step = grid_step.to(device, non_blocking=True)
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(data, target, random_state=i + test_random_state, vflip=False)

            wavelength, grid_step, data, target = [x.flatten(0, 1) for x in [wavelength, grid_step, data, target]]
            output = model(data, wavelength, grid_step)
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(2.5, 1))
            waveprior = model.observe_waveprior(data, wavelength, grid_step)[0].cpu().numpy()

            def normalize(x):
                x_max, x_min = x.max(), x.min()
                return (x - x_min) / (x_max - x_min)

            axes[0].imshow(normalize(waveprior[0]), cmap="RdBu", vmin=-0.5, vmax=1.5)
            axes[1].imshow(normalize(waveprior[2]), cmap="RdBu", vmin=-0.5, vmax=1.5)
            for ax in axes:
                ax.axis("off")
            fig.savefig(f"mmi3x3_waveprior.png", dpi=400)

            features = model.observe_stem_output(data, wavelength, grid_step)[0].cpu().numpy()
            fig, axes = plt.subplots(8, 8, constrained_layout=True, figsize=(5 * 2, 1 * 2))
            for i in range(features.shape[0] // 8):
                for j in range(8):
                    axes[i, j].imshow(normalize(features[i * 8 + j]), cmap="RdBu", vmin=-0.5, vmax=0.8)
                    axes[i, j].axis("off")
            fig.savefig(f"mmi3x3_features.png", dpi=600)

            exit(0)
            # print(output.shape)

            val_loss = criterion(output, target)
            mse_meter.update(val_loss.item())
            mse_vec.append(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\n(MM) Test set: Average loss: {:.4e} std: {:.4e}\n".format(mse_meter.avg, np.std(mse_vec)))


def observe_dynamics(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
    test_split: str = "test",
    test_random_state: int = 0,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    mse_vec = []
    with torch.no_grad():
        for i, (wavelength, grid_step, data, target) in enumerate(test_loader):
            break
        # wavelength = wavelength[0:1]
        # grid_step = grid_step[0:1]
        # data = data[:1]
        # target = target[:1]
        xl, xh = 70, 384 - 70
        yl1, yh1 = 8, 22
        yl2, yh2 = 33, 47
        yl3, yh3 = 58, 72
        wavelength = wavelength.to(device, non_blocking=True)
        grid_step = grid_step.to(device, non_blocking=True)
        data = data.to(device, non_blocking=True)
        print(data.shape)
        target = target.to(device, non_blocking=True)

        wavelength, grid_step, data, target = [x.flatten(0, 1) for x in [wavelength, grid_step, data, target]]
        print(data.shape)
        wavelength = wavelength[1:2]
        grid_step = grid_step[1:2]
        data = data[1:2]
        target = target[1:2]
        tasks = [(e, 11, 11, 0) for e in np.linspace(11, 12.5, 50)]
        tasks += [(12.5, e, 11, 1) for e in np.linspace(11, 12.5, 50)]
        tasks += [(12.5, 12.5, e, 2) for e in np.linspace(11, 12.5, 50)]
        ref_eps = data.clone()
        ref_eps[:, 0:1, yl1:yh1, xl:xh] = (11 - 1) / (12.3 - 1)
        ref_eps[:, 0:1, yl2:yh2, xl:xh] = (11 - 1) / (12.3 - 1)
        ref_eps[:, 0:1, yl3:yh3, xl:xh] = (11 - 1) / (12.3 - 1)
        for eps1, eps2, eps3, box_id in tasks:
            data[:, 0:1, yl1:yh1, xl:xh] = (eps1 - 1) / (12.3 - 1)
            data[:, 0:1, yl2:yh2, xl:xh] = (eps2 - 1) / (12.3 - 1)
            data[:, 0:1, yl3:yh3, xl:xh] = (eps3 - 1) / (12.3 - 1)

            output = model(data, wavelength, grid_step)

            dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
            os.makedirs(dir_path, exist_ok=True)
            filepath = os.path.join(dir_path, f"eps-{eps1:.3f}-{eps2:.3f}-{eps3:.3f}_test.png")
            plot_dynamics(
                wavelength[0:1],
                grid_step=grid_step[0:1],
                epsilon=data[0:1, 0],
                pred_fields=output[0:1, -1],
                target_fields=target[0:1, -1],
                filepath=filepath,
                eps_list=[eps1, eps2, eps3],
                eps_text_loc_list=[
                    ((xl + xh) / 2 - 30, (yl1 + yh1) / 2 - 4),
                    ((xl + xh) / 2 - 30, (yl2 + yh2) / 2 - 4),
                    ((xl + xh) / 2 - 30, (yl3 + yh3) / 2 - 4),
                ],
                region_list=[(xl, xh, yl1, yh1), (xl, xh, yl2, yh2), (xl, xh, yl3, yh3)],
                box_id=box_id,
                ref_eps=ref_eps[0:1, 0],
                norm=False,
            )

        exit(0)
    loss_vector.append(mse_meter.avg)

    lg.info("\n(MM) Test set: Average loss: {:.4e} std: {:.4e}\n".format(mse_meter.avg, np.std(mse_vec)))


def observe_dynamics_spectra(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
    test_split: str = "test",
    test_random_state: int = 0,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    mse_vec = []
    with torch.no_grad():
        for i, (wavelength, grid_step, data, target) in enumerate(test_loader):
            break
        # wavelength = wavelength[0:1]
        # grid_step = grid_step[0:1]
        # data = data[:1]
        # target = target[:1]
        xl, xh = 70, 384 - 70
        yl1, yh1 = 8, 22
        yl2, yh2 = 33, 47
        yl3, yh3 = 58, 72
        wavelength = wavelength.to(device, non_blocking=True)
        grid_step = grid_step.to(device, non_blocking=True)
        data = data.to(device, non_blocking=True)
        print(data.shape)
        target = target.to(device, non_blocking=True)

        wavelength, grid_step, data, target = [x.flatten(0, 1) for x in [wavelength, grid_step, data, target]]
        print(data.shape)
        wavelength = wavelength[1:2]
        grid_step = grid_step[1:2]
        data = data[1:2]
        target = target[1:2]
        n_frames = 50
        tasks = [(1.50, e, 11, 11, 0) for e in np.linspace(11, 12.5, n_frames)]
        tasks += [(1.50, 12.5, e, 11, 1) for e in np.linspace(11, 12.5, n_frames)]
        tasks += [(1.50, 12.5, 12.5, e, 2) for e in np.linspace(11, 12.5, n_frames)]
        tasks += [(f, 12.5, 12.5, 12.5, 3) for f in np.linspace(1.50, 1.6, n_frames)]
        ref_eps = data.clone()
        ref_eps[:, 0:1, yl1:yh1, xl:xh] = (11 - 1) / (12.3 - 1)
        ref_eps[:, 0:1, yl2:yh2, xl:xh] = (11 - 1) / (12.3 - 1)
        ref_eps[:, 0:1, yl3:yh3, xl:xh] = (11 - 1) / (12.3 - 1)
        for j, (f, eps1, eps2, eps3, box_id) in enumerate(tasks):
            wavelength.data.fill_(f)
            data[:, 0:1, yl1:yh1, xl:xh] = (eps1 - 1) / (12.3 - 1)
            data[:, 0:1, yl2:yh2, xl:xh] = (eps2 - 1) / (12.3 - 1)
            data[:, 0:1, yl3:yh3, xl:xh] = (eps3 - 1) / (12.3 - 1)

            output = model(data, wavelength, grid_step)

            dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
            os.makedirs(dir_path, exist_ok=True)
            file_name = f"eps-{eps1:.3f}-{eps2:.3f}-{eps3:.3f}_wl-{wavelength[0:1].item():.3f}_test.png"
            filepath = os.path.join(dir_path, file_name)
            time = 0.0083 * (j + 1)
            print(file_name, time)
            plot_dynamics(
                wavelength[0:1],
                grid_step=grid_step[0:1],
                epsilon=data[0:1, 0],
                pred_fields=output[0:1, -1],
                target_fields=target[0:1, -1],
                filepath=filepath,
                eps_list=[eps1, eps2, eps3],
                eps_text_loc_list=[
                    ((xl + xh) / 2 - 30, (yl1 + yh1) / 2 - 4),
                    ((xl + xh) / 2 - 30, (yl2 + yh2) / 2 - 4),
                    ((xl + xh) / 2 - 30, (yl3 + yh3) / 2 - 4),
                ],
                region_list=[(xl, xh, yl1, yh1), (xl, xh, yl2, yh2), (xl, xh, yl3, yh3)],
                box_id=box_id,
                ref_eps=ref_eps[0:1, 0],
                wl_text_pos=(5, (yl3 + yh3) / 2 - 5),
                time=time,
                fps=1 / 0.0083,
                norm=False,
            )


def observe_dynamics_spectra(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
    test_split: str = "test",
    test_random_state: int = 0,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    mse_vec = []
    with torch.no_grad():
        for i, (wavelength, grid_step, data, target) in enumerate(test_loader):
            break
        # wavelength = wavelength[0:1]
        # grid_step = grid_step[0:1]
        # data = data[:1]
        # target = target[:1]
        xl, xh = 70, 384 - 70
        yl1, yh1 = 8, 22
        yl2, yh2 = 33, 47
        yl3, yh3 = 58, 72
        wavelength = wavelength.to(device, non_blocking=True)
        grid_step = grid_step.to(device, non_blocking=True)
        data = data.to(device, non_blocking=True)
        print(data.shape)
        target = target.to(device, non_blocking=True)
        if mixup_fn is not None:
            data, target = mixup_fn(data, target, random_state=i + test_random_state, vflip=False)

        wavelength, grid_step, data, target = [x.flatten(0, 1) for x in [wavelength, grid_step, data, target]]
        print(data.shape)
        wavelength = wavelength[1:2]
        grid_step = grid_step[1:2]
        # data = data[1:2]
        data = data[0:1]
        target = target[1:2]
        g_z, g_x = grid_step[..., 0].item() - 0.005, grid_step[..., 1].item() - 0.005
        n_frames = 50
        tasks = [(g_z, g_x, 1.50, e, 11, 11, 0) for e in np.linspace(11, 12.5, n_frames)]
        tasks += [(g_z, g_x, 1.50, 12.5, e, 11, 1) for e in np.linspace(11, 12.5, n_frames)]
        tasks += [(g_z, g_x, 1.50, 12.5, 12.5, e, 2) for e in np.linspace(11, 12.5, n_frames)]
        tasks += [(g_z, g_x, f, 12.5, 12.5, 12.5, 3) for f in np.linspace(1.50, 1.6, n_frames)]
        tasks += [(g_z + g, g_x + g, 1.6, 12.5, 12.5, 12.5, 4) for g in np.linspace(0, 0.013, n_frames)]
        ref_eps = data.clone()
        ref_eps[:, 0:1, yl1:yh1, xl:xh] = (11 - 1) / (12.3 - 1)
        ref_eps[:, 0:1, yl2:yh2, xl:xh] = (11 - 1) / (12.3 - 1)
        ref_eps[:, 0:1, yl3:yh3, xl:xh] = (11 - 1) / (12.3 - 1)
        for j, (g_z, g_x, f, eps1, eps2, eps3, box_id) in enumerate(tasks):
            wavelength.data.fill_(f)
            grid_step.data[..., 0] = g_z
            grid_step.data[..., 1] = g_x
            data[:, 0:1, yl1:yh1, xl:xh] = (eps1 - 1) / (12.3 - 1)
            data[:, 0:1, yl2:yh2, xl:xh] = (eps2 - 1) / (12.3 - 1)
            data[:, 0:1, yl3:yh3, xl:xh] = (eps3 - 1) / (12.3 - 1)

            output = model(data, wavelength, grid_step)

            dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
            os.makedirs(dir_path, exist_ok=True)
            file_name = f"eps-{eps1:.3f}-{eps2:.3f}-{eps3:.3f}_wl-{wavelength[0:1].item():.3f}_g-{g_z*1000:.2f}x{g_x*1000:.2f}_test.png"
            filepath = os.path.join(dir_path, file_name)
            time = 0.0083 * (j + 1)
            print(file_name, time)
            plot_dynamics(
                wavelength[0:1],
                grid_step=grid_step[0:1],
                epsilon=data[0:1, 0],
                pred_fields=output[0:1, -1],
                target_fields=target[0:1, -1],
                filepath=filepath,
                eps_list=[eps1, eps2, eps3],
                eps_text_loc_list=[
                    ((xl + xh) / 2 - 30, (yl1 + yh1) / 2 - 4),
                    ((xl + xh) / 2 - 30, (yl2 + yh2) / 2 - 4),
                    ((xl + xh) / 2 - 30, (yl3 + yh3) / 2 - 4),
                ],
                region_list=[(xl, xh, yl1, yh1), (xl, xh, yl2, yh2), (xl, xh, yl3, yh3)],
                box_id=box_id,
                ref_eps=ref_eps[0:1, 0],
                wl_text_pos=(5, (yl3 + yh3) / 2 - 5),
                time=time,
                fps=1 / 0.0083,
                norm=False,
            )


def singleport_train_multiport_test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
    test_split: str = "test",
    test_random_state: int = 0,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    mse_vec = []
    with torch.no_grad():
        for i, (wavelength, grid_step, data, target) in enumerate(test_loader):
            wavelength = wavelength.to(device, non_blocking=True)
            grid_step = grid_step.to(device, non_blocking=True)
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data.flatten(0, 1), wavelength.flatten(0, 1), grid_step.flatten(0, 1))
            output = output.view_as(target)
            if mixup_fn is not None:
                output, target = mixup_fn(output, target, random_state=i + test_random_state, vflip=False, mode_dim=0)

            output = output.flatten(0, 1)
            target = target.flatten(0, 1)
            data = data.flatten(0, 1)
            wavelength = wavelength.flatten(0, 1)
            grid_step = grid_step.flatten(0, 1)

            val_loss = criterion(output, target)
            mse_meter.update(val_loss.item())
            mse_vec.append(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\n(SM) Test set: Average loss: {:.4e} std: {:.4e}\n".format(mse_meter.avg, np.std(mse_vec)))
    mlflow.log_metrics({"test_loss": mse_meter.avg}, step=epoch)

    if plot and (epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"sm_{test_split}.png")
        plot_compare(
            wavelength[0:3],
            grid_step=grid_step[0:3],
            epsilon=data[0:3, 0],
            pred_fields=output[0:3, -1],
            target_fields=target[0:3, -1],
            filepath=filepath,
            pol="Hz",
            norm=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic()

    test_split = getattr(configs.run, "test_split", "test")
    if test_split == "test":
        _, _, test_loader = builder.make_dataloader(splits=["test"])
    elif test_split == "valid":
        _, test_loader, _ = builder.make_dataloader(splits=["valid"])
    elif test_split == "train":
        test_loader, _, _ = builder.make_dataloader(splits=["train"])
    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
        eps_min=test_loader.dataset.eps_min.item(),
        eps_max=test_loader.dataset.eps_max.item(),
    )
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(device)
    aux_criterions = {
        name: [builder.make_criterion(name, cfg=config), float(config.weight)]
        for name, config in configs.aux_criterion.items()
        if float(config.weight) > 0
    }
    print(aux_criterions)

    test_mixup_fn = MixupAll(**configs.dataset.test_augment)

    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(
        {
            "exp_name": configs.run.experiment,
            "exp_id": experiment.experiment_id,
            "run_id": mlflow.active_run().info.run_id,
            "init_lr": configs.optimizer.lr,
            "checkpoint": checkpoint,
            "restore_checkpoint": configs.checkpoint.restore_checkpoint,
            "pid": os.getpid(),
        }
    )

    lossv = [0]
    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        if int(configs.checkpoint.resume) and len(configs.checkpoint.restore_checkpoint) > 0:
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )

            lg.info("Validate resumed model...")
            test_mode = getattr(configs.run, "test_mode", "mm")
            if test_mode == "sm":
                singleport_train_multiport_test(
                    model,
                    test_loader,
                    0,
                    criterion,
                    lossv,
                    device,
                    test_mixup_fn,
                    plot=True,
                    test_split=test_split,
                    test_random_state=configs.run.test_random_state,
                )
            elif test_mode == "mm":
                multiport_train_multiport_test(
                    model,
                    test_loader,
                    0,
                    criterion,
                    lossv,
                    device,
                    test_mixup_fn,
                    plot=True,
                    test_split=test_split,
                    test_random_state=configs.run.test_random_state,
                )
            elif test_mode == "feat":
                observe_features(
                    model,
                    test_loader,
                    0,
                    criterion,
                    lossv,
                    device,
                    test_mixup_fn,
                    plot=True,
                    test_split=test_split,
                    test_random_state=configs.run.test_random_state,
                )
            elif test_mode == "dyn":
                observe_dynamics(
                    model,
                    test_loader,
                    0,
                    criterion,
                    lossv,
                    device,
                    test_mixup_fn,
                    plot=True,
                    test_split=test_split,
                    test_random_state=configs.run.test_random_state,
                )
            elif test_mode == "dyn_spec":
                observe_dynamics_spectra(
                    model,
                    test_loader,
                    0,
                    criterion,
                    lossv,
                    device,
                    test_mixup_fn,
                    plot=True,
                    test_split=test_split,
                    test_random_state=configs.run.test_random_state,
                )

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
